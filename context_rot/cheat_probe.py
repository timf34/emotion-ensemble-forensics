#!/usr/bin/env python3
"""Option 1: a cheat probe trained directly on our own rollouts vs the emotion probes.

Target: y = 1 on the turns that start the cheat or come up to 3 turns before it (tau in [-3, 0]), 0 on
earlier turns of cheating runs and on the turns of non-cheating runs. Turns after the onset are dropped;
for runs that were cut off mid-work (labels.json truncated) the last 3 turns are dropped too (their
future is unknown). Every method sees the same rows and is evaluated leave-one-run-out; anything fitted
(standardisation, PCA, the cheat direction, regularisation) is fitted on the training runs only.

  step            step-number spline only
  emotions        42 emotion scores (unit vectors) + step
  random          42 random unit directions + step (5 seeds)
  pca             42 training-fitted principal components + step
  dom             difference-of-means cheat direction (training runs) -> one score + step
  logistic        full activation (standardised) + step, L2, C by inner grouped CV

    python context_rot/cheat_probe.py --acts <readouts> --labels <labels_code.json> --vectors <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import LAYERS, load_runs, load_vectors, spline_basis  # noqa: E402


def rows_for(runs):
    X, y, g, step = [], [], [], []
    for r in runs:
        n = r["X"].shape[0]
        last_ok = n - 1 - (3 if r.get("truncated") and not r["cheated"] else 0)
        for k in range(n):
            if r["cheated"]:
                if k > r["onset"]:
                    continue
                lab = int(r["onset"] - 3 <= k <= r["onset"])
            else:
                if k > last_ok:
                    continue
                lab = 0
            X.append(r["X"][k]); y.append(lab); g.append(r["run"]); step.append(k)
    return np.array(X, np.float64), np.array(y), np.array(g), np.array(step, float)


def _lr(C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def boot_ci(y, s, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    by = {u: np.flatnonzero(g == u) for u in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[u] for u in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[idx])) == 2:
            vals.append(auroc(y[idx], s[idx]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def loro(feature_fn, X, y, g, step, inner_C=False):
    """feature_fn(X_tr, y_tr, X_te) -> (F_tr, F_te); a logistic head on [F, step spline] per held-out run."""
    from sklearn.model_selection import GroupKFold

    S = spline_basis(step)
    pred = np.zeros(len(y))
    for u in np.unique(g):
        te, tr = g == u, g != u
        F_tr, F_te = feature_fn(X[tr], y[tr], X[te])
        A_tr = np.column_stack([F_tr, S[tr]]) if F_tr is not None else S[tr]
        A_te = np.column_stack([F_te, S[te]]) if F_te is not None else S[te]
        C = 0.1
        if inner_C:
            best = -1
            for c in (1e-3, 1e-2, 1e-1):
                ps = np.zeros(tr.sum())
                gtr = g[tr]
                for itr, iva in GroupKFold(5).split(A_tr, y[tr], gtr):
                    m = _lr(c).fit(A_tr[itr], y[tr][itr])
                    ps[iva] = m.decision_function(A_tr[iva])
                a = auroc(y[tr], ps)
                if a > best:
                    best, C = a, c
        m = _lr(C).fit(A_tr, y[tr])
        pred[te] = m.decision_function(A_te)
    return pred


def methods(U, seeds=5, k=42):
    from sklearn.decomposition import PCA

    out = {"step": lambda Xtr, ytr, Xte: (None, None),
           "emotions": lambda Xtr, ytr, Xte: (Xtr @ U.T, Xte @ U.T)}
    rng = np.random.default_rng(0)
    H = U.shape[1]
    rand = [rng.standard_normal((k, H)) for _ in range(seeds)]
    rand = [R / np.linalg.norm(R, axis=1, keepdims=True) for R in rand]
    for i, R in enumerate(rand):
        out[f"random_s{i}"] = (lambda R: lambda Xtr, ytr, Xte: (Xtr @ R.T, Xte @ R.T))(R)

    def pca(Xtr, ytr, Xte):
        p = PCA(n_components=k, random_state=0).fit(Xtr)
        return p.transform(Xtr), p.transform(Xte)

    def dom(Xtr, ytr, Xte):
        d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        d /= np.linalg.norm(d) + 1e-12
        return (Xtr @ d)[:, None], (Xte @ d)[:, None]

    out["pca"] = pca
    out["dom"] = dom
    out["logistic"] = lambda Xtr, ytr, Xte: (Xtr, Xte)
    return out


def run_one(acts, labels, vec_dir, layer, rep, which=None):
    names, U_all, n_emo = load_vectors(vec_dir, layer)
    U = U_all[:n_emo]
    lab = {l["run"]: l for l in labels}
    runs = load_runs(acts, labels, layer, rep)
    for r in runs:
        r["truncated"] = lab[r["run"]].get("truncated", False)
    X, y, g, step = rows_for(runs)
    res = {"n_rows": int(len(y)), "n_pos": int(y.sum()), "n_runs": int(len(np.unique(g)))}
    for name, fn in methods(U).items():
        if which and not any(name.startswith(w) for w in which):
            continue
        s = loro(fn, X, y, g, step, inner_C=(name == "logistic"))
        res[name] = {"auroc": auroc(y, s), "ci95": boot_ci(y, s, g)}
    rs = [v["auroc"] for k, v in res.items() if k.startswith("random_s")]
    if rs:
        res["random"] = {"auroc": float(np.mean(rs)), "sd": float(np.std(rs))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 14, 20])
    ap.add_argument("--sweep", action="store_true", help="all vector layers, emotions / dom / random / step only")
    a = ap.parse_args()
    labels = json.load(open(a.labels))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for layer in a.layers:
        for rep in ("pre", "reason"):
            r = run_one(Path(a.acts), labels, Path(a.vectors), layer, rep)
            summary[f"L{layer}_{rep}"] = r
            print(f"L{layer:2d} {rep:6s} rows {r['n_rows']} pos {r['n_pos']} | " + "  ".join(
                f"{m} {r[m]['auroc']:.2f}" + (f" [{r[m]['ci95'][0]:.2f},{r[m]['ci95'][1]:.2f}]" if "ci95" in r[m] else "")
                for m in ("step", "emotions", "random", "pca", "dom", "logistic")), flush=True)
    if a.sweep:
        for layer in LAYERS:
            for rep in ("pre", "reason"):
                key = f"L{layer}_{rep}"
                if key in summary:
                    continue
                summary[key] = run_one(Path(a.acts), labels, Path(a.vectors), layer, rep,
                                       which=("step", "emotions", "random", "dom"))
                r = summary[key]
                print(f"sweep L{layer:2d} {rep:6s}: " + "  ".join(f"{m} {r[m]['auroc']:.2f}"
                                                                  for m in ("step", "emotions", "random", "dom")), flush=True)
    json.dump(summary, open(out / "cheat_probe.json", "w"), indent=1)
    print("->", out / "cheat_probe.json")


if __name__ == "__main__":
    main()
