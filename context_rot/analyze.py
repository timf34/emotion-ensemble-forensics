#!/usr/bin/env python3
"""Emotion-probe trajectories around the cheat in GPT-OSS-120B optimize_cache_c rollouts.

Inputs: per-run readouts (run_readout.py: pre / reason residuals per assistant turn at the vector
layers), code-based cheat labels (labels.py: onset = turn whose tool call first writes the cheat),
gpt-oss-120b vectors (42 emotions, 6 syndromes, assistant axis; denoised, unit-normalised).

Turn k of the readout = the k-th assistant message = harness step-k (its tool call produced the
step-k workspace). tau = k - onset; tau = 0 is the turn that writes the cheat.
  pre    : residual at <|start|>assistant before turn k (state before deciding)
  reason : mean residual over turn k's reasoning (state while deciding)

Analyses (all per probe, at one layer):
  1 event-locked mean z, tau in [-8, +4], cheat runs, next to honest runs at the same step numbers
  2 drift-controlled effects: z ~ run offset + step spline (all runs) + window(tau in [-2,0]) + post(tau>0);
    cluster bootstrap over runs; BH over probes
  3 pre-cheat build-up: slope over tau in [-8,-1] minus the honest runs' slope over the same steps
  4 prediction of "cheat within the next 3 turns" (turns before onset + honest runs), leave-one-run-out
  5 principal components of the 42 emotion scores: loadings and event-locked trajectories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = [6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 26, 28, 30, 32, 34, 35]
KEY = ["desperate", "frustrated", "stuck", "trapped", "stressed", "anxious", "nervous", "guilty", "ashamed",
       "calm", "hopeful", "self-confident", "content", "resigned"]


def load_vectors(vec_dir: Path, layer: int):
    import torch

    def unit(M):
        M = np.asarray(M, np.float64)
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    emo = torch.load(vec_dir / "vectors_emotions_dn.pt", map_location="cpu", weights_only=True)
    syn = torch.load(vec_dir / "vectors_syndromes_dn.pt", map_location="cpu", weights_only=True)
    ax = torch.load(vec_dir / "axis_reencoded" / "assistant_axis.pt", map_location="cpu", weights_only=False)
    names = list(emo) + [f"syn:{k}" for k in syn] + ["assistant_axis"]
    M = np.stack([emo[k][layer].float().numpy() for k in emo] + [syn[k][layer].float().numpy() for k in syn]
                 + [ax[layer].float().numpy()])
    return names, unit(M), len(emo)


def load_runs(acts: Path, labels: list[dict], layer: int, rep: str):
    lab = {l["run"]: l for l in labels}
    out = []
    for f in sorted(acts.glob("run-*.npz"), key=lambda p: int(p.stem.split("-")[1])):
        if f.stem not in lab:
            continue
        z = np.load(f, allow_pickle=False)
        j = list(z["blocks"]).index(layer)
        out.append({"run": f.stem, "X": z[rep][:, j, :].astype(np.float32), **lab[f.stem]})
    return out


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def spline_basis(step, knots=(5, 10, 15, 20, 30, 40)):
    s = np.asarray(step, float)
    cols = [s] + [np.maximum(0, s - k) for k in knots]
    return np.stack(cols, 1)


def window_effects(rows, Z, n_boot=500, seed=0):
    """Per-probe coefficients of window (tau in [-2,0]) and post (tau>0) after run offsets + step spline."""
    run = np.array([r["run"] for r in rows])
    step = np.array([r["step"] for r in rows])
    tau = np.array([r["tau"] if r["tau"] is not None else -999 for r in rows])
    win = ((tau >= -2) & (tau <= 0)).astype(float)
    post = (tau > 0).astype(float)
    B = spline_basis(step)
    design = np.column_stack([win, post, B])

    def fit(idx):
        # within-run demeaning = run fixed effects
        Xd = design[idx].copy()
        Yd = Z[idx].copy()
        r = run[idx]
        for u in np.unique(r):
            m = r == u
            Xd[m] -= Xd[m].mean(0)
            Yd[m] -= Yd[m].mean(0)
        coef, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
        return coef[:2]                       # [2, n_probes]

    est = fit(np.arange(len(rows)))
    rng = np.random.default_rng(seed)
    runs = np.unique(run)
    idx_by = {u: np.flatnonzero(run == u) for u in runs}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(runs, size=len(runs), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        # relabel duplicated runs so they get their own offsets
        boots.append(fit(idx))
    boots = np.stack(boots)                   # [n_boot, 2, n]
    se = boots.std(0)
    p = 2 * np.minimum((boots > 0).mean(0), (boots < 0).mean(0))
    return est, se, np.maximum(p, 1 / n_boot)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def prediction(rows, Z, n_emo, rand_feats):
    """Leave-one-run-out logistic regression for y = cheat within the next 3 turns (tau in [-3,0]),
    on turns up to the onset (cheat runs) and all turns (honest runs). rand_feats: list of [n_rows, 42]
    z-scored projections on random directions in the residual space (one per seed)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = np.array([(r["tau"] is None) or (r["tau"] <= 0) for r in rows])
    run = np.array([r["run"] for r in rows])[keep]
    y = np.array([int(r["tau"] is not None and -3 <= r["tau"] <= 0) for r in rows])[keep]
    step = spline_basis(np.array([r["step"] for r in rows], float)[keep])
    E = Z[keep][:, :n_emo]

    def loro(F):
        pred = np.zeros(len(y))
        for u in np.unique(run):
            te = run == u
            tr = ~te
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
            m.fit(F[tr], y[tr])
            pred[te] = m.decision_function(F[te])
        return auroc(y, pred)

    out = {"step only": {"auroc": loro(step)}, "emotions (42)": {"auroc": loro(E)},
           "emotions + step": {"auroc": loro(np.column_stack([E, step]))}}
    ra = [loro(np.column_stack([R[keep], step])) for R in rand_feats]
    out["random 42 dirs + step"] = {"auroc": float(np.mean(ra)), "sd": float(np.std(ra))}
    out["n_turns"] = int(len(y))
    out["n_positive"] = int(y.sum())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acts", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--vectors", required=True, help="dir with vectors_emotions_dn.pt, vectors_syndromes_dn.pt, axis_reencoded/")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 14, 20])
    args = p.parse_args()
    labels = json.load(open(args.labels))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for layer in args.layers:
        names, U, n_emo = load_vectors(Path(args.vectors), layer)
        for rep in ("reason", "pre"):
            runs = load_runs(Path(args.acts), labels, layer, rep)
            rows, S, XR = [], [], []
            rng = np.random.default_rng(0)
            H = runs[0]["X"].shape[1]
            Rdirs = [rng.standard_normal((42, H)) for _ in range(5)]
            Rdirs = [R / np.linalg.norm(R, axis=1, keepdims=True) for R in Rdirs]
            for r in runs:
                k_final = r["X"].shape[0] - 1             # last turn = the final report (no tool call)
                for k in range(r["X"].shape[0]):
                    tau = (k - r["onset"]) if r["cheated"] else None
                    rows.append({"run": r["run"], "step": k, "tau": tau, "cheated": r["cheated"],
                                 "final_turn": k == k_final})
                S.append(r["X"].astype(np.float64) @ U.T)
                XR.append([r["X"].astype(np.float64) @ R.T for R in Rdirs])
            S = np.concatenate(S)
            rand_feats = [np.concatenate([x[s_] for x in XR]) for s_ in range(len(Rdirs))]
            rand_feats = [(F - F.mean(0)) / F.std(0) for F in rand_feats]
            Z = (S - S.mean(0)) / S.std(0)
            # 1 event-locked
            taus = list(range(-8, 5))
            ev, ev_se, ev_n = {}, {}, {}
            for i, n in enumerate(names):
                ev[n], ev_se[n], ev_n[n] = [], [], []
                for t in taus:
                    v = [Z[j, i] for j, r in enumerate(rows) if r["tau"] == t]
                    ev[n].append(float(np.mean(v)) if v else None)
                    ev_se[n].append(float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None)
                    ev_n[n].append(len(v))
            honest_by_step = {}
            for j, r in enumerate(rows):
                if not r["cheated"]:
                    honest_by_step.setdefault(r["step"], []).append(j)
            # honest runs at the same absolute step numbers as cheat runs' tau positions
            hon = {}
            for i, n in enumerate(names):
                vals = []
                for t in taus:
                    js = [honest_by_step.get(r2["onset"] + t, []) for r2 in runs if r2["cheated"]]
                    js = [j for lst in js for j in lst]
                    vals.append(float(np.mean(Z[js, i])) if js else None)
                hon[n] = vals
            # 2 drift-controlled window effects
            est, se, pv = window_effects(rows, Z)
            q_win, q_post = bh(pv[0]), bh(pv[1])
            eff = {n: {"window": float(est[0, i]), "window_se": float(se[0, i]), "window_q": float(q_win[i]),
                       "post": float(est[1, i]), "post_se": float(se[1, i]), "post_q": float(q_post[i])}
                   for i, n in enumerate(names)}
            # 3 build-up slope, tau -8..-1 vs honest runs over the same steps
            build = {}
            for i, n in enumerate(names):
                d = []
                for r2 in runs:
                    if not r2["cheated"] or r2["onset"] < 6:
                        continue
                    ks = list(range(max(0, r2["onset"] - 8), r2["onset"]))
                    js = [j for j, rr in enumerate(rows) if rr["run"] == r2["run"] and rr["step"] in ks]
                    if len(js) >= 3:
                        sl = np.polyfit([rows[j]["step"] for j in js], Z[js, i], 1)[0]
                        hj = [j for k in ks for j in honest_by_step.get(k, [])]
                        hs = np.polyfit([rows[j]["step"] for j in hj], Z[hj, i], 1)[0] if len(set(rows[j]["step"] for j in hj)) >= 3 else 0.0
                        d.append(sl - hs)
                build[n] = {"mean_slope_diff": float(np.mean(d)) if d else None,
                            "se": float(np.std(d) / np.sqrt(len(d))) if len(d) > 1 else None, "n_runs": len(d)}
            # 4 prediction
            pred = prediction(rows, Z, n_emo, rand_feats)
            # 5 PCA of emotion scores
            E = Z[:, :n_emo]
            u, sv, vt = np.linalg.svd(E - E.mean(0), full_matrices=False)
            var = (sv ** 2 / (sv ** 2).sum())[:5]
            pcs = {}
            for c in range(3):
                load = vt[c]
                top = np.argsort(-np.abs(load))[:6]
                sc = E @ load
                pcs[f"PC{c + 1}"] = {"var": float(var[c]), "top": {names[t]: round(float(load[t]), 2) for t in top},
                                     "event_locked": [float(np.mean([sc[j] for j, r in enumerate(rows) if r["tau"] == t])) if any(r["tau"] == t for r in rows) else None for t in taus]}
            key = f"L{layer}_{rep}"
            summary[key] = {"n_runs": len(runs), "n_cheat_runs": sum(r["cheated"] for r in runs), "taus": taus,
                            "event_locked": ev, "event_locked_se": ev_se, "event_locked_n": ev_n, "honest_same_steps": hon, "effects": eff, "buildup": build,
                            "prediction": pred, "pca": pcs}
            top = sorted(eff, key=lambda n: eff[n]["window_q"])[:8]
            print(f"\n=== layer {layer} {rep}: {len(runs)} runs ({summary[key]['n_cheat_runs']} cheat) | prediction "
                  + ", ".join(f"{k} {v['auroc']:.2f}" for k, v in pred.items() if isinstance(v, dict) and v.get("auroc") is not None))
            for n in top:
                e = eff[n]
                print(f"  {n:24s} window {e['window']:+.2f}±{e['window_se']:.2f} (q={e['window_q']:.3f})   "
                      f"post {e['post']:+.2f}±{e['post_se']:.2f} (q={e['post_q']:.3f})   build {build[n]['mean_slope_diff']}")
            for c, v in pcs.items():
                print(f"  {c} ({v['var']:.0%}): {v['top']}")
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    print(f"\n-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
