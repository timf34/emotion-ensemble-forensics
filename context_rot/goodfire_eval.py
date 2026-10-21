#!/usr/bin/env python3
"""Option 2: the independent reward-hacking direction (from Goodfire's labelled rollouts) vs emotion probes.

1. Direction per layer and representation: mean over hack steps minus mean over honest steps of the
   Goodfire readouts (run_goodfire.py), unit-normalised. Validated inside Goodfire by leave-one-source-
   model-out: build it on two source models' rollouts, score the third's steps (AUROC hack vs honest).
2. Applied, with NO fitting on the test data, to a test set; compared with single pre-chosen emotions and,
   with a logistic head under the same leave-one-run-out folds, with the emotion ensemble and random
   directions (cheat_probe.py rows).

  test set "lru"  : our optimize_cache_c readouts (pre / reason per turn), target = cheat within 3 turns
  test set "srh"  : the SRH pilot (Qwen3.6, last / resp per response), target = hack (scored prompts)

    python context_rot/goodfire_eval.py --gf <goodfire npz dir> --test lru --acts <readouts> --labels <labels>
        --vectors <emotion vector dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cheat_probe import auroc, boot_ci, loro, rows_for  # noqa: E402

PRESET = ("desperate", "dispirited", "frustrated", "stuck", "calm", "guilty")


def load_gf(gf_dir: Path, block: int, rep: str):
    X, lab, src = [], [], []
    for f in sorted(gf_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        j = list(z["blocks"]).index(block)
        steps = json.loads(str(z["steps"]))
        A = z["step" if rep in ("step", "reason", "resp") else "pre"][:, j, :].astype(np.float64)
        for k, s in enumerate(steps):
            if s["label"] in ("hack", "honest"):
                X.append(A[k]); lab.append(int(s["label"] == "hack")); src.append(f.name.split("__")[0])
    return np.array(X), np.array(lab), np.array(src)


def dom(X, y):
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def validate(X, y, src):
    out = {}
    for s in np.unique(src):
        tr, te = src != s, src == s
        out[s] = auroc(y[te], X[te] @ dom(X[tr], y[tr]))
    return out


def load_emotions(vec_dir: Path, block: int):
    import torch

    emo = torch.load(vec_dir / "vectors_emotions_dn.pt", map_location="cpu", weights_only=True)
    names = list(emo)
    U = np.stack([emo[k][block].float().numpy() for k in names]).astype(np.float64)
    return names, U / np.linalg.norm(U, axis=1, keepdims=True)


def eval_lru(a, block, gf_rep, test_rep):
    from analyze import load_runs

    labels = json.load(open(a.labels))
    lab = {l["run"]: l for l in labels}
    runs = load_runs(Path(a.acts), labels, block, test_rep)
    for r in runs:
        r["truncated"] = lab[r["run"]].get("truncated", False)
    X, y, g, step = rows_for(runs)
    Xg, yg, sg = load_gf(Path(a.gf), block, gf_rep)
    d = dom(Xg, yg)
    names, U = load_emotions(Path(a.vectors), block)
    res = {"n_rows": int(len(y)), "n_pos": int(y.sum()), "gf_steps": int(len(yg)), "gf_hack": int(yg.sum()),
           "gf_validation_loso": validate(Xg, yg, sg)}
    # no fitting on the test data at all
    res["raw"] = {"goodfire_direction": auroc(y, X @ d)}
    for e in PRESET:
        res["raw"][e] = auroc(y, X @ U[names.index(e)])
    # same LORO logistic head as cheat_probe (+ step spline)
    rng = np.random.default_rng(0)
    R = rng.standard_normal((42, X.shape[1])); R /= np.linalg.norm(R, axis=1, keepdims=True)
    feats = {"step": lambda Xtr, ytr, Xte: (None, None),
             "goodfire+step": lambda Xtr, ytr, Xte: ((Xtr @ d)[:, None], (Xte @ d)[:, None]),
             "emotions+step": lambda Xtr, ytr, Xte: (Xtr @ U.T, Xte @ U.T),
             "emotions+goodfire+step": lambda Xtr, ytr, Xte: (np.column_stack([Xtr @ U.T, Xtr @ d]),
                                                              np.column_stack([Xte @ U.T, Xte @ d])),
             "random42+step": lambda Xtr, ytr, Xte: (Xtr @ R.T, Xte @ R.T)}
    for k, fn in feats.items():
        s = loro(fn, X, y, g, step)
        res[k] = {"auroc": auroc(y, s), "ci95": boot_ci(y, s, g)}
    return res


def eval_srh(a, block, gf_rep, test_rep):
    """SRH pilot (Qwen3.6): hack vs not among scored responses; the emotion ensemble and the Goodfire
    direction are compared by AUROC with prompt-grouped CV (no family hold-out here: raw directions need none)."""
    from sklearn.model_selection import GroupKFold

    from cheat_probe import _lr

    d_act = Path(a.acts)
    idx = [json.loads(l) for l in open(d_act / "index.jsonl")]
    meta = json.loads((d_act / "meta.json").read_text())
    j = meta["blocks"].index(block)
    A = np.load(d_act / f"{test_rep}.npy", mmap_mode="r")
    lab = {json.loads(l)["id"]: json.loads(l) for l in open(a.labels)}
    rows = [i for i, r in enumerate(idx) if r.get("variant") == "scored" and lab.get(r["id"], {}).get("hack") is not None]
    X = np.asarray(A[rows, j, :], dtype=np.float64)
    y = np.array([lab[idx[i]["id"]]["hack"] for i in rows])
    g = np.array([idx[i]["task_id"] for i in rows])
    fam = np.array([idx[i]["exploit_family"] for i in rows])
    emo_metric = np.array([bool(idx[i].get("emotion_metric")) for i in rows])
    Xg, yg, sg = load_gf(Path(a.gf), block, gf_rep)
    d = dom(Xg, yg)
    names, U = load_emotions(Path(a.vectors), block)
    res = {"n": int(len(y)), "n_hack": int(y.sum()), "gf_steps": int(len(yg)), "gf_hack": int(yg.sum()),
           "gf_validation_loso": validate(Xg, yg, sg), "raw": {"goodfire_direction": auroc(y, X @ d)},
           "raw_excluding_emotion_metric": {"goodfire_direction": auroc(y[~emo_metric], (X @ d)[~emo_metric])}}
    for e in PRESET:
        res["raw"][e] = auroc(y, X @ U[names.index(e)])
    res["raw_by_family"] = {f: auroc(y[fam == f], (X @ d)[fam == f]) for f in np.unique(fam)}

    def cv(F):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F, y, g):
            p[te] = _lr(0.1).fit(F[tr], y[tr]).decision_function(F[te])
        return {"auroc": auroc(y, p), "ci95": boot_ci(y, p, g)}

    res["cv_emotions"] = cv(X @ U.T)
    res["cv_goodfire"] = cv((X @ d)[:, None])
    res["cv_emotions+goodfire"] = cv(np.column_stack([X @ U.T, X @ d]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gf", required=True)
    ap.add_argument("--test", choices=["lru", "srh"], required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = {}
    pairs = [("step", "reason"), ("pre", "pre")] if a.test == "lru" else [("pre", "last"), ("step", "resp")]
    for block in a.blocks:
        for gf_rep, test_rep in pairs:
            r = (eval_lru if a.test == "lru" else eval_srh)(a, block, gf_rep, test_rep)
            out[f"L{block}_{test_rep}"] = r
            print(f"L{block} {test_rep:6s} | goodfire within-Goodfire LOSO "
                  f"{ {k[:22]: round(v, 2) for k, v in r['gf_validation_loso'].items()} } | raw "
                  f"{ {k: round(v, 2) for k, v in r['raw'].items()} }", flush=True)
            fit = {k: round(v['auroc'], 2) for k, v in r.items() if isinstance(v, dict) and 'auroc' in v}
            print(f"     fitted: {fit}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / f"goodfire_eval_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
