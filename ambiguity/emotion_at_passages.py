"""Which emotion probes light up at the ambiguous passages the miner found? (GPT-OSS local runs only)

    python -m ambiguity.emotion_at_passages \
        --acts outputs/hf_efor/context_rot/gptoss120b_lru_main3 --prefix cache120_main3 \
        --vectors outputs/hf_vectors/gptoss_120b [--layer 14] [--rep reason] [--min-strength 3]

Activations: per-step readouts from context_rot/run_readout.py (HF timf34/emotion-ensemble-forensics,
context_rot/<tag>/run-N.npz; step k = k-th assistant turn = the miner's step k, checked by step count).
Vectors: HF timf34/emotion-vectors <model>/vectors_emotions_dn.pt (42 emotions, unit-normalised here).

Score = residual . unit vector. Each emotion's scores over every (run, reasoning step) in the readout set are
cleaned of run offsets and step-number drift (alternating demeaning, step bins of 3), then divided by their
SD. A passage group's mean is compared with a null that redraws, for each passage, another reasoning step of
the SAME run (10k draws), so run identity is controlled; two-sided p, BH over the 42 emotions.
Groups:
  ambiguous  passages with strength >= --min-strength, not resolved later, in runs the miner calls misbehaving
  onset      each misbehaving run's first misbehaviour step (whether or not a passage sits there)
Prints aggregates and per-passage top emotions (ids + steps only; these are Tim's own GPT-OSS rollouts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ambiguity.mine import OUT


def unit_vectors(vec_dir: Path, layer: int):
    import torch
    emo = torch.load(vec_dir / "vectors_emotions_dn.pt", map_location="cpu", weights_only=True)
    names = list(emo)
    M = np.stack([emo[k][layer].float().numpy() for k in names]).astype(np.float64)
    return names, M / np.linalg.norm(M, axis=1, keepdims=True)


def clean(rows: list[tuple[str, int]], S: np.ndarray, iters: int = 10) -> np.ndarray:
    """Remove run offsets and step drift from S (n_obs x n_emo) by alternating demeaning; return z."""
    runs = np.array([r for r, _ in rows])
    sbin = np.array([s // 3 for _, s in rows])
    R = S - S.mean(0)
    for _ in range(iters):
        for key in (runs, sbin):
            for k in np.unique(key):
                m = key == k
                R[m] -= R[m].mean(0)
    return R / R.std(0)


def group_test(idx: list[int], rows, Z, pool_by_run, n_draw=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = Z[idx].mean(0)
    null = np.zeros((n_draw, Z.shape[1]))
    for i in idx:
        pool = pool_by_run[rows[i][0]]
        null += Z[rng.choice(pool, n_draw)]
    null /= len(idx)
    p = (np.abs(null - null.mean(0)) >= np.abs(obs - null.mean(0))).mean(0)
    return obs, np.maximum(p, 1 / n_draw)


def bh(p):
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--prefix", required=True, help="sample-id prefix of these runs, e.g. cache120_main3")
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--mined", default=str(OUT / "mined" / "google_gemini-4-argon"))
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rep", default="reason", choices=["reason", "pre"])
    ap.add_argument("--min-strength", type=int, default=3)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    names, U = unit_vectors(Path(a.vectors), a.layer)
    asg_p = Path(a.mined).parent / "assignments.json"
    asg = json.loads(asg_p.read_text()) if asg_p.exists() else {}

    rows, feats = [], []
    for f in sorted(Path(a.acts).glob("run-*.npz"), key=lambda p: int(p.stem.split("-")[1])):
        z = np.load(f)
        j = list(z["blocks"]).index(a.layer)
        X = z[a.rep][:, j, :].astype(np.float64)
        ok = z["reason_tokens"] > 0 if a.rep == "reason" else np.ones(len(X), bool)
        for s in np.flatnonzero(ok):
            rows.append((f.stem, int(s)))
            feats.append(X[s])
    S = np.stack(feats) @ U.T
    Z = clean(rows, S)
    at = {r: i for i, r in enumerate(rows)}
    pool_by_run: dict[str, list[int]] = {}
    for i, (r, _) in enumerate(rows):
        pool_by_run.setdefault(r, []).append(i)

    amb, onset, detail = [], [], []
    for p in sorted(Path(a.mined).glob(f"{a.prefix}__run-*.json")):
        d = json.loads(p.read_text())
        run = p.stem.split("__")[1]
        if d.get("misbehaved") not in ("yes", "partial"):
            continue
        ms = [int(s) for s in d.get("misbehaviour_steps") or [] if str(s).isdigit()]
        if ms and (run, min(ms)) in at:
            onset.append(at[(run, min(ms))])
        for i, x in enumerate(d.get("passages") or []):
            k = (run, int(x.get("step", -1)))
            if x.get("strength", 0) >= a.min_strength and not x.get("resolved_later") and k in at:
                amb.append(at[k])
                cat = asg.get(f"{a.prefix}/{run}#{i}", {}).get("primary", "?")
                detail.append((run, k[1], x["strength"], d.get("verbalized_transgression"), cat, at[k]))

    print(f"## {a.prefix}: layer {a.layer}, rep={a.rep}; background {len(rows)} reasoning steps in "
          f"{len(pool_by_run)} runs; {len(amb)} ambiguous passages (strength >= {a.min_strength}) in "
          f"{len({d[0] for d in detail})} runs; {len(onset)} onset steps\n")
    for label, idx in (("ambiguous passages", amb), ("misbehaviour onset", onset)):
        if len(idx) < 3:
            print(f"{label}: too few ({len(idx)})\n")
            continue
        obs, p = group_test(idx, rows, Z, pool_by_run)
        q = bh(p)
        o = np.argsort(-obs)
        print(f"### {label} (n={len(idx)}): mean z vs same-run null")
        print("| emotion | mean z | p | BH q |\n|---|---|---|---|")
        for k in list(o[:a.top]) + list(o[-a.top:]):
            print(f"| {names[k]} | {obs[k]:+.2f} | {p[k]:.3f} | {q[k]:.2f} |")
        print(f"(min BH q over 42 emotions: {q.min():.2f})\n")
    print("### per passage: top 3 / bottom 2 emotions (z)")
    for run, st, s, v, cat, i in detail:
        o = np.argsort(-Z[i])
        top = ", ".join(f"{names[k]} {Z[i, k]:+.1f}" for k in o[:3])
        bot = ", ".join(f"{names[k]} {Z[i, k]:+.1f}" for k in o[-2:])
        print(f"- {a.prefix}/{run} step {st} (s{s}, verbal={v}, {cat}): {top} | {bot}")


if __name__ == "__main__":
    main()
