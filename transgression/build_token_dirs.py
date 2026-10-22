#!/usr/bin/env python3
"""Fixed direction set for the token-level readout (GPT-OSS-120B), per block:
story probes from analyze.py (probe_dirs.npz), the 42 emotion vectors, the Goodfire reward-hacking direction and
100 seeded random unit directions. Writes data/transgression/token_dirs.npz ([n_blocks, n_dirs, H] fp16).

    python -m transgression.build_token_dirs --probe-dirs outputs/transgression/120b/probe_dirs.npz \
        --vectors ../emotion-ensemble-forensics/outputs/vectors/gptoss_120b --goodfire <hf>/goodfire_gptoss120b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from transgression.analyze import load_emotions, load_goodfire, unit

ROOT = Path(__file__).resolve().parents[1]
N_RAND = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dirs", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--goodfire", required=True)
    ap.add_argument("--out", default=str(ROOT / "data/transgression/token_dirs.npz"))
    a = ap.parse_args()
    P = np.load(a.probe_dirs)
    blocks = [int(b) for b in P["blocks"]]
    probe_names = sorted({k.rsplit("_", 1)[0] for k in P.files if k != "blocks"})
    emo = load_emotions(Path(a.vectors))
    gf = load_goodfire(Path(a.goodfire), blocks)
    H = P[f"dom_{blocks[0]}"].shape[0]
    rng = np.random.default_rng(12345)
    R = unit(rng.standard_normal((N_RAND, H)))               # same random set at every block
    names = probe_names + [f"emo:{e}" for e in sorted(emo)] + (["goodfire"] if gf else []) + \
        [f"rand:{i}" for i in range(N_RAND)]
    dirs = np.zeros((len(blocks), len(names), H), np.float16)
    for i, b in enumerate(blocks):
        rows = [unit(P[f"{p}_{b}"]) for p in probe_names] + [emo[e][b] for e in sorted(emo)]
        if gf:
            rows.append(gf[b])
        dirs[i] = np.concatenate([np.stack(rows), R]).astype(np.float16)
    np.savez_compressed(a.out, blocks=np.array(blocks), names=np.array(names), dirs=dirs)
    print(f"[dirs] {len(names)} directions x {len(blocks)} blocks -> {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.1f} MB); probes {probe_names}")


if __name__ == "__main__":
    main()
