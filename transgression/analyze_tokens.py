#!/usr/bin/env python3
"""Token-level allow-vs-forbid test on the LRU replays (outputs of transgression/token_readout.py; CPU, < 2 GB).

Targets (identical tokens in both conditions): sentence (first shortcut-naming reasoning sentence), action (cheat
tool call), action32 (its first 32 tokens), pre (generation-prompt token of the cheat turn).

1 fixed directions  per run: mean over target tokens of s(forbid) - s(allow), in units of the SD of s over all tokens
                    of that target (both conditions, all runs). Mean over runs with a bootstrap CI over runs; share of
                    runs > 0; p = share of the 100 random directions with |effect| >= |effect|.
2 fitted control    is the allow/forbid difference linearly visible at these tokens at all? Leave-one-run-out DoM
                    of (forbid - allow) span means; held-out run scored by (mean_F - mean_A) . w > 0. Sign accuracy
                    and a sign-flip permutation p. Also its cosine with the story probe.

    python -m transgression.analyze_tokens --tok outputs/hf/context_rot/transgression_gptoss120b_tokens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from transgression.analyze import boot_mean, r3, unit

ROOT = Path(__file__).resolve().parents[1]
SHOW = ("dom", "dom_perp", "dom@first", "dom@both", "logreg", "rule_text", "emo:guilty", "emo:ashamed",
        "emo:nervous", "emo:calm", "emo:desperate", "goodfire")
TARGETS = ("sentence", "action", "action32", "pre")


def load(tok_dir: Path):
    runs = {}
    for fa in sorted((tok_dir / "allow").glob("run-*.npz")):
        ff = tok_dir / "forbid" / fa.name
        if not ff.exists():
            continue
        A, F = np.load(fa), np.load(ff)
        r = {}
        for t in ("sentence", "action", "pre"):
            if f"{t}_proj" in A.files and f"{t}_proj" in F.files:
                if not np.array_equal(A[f"{t}_ids"], F[f"{t}_ids"]):
                    print(f"[tok] {fa.stem} {t}: token ids differ, skipped")
                    continue
                r[t] = {"A": A[f"{t}_proj"].astype(np.float32), "F": F[f"{t}_proj"].astype(np.float32),
                        "mA": A[f"{t}_mean"].astype(np.float64), "mF": F[f"{t}_mean"].astype(np.float64)}
        if "action" in r:
            a = r["action"]
            r["action32"] = {"A": a["A"][:32], "F": a["F"][:32]}
        runs[fa.stem] = r
        names = [str(n) for n in A["names"]]
        blocks = [int(b) for b in A["blocks"]]
    return runs, names, blocks


def fixed(runs, names, blocks, target):
    rs = [r[target] for r in runs.values() if target in r]
    allp = np.concatenate([np.concatenate([x["A"], x["F"]]) for x in rs])          # [tok, B, D]
    sd = allp.std(0) + 1e-12                                                       # [B, D]
    eff = np.stack([((x["F"] - x["A"]).mean(0)) / sd for x in rs])                # [runs, B, D]
    rand = [i for i, n in enumerate(names) if n.startswith("rand:")]
    out = {"n_runs": len(rs), "n_tokens_median": float(np.median([len(x["A"]) for x in rs])), "per_block": {}}
    for j, b in enumerate(blocks):
        E = eff[:, j]
        m = E.mean(0)
        null = np.abs(m[rand])
        pb = {}
        for i, n in enumerate(names):
            if n in SHOW:
                pb[n] = {"effect": float(m[i]), "ci": boot_mean(E[:, i]), "share_pos": float((E[:, i] > 0).mean()),
                         "p_vs_random": float((null >= abs(m[i])).mean())}
        pb["random"] = {"abs_p95": float(np.quantile(null, .95)), "abs_max": float(null.max())}
        out["per_block"][b] = pb
    return out


def fitted(runs, blocks, target, story_dirs, rng):
    rs = [r[target] for r in runs.values() if target in r and "mA" in r[target]]
    if len(rs) < 6:
        return None
    Dm = np.stack([x["mF"] - x["mA"] for x in rs])                                # [runs, B, H]
    out = {}
    for j, b in enumerate(blocks):
        X = Dm[:, j]
        n = len(X)
        tot = X.sum(0)
        held = np.array([X[i] @ unit((tot - X[i]) / (n - 1)) for i in range(n)])
        acc = float((held > 0).mean())
        # sign-flip permutation: random signs per run destroy any consistent forbid-allow direction
        null = []
        for _ in range(200):
            s = rng.choice([-1.0, 1.0], n)[:, None]
            Y = X * s
            t = Y.sum(0)
            null.append(np.mean([Y[i] @ unit((t - Y[i]) / (n - 1)) > 0 for i in range(n)]))
        rel = float(np.linalg.norm(X, axis=1).mean() / (np.linalg.norm(np.stack([x["mA"][j] for x in rs]), axis=1).mean() + 1e-12))
        w = unit(tot / n)
        out[b] = {"loro_sign_acc": acc, "perm_p": float((np.array(null) >= acc).mean()), "rel_diff_norm": rel,
                  "cos_with_story_dom": float(w @ story_dirs[b]) if b in story_dirs else None}
    return out


def figure(res, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tg = [t for t in TARGETS if t in res["fixed"]]
    fig, ax = plt.subplots(1, len(tg) + 1, figsize=(4.2 * (len(tg) + 1), 4))
    b = str(res["block"])
    for a, t in zip(ax, tg):
        pb = res["fixed"][t]["per_block"][b]
        ks = [k for k in SHOW if k in pb]
        v = [pb[k]["effect"] for k in ks]
        lo = [pb[k]["effect"] - pb[k]["ci"][0] for k in ks]
        hi = [pb[k]["ci"][1] - pb[k]["effect"] for k in ks]
        a.barh(range(len(ks)), v, xerr=[lo, hi], color=["C0" if not k.startswith("emo") and k != "goodfire" else "C3" for k in ks])
        p95 = pb["random"]["abs_p95"]
        a.axvspan(-p95, p95, color="0.85", zorder=0, label="random |95%|")
        a.set_yticks(range(len(ks)), ks, fontsize=7)
        a.axvline(0, c="k", lw=.5)
        a.set(title=f"{t} (n={res['fixed'][t]['n_runs']}), block {b}", xlabel="forbid - allow (SD)")
        a.legend(fontsize=7)
    a = ax[-1]
    for t in tg:
        if res["fitted"].get(t):
            bl = sorted(res["fitted"][t], key=int)
            a.plot([int(x) for x in bl], [res["fitted"][t][x]["loro_sign_acc"] for x in bl], "-o", ms=3, label=t)
    a.axhline(.5, c="0.5", lw=.5)
    a.set(title="fitted control: LORO sign accuracy", xlabel="block", ylim=(0, 1.02))
    a.legend(fontsize=7)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "tg_120b_tokens.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--probe-dirs", default=str(ROOT / "outputs/transgression/120b/probe_dirs.npz"))
    ap.add_argument("--block", type=int, default=16, help="main block (the story dom block)")
    ap.add_argument("--out", default=str(ROOT / "outputs/transgression/120b/tokens.json"))
    ap.add_argument("--figdir", default=str(ROOT / "docs/figures"))
    a = ap.parse_args()
    runs, names, blocks = load(Path(a.tok))
    P = np.load(a.probe_dirs) if Path(a.probe_dirs).exists() else None
    story = {b: unit(P[f"dom_{b}"]) for b in blocks if P is not None and f"dom_{b}" in P.files}
    block = a.block if a.block in blocks else blocks[len(blocks) // 2]
    res = {"block": block, "n_pairs": len(runs), "fixed": {}, "fitted": {}}
    rng = np.random.default_rng(0)
    for t in TARGETS:
        if any(t in r for r in runs.values()):
            res["fixed"][t] = fixed(runs, names, blocks, t)
            if t != "action32":
                res["fitted"][t] = fitted(runs, blocks, t, story, rng)
    res = r3(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    res = json.loads(json.dumps(res))
    figure(res, Path(a.figdir))
    for t, f in res["fixed"].items():
        pb = f["per_block"][str(block)]
        print(f"[tok] {t:9s} n={f['n_runs']} block {block} random|p95| {pb['random']['abs_p95']}: " +
              " ".join(f"{k}={pb[k]['effect']}(p{pb[k]['p_vs_random']})" for k in ("dom", "dom_perp", "dom@first", "emo:guilty", "goodfire") if k in pb))
        if res["fitted"].get(t):
            ft = res["fitted"][t]
            best = max(ft, key=lambda x: ft[x]["loro_sign_acc"])
            print(f"          fitted: block {block} acc {ft[str(block)]['loro_sign_acc']} p {ft[str(block)]['perm_p']} "
                  f"cos_story {ft[str(block)]['cos_with_story_dom']} | best block {best} acc {ft[best]['loro_sign_acc']} p {ft[best]['perm_p']}")
    print(f"[tok] wrote {a.out}")


if __name__ == "__main__":
    main()
