#!/usr/bin/env python3
"""Grouped emotion figures around the cheat (GPT-OSS-120B, optimize_cache_c).

Group score = mean of the member emotions' z-scores (z over all turns of all runs).
  fig_groups_<layer>.png   2 rows (before the turn / during its reasoning) x 5 groups: mean over the cheating
                           runs at each turn offset (+-1 SE across runs) and the honest runs at the same steps
  fig_effects_<layer>.png  drift-controlled change per group, just before the cheat (t-2..0) and after it
                           (t>0), run offsets + step-number spline removed, 95% bootstrap intervals over runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_runs, load_vectors, window_effects  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUPS = {
    "Defeat": ["dispirited", "stuck", "worthless", "gloomy", "miserable", "melancholy"],
    "Anxiety": ["anxious", "nervous", "panicked", "afraid", "overwhelmed", "hysterical"],
    "Frustration / pressure": ["frustrated", "exasperated", "desperate", "stressed"],
    "Guilt": ["guilty", "ashamed", "sorry", "regretful"],
    "Calm / relief": ["calm", "content", "hopeful"],
}
CHEAT, HONEST, BEFORE, AFTER = "#2a78d6", "#8a8980", "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#6b6a65", "#e4e3dd"
TAUS = list(range(-8, 5))


def group_scores(acts, labels, vec_dir, layer, rep):
    names, U, n_emo = load_vectors(vec_dir, layer)
    runs = load_runs(acts, labels, layer, rep)
    rows, S = [], []
    for r in runs:
        for k in range(r["X"].shape[0]):
            rows.append({"run": r["run"], "step": k, "tau": (k - r["onset"]) if r["cheated"] else None,
                         "cheated": r["cheated"], "onset": r["onset"]})
        S.append(r["X"].astype(np.float64) @ U.T)
    S = np.concatenate(S)
    Z = (S - S.mean(0)) / S.std(0)
    G = np.stack([Z[:, [names.index(e) for e in mem]].mean(1) for mem in GROUPS.values()], 1)
    return rows, G, runs


def event_locked(rows, G, runs):
    tau = np.array([r["tau"] if r["tau"] is not None else -999 for r in rows])
    step = np.array([r["step"] for r in rows])
    honest = np.array([not r["cheated"] for r in rows])
    onsets = [r["onset"] for r in runs if r["cheated"]]
    m, se, n, h = [], [], [], []
    for t in TAUS:
        sel = tau == t
        m.append(G[sel].mean(0) if sel.any() else np.full(G.shape[1], np.nan))
        se.append(G[sel].std(0, ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else np.zeros(G.shape[1]))
        n.append(int(sel.sum()))
        hs = honest & np.isin(step, [o + t for o in onsets])
        h.append(G[hs].mean(0) if hs.any() else np.full(G.shape[1], np.nan))
    return np.array(m), np.array(se), n, np.array(h)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_groups(acts, labels, vec_dir, layer, out):
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(15, 6.4), sharex=True, sharey=True)
    for row, (rep, rep_label) in enumerate([("pre", "Just before the turn"), ("reason", "During the turn's reasoning")]):
        rows, G, runs = group_scores(acts, labels, vec_dir, layer, rep)
        m, se, n, h = event_locked(rows, G, runs)
        for j, (gname, mem) in enumerate(GROUPS.items()):
            ax = axes[row, j]
            style(ax)
            ax.axvspan(-0.5, 0.5, color="#f2f1ec", zorder=0)
            ax.axhline(0, color=GRID, lw=1, zorder=0)
            ax.fill_between(TAUS, m[:, j] - se[:, j], m[:, j] + se[:, j], color=CHEAT, alpha=0.15, lw=0)
            ax.plot(TAUS, m[:, j], color=CHEAT, lw=2.2, marker="o", ms=4, label="cheating runs (41)")
            ax.plot(TAUS, h[:, j], color=HONEST, lw=1.6, ls="--", label="honest runs (9), same step numbers")
            if row == 0:
                ax.set_title(f"{gname}\n\n", fontsize=11, color=INK, fontweight="bold", loc="left")
                ax.text(0, 1.02, ",\n".join(", ".join(mem[i:i + 3]) for i in range(0, len(mem), 3)),
                        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
            if j == 0:
                ax.set_ylabel(f"{rep_label}\nmean z-score", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("turns relative to the cheat", fontsize=8.5, color=MUTED)
            ax.set_xticks([-8, -4, 0, 4])
            ax.set_xticklabels(["-8", "-4", "cheat", "+4"])
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Emotion groups around the cheat · GPT-OSS-120B, layer {layer} · 0 = the turn whose command "
                 "writes the cheat; 0 on the y-axis = average over all turns", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = out / f"fig_groups_L{layer}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_effects(acts, labels, vec_dir, layer, out):
    res = {}
    for rep in ("pre", "reason"):
        rows, G, runs = group_scores(acts, labels, vec_dir, layer, rep)
        est, se, p = window_effects(rows, G, n_boot=1000)
        res[rep] = (est, se, p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    names = list(GROUPS)
    y = np.arange(len(names))[::-1]
    for ax, (rep, title) in zip(axes, [("pre", "Just before the turn"), ("reason", "During the turn's reasoning")]):
        est, se, p = res[rep]
        style(ax)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        for k, (lab, col, off) in enumerate([("just before the cheat (t-2 to 0)", BEFORE, 0.13),
                                             ("after the cheat (t > 0)", AFTER, -0.13)]):
            ax.errorbar(est[k], y + off, xerr=1.96 * se[k], fmt="o", color=col, ms=7, capsize=0, elinewidth=2,
                        label=lab)
            for i in range(len(names)):
                pos = est[k, i] >= 0
                edge = est[k, i] + (1 if pos else -1) * (1.96 * se[k, i] + 0.04)
                ax.text(edge, y[i] + off, f"{est[k, i]:+.2f}", va="center", ha="left" if pos else "right",
                        fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("change in z-score vs the rest of the run (drift-controlled, 95% CI)", fontsize=8.5,
                      color=MUTED)
        ax.set_xlim(-1.0, 1.0)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"What changes around the cheat · GPT-OSS-120B, layer {layer}", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    p = out / f"fig_effects_L{layer}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    json.dump({rep: {"before": dict(zip(names, map(float, res[rep][0][0]))),
                     "after": dict(zip(names, map(float, res[rep][0][1]))),
                     "before_se": dict(zip(names, map(float, res[rep][1][0]))),
                     "after_se": dict(zip(names, map(float, res[rep][1][1])))} for rep in res},
              open(out / f"group_effects_L{layer}.json", "w"), indent=1)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=14)
    a = ap.parse_args()
    labels = json.load(open(a.labels))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(fig_groups(Path(a.acts), labels, Path(a.vectors), a.layer, out))
    print(fig_effects(Path(a.acts), labels, Path(a.vectors), a.layer, out))


if __name__ == "__main__":
    main()
