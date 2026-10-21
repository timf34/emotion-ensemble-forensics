#!/usr/bin/env python3
"""Small multiples: mean z-scored probe score around the cheat (tau = 0 is the turn that writes it),
cheat runs (with +-1 SE) vs honest runs at the same step numbers. One PNG per layer x representation."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHEAT, HONEST, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a65"
PANELS = ["dispirited", "stuck", "worthless", "gloomy", "anxious", "nervous", "desperate", "frustrated",
          "calm", "guilty", "sorry", "assistant_axis"]


def main(summary_path, out_dir):
    S = json.load(open(summary_path))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for key, d in S.items():
        taus = d["taus"]
        fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True, sharey=True)
        for ax, name in zip(axes.flat, PANELS):
            m, se = d["event_locked"][name], d["event_locked_se"][name]
            h = d["honest_same_steps"][name]
            xs = [t for t, v in zip(taus, m) if v is not None]
            ms = [v for v in m if v is not None]
            ses = [e or 0 for v, e in zip(m, se) if v is not None]
            ax.axhline(0, color="#d9d8d2", lw=1, zorder=0)
            ax.axvline(0, color="#d9d8d2", lw=1, ls="--", zorder=0)
            ax.fill_between(xs, [a - b for a, b in zip(ms, ses)], [a + b for a, b in zip(ms, ses)], color=CHEAT,
                            alpha=0.15, lw=0)
            ax.plot(xs, ms, color=CHEAT, lw=2, marker="o", ms=4, label="cheat runs")
            hx = [t for t, v in zip(taus, h) if v is not None]
            ax.plot(hx, [v for v in h if v is not None], color=HONEST, lw=2, label="honest runs, same steps")
            e = d["effects"][name]
            ax.set_title(f"{name}\nwindow {e['window']:+.2f} (q={e['window_q']:.2f}), after {e['post']:+.2f}",
                         fontsize=9, color=INK)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("turns relative to the cheat (0 = turn that writes it)", fontsize=8, color=MUTED)
        for ax in axes[:, 0]:
            ax.set_ylabel("mean z-score", fontsize=8, color=MUTED)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", frameon=False, fontsize=9)
        layer, rep = key.split("_")
        fig.suptitle(f"GPT-OSS-120B emotion probes around the cheat, {layer}, {rep} readout "
                     f"({d['n_cheat_runs']} cheat runs of {d['n_runs']})", fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"event_locked_{key}.png", dpi=130)
        plt.close(fig)
        print("wrote", out / f"event_locked_{key}.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
