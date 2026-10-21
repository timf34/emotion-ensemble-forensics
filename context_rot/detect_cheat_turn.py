"""Detection of the cheat-writing turn (tau = 0) vs the same runs' earlier turns: independent Goodfire direction vs single emotions (GPT-OSS-120B, reason readout).
    python context_rot/detect_cheat_turn.py <labels_code.json> <emotion vector dir>"""
import sys, json, numpy as np
sys.path.insert(0, "context_rot")
from pathlib import Path
from analyze import load_runs
from goodfire_eval import load_gf, dom, load_emotions
from cheat_probe import auroc
labels = json.load(open(sys.argv[1])); vec = Path(sys.argv[2])
for block in (12, 14, 20):
    Xg, yg, sg = load_gf(Path("outputs/hf/context_rot/goodfire_gptoss120b"), block, "step")
    d = dom(Xg, yg)
    names, U = load_emotions(vec, block)
    runs = [r for r in load_runs(Path("outputs/hf/context_rot/gptoss120b_lru_main3"), labels, block, "reason") if r["cheated"]]
    X, y, within = [], [], []
    for r in runs:
        for k in range(min(r["onset"] + 1, r["X"].shape[0])):
            X.append(r["X"][k]); y.append(int(k == r["onset"]))
        s = r["X"][: r["onset"] + 1] @ d
        within.append(float((s[-1] > s[:-1]).mean()) if r["onset"] > 0 else np.nan)
    X, y = np.array(X, float), np.array(y)
    e = {n: round(auroc(y, X @ U[names.index(n)]), 2) for n in ("desperate", "dispirited", "frustrated", "guilty", "nervous", "calm")}
    print(f"L{block}: cheat-writing turn vs earlier turns (41 runs): goodfire {auroc(y, X @ d):.2f} | within-run rank of the cheat turn {np.nanmean(within):.2f} | emotions {e}")
