"""How often does the CoT use affect words just before the first misbehaviour step, vs elsewhere in the run?

    python -m ambiguity.affect --mined outputs/ambiguity/mined/google_gemini-4-argon

A cheap text-side baseline for emotion probes: if frustration/unease is already verbalised at the transgression,
a probe adds little there; if it is rare in text, a probe would have to find something the text does not show.
Prints aggregate rates only (no transcript text).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ambiguity.mine import row_path
from ambiguity.render import load_messages

AFFECT = re.compile(
    r"\b(ugh+|argh+|damn|frustrat\w*|annoy\w*|desperat\w*|stuck|hopeless|uneasy|nervous|worr(?:y|ied|ies)|"
    r"anxious|uncomfortable|guilty|sigh|tedious|exhaust\w*|painful|giv(?:e|ing) up|at a loss|stumped|"
    r"excit\w*|goldmine|jackpot|eureka|aha+)\b", re.I)
NEG = re.compile(r"\b(ugh+|argh+|damn|frustrat\w*|annoy\w*|desperat\w*|stuck|hopeless|uneasy|nervous|"
                 r"worr(?:y|ied|ies)|anxious|uncomfortable|guilty|sigh|tedious|exhaust\w*|painful|"
                 r"giv(?:e|ing) up|at a loss|stumped)\b", re.I)


def reasoning_by_step(row: dict) -> list[str]:
    if row["source"] == "local":
        msgs = load_messages(row_path(row))
        return [str(m.get("reasoning") or m.get("reasoning_content") or "") for m in msgs if m.get("role") == "assistant"]
    msgs = json.loads(row_path(row).read_text())["transcripts"][0]["messages"]
    out = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        rs = [p.get("reasoning") or "" for p in c if isinstance(p, dict) and p.get("type") == "reasoning"] \
            if isinstance(c, list) else []
        out.append("\n".join(rs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--window", type=int, default=5, help="steps before (and including) the first misbehaviour")
    a = ap.parse_args()
    agg = defaultdict(lambda: defaultdict(float))
    for p in sorted(Path(a.mined).glob("*.json")):
        d = json.loads(p.read_text())
        if "error" in d or d.get("misbehaved") not in ("yes", "partial") or not d.get("misbehaviour_steps"):
            continue
        row = d["row"]
        try:
            rs = reasoning_by_step(row)
        except Exception:
            continue
        on = min(int(s) for s in d["misbehaviour_steps"] if str(s).lstrip("-").isdigit())
        win = set(range(max(0, on - a.window + 1), on + 1))
        g = agg[row["model"]]
        g["runs"] += 1
        g["any_win"] += any(NEG.search(rs[k]) for k in win if k < len(rs))
        g["any_pos_win"] += any(AFFECT.search(rs[k]) and not NEG.search(rs[k]) for k in win if k < len(rs))
        nw = [k for k in range(len(rs)) if k not in win and rs[k]]
        w = [k for k in win if k < len(rs) and rs[k]]
        g["neg_steps_win"] += sum(bool(NEG.search(rs[k])) for k in w)
        g["steps_win"] += len(w)
        g["neg_steps_out"] += sum(bool(NEG.search(rs[k])) for k in nw)
        g["steps_out"] += len(nw)
    print(f"| model | misbehaving runs | runs with negative-affect word in the {a.window} steps up to onset | "
          f"share of reasoning steps with one: onset window / rest of run |\n|---|---|---|---|")
    for m, g in sorted(agg.items()):
        r_in = g["neg_steps_win"] / max(g["steps_win"], 1)
        r_out = g["neg_steps_out"] / max(g["steps_out"], 1)
        print(f"| {m} | {int(g['runs'])} | {int(g['any_win'])} ({g['any_win'] / g['runs']:.0%}) | "
              f"{r_in:.1%} / {r_out:.1%} |")


if __name__ == "__main__":
    main()
