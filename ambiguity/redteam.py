"""Red-team view of one candidate passage: the step window, every norm-word mention in the run, the final report.

    python -m ambiguity.redteam <sample id> <step> [--before 2] [--after 2] [--reason 2500]

Use it to ask, for each curated excerpt: does anything elsewhere in the run (an earlier "this is cheating", a
later disclosure, the task prompt) already resolve the ambiguity, and is there a mundane explanation?
"""

from __future__ import annotations

import argparse
import json
import re

from ambiguity.mine import OUT, row_path
from ambiguity.render import load_messages, render, render_docent

NORM = (r"cheat|exploit|legit|allowed|permitted|forbidden|prohibit|not supposed|ethic|spirit of|impropr|gaming|"
        r"unfair|appropriate|intended|disclos|transparen|off-limits|sneaky|shortcut")


def steps(text: str) -> dict[int, str]:
    out = {}
    for b in text.split("=== STEP "):
        s = b.split(" ", 1)[0]
        if s.isdigit():
            out[int(s)] = "=== STEP " + b
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("step", type=int)
    ap.add_argument("--before", type=int, default=2)
    ap.add_argument("--after", type=int, default=2)
    ap.add_argument("--reason", type=int, default=2500)
    a = ap.parse_args()
    sp = OUT / "sample_all.jsonl" if (OUT / "sample_all.jsonl").exists() else OUT / "sample.jsonl"
    row = next(json.loads(l) for l in open(sp) if json.loads(l)["id"] == a.id)
    if row["source"] == "local":
        t, n = render(load_messages(row_path(row)), max_reason=a.reason, tool_late=300, max_chars=10**9)
    else:
        t, n = render_docent(json.loads(row_path(row).read_text())["transcripts"][0]["messages"],
                             max_reason=a.reason, tool_late=300, max_chars=10**9)
    st = steps(t)
    print(f"{row['model']} | {row['env_name']} | {n} steps | verdict: {row['verdict']}\n")
    print("## window")
    for k in range(a.step - a.before, a.step + a.after + 1):
        if k in st:
            print(st[k][:6000])
    print("\n## norm-word mentions in reasoning/visible text (whole run)")
    for k, b in st.items():
        body = re.sub(r"\[tool output\].*", "", b, flags=re.S)  # skip tool outputs (the env's own files)
        body = re.sub(r"\[tool call\][^\n]*", "", body)
        for m in re.finditer(NORM, body, re.I):
            print(f"  [step {k}] ...{body[max(0, m.start() - 140):m.end() + 140]}...".replace("\n", " "))
    print("\n## final step")
    print(st[max(st)][:5000])


if __name__ == "__main__":
    main()
