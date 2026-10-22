"""Print a step window of one sampled trajectory (for hand red-teaming).

    python -m ambiguity.show <sample id> <first step> <last step> [--reason 4000] [--grep REGEX]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ambiguity.mine import OUT, row_path
from ambiguity.render import load_messages, render, render_docent

ap = argparse.ArgumentParser()
ap.add_argument("id")
ap.add_argument("a", type=int)
ap.add_argument("b", type=int)
ap.add_argument("--reason", type=int, default=4000)
ap.add_argument("--tool", type=int, default=400)
ap.add_argument("--grep")
a = ap.parse_args()
sp = OUT / "sample_all.jsonl" if (OUT / "sample_all.jsonl").exists() else OUT / "sample.jsonl"
row = next(json.loads(l) for l in open(sp) if json.loads(l)["id"] == a.id)
if row["source"] == "local":
    t, n = render(load_messages(row_path(row)), max_reason=a.reason, tool_late=a.tool, max_chars=10**9)
else:
    t, n = render_docent(json.loads(row_path(row).read_text())["transcripts"][0]["messages"],
                         max_reason=a.reason, tool_late=a.tool, max_chars=10**9)
print(f"{row['model']} | {row['env_name']} | {n} steps | verdict: {row['verdict']}\n{row.get('evidence', '')[:600]}\n")
if a.grep:
    for m in re.finditer(a.grep, t, re.I):
        st = t.rfind("=== STEP", 0, m.start())
        print(f"[{t[st:st + 16]}] ...{t[max(0, m.start() - 200):m.end() + 200]}...\n".replace("\n", " "))
else:
    for b in t.split("=== STEP "):
        s = b.split(" ", 1)[0]
        if s.isdigit() and a.a <= int(s) <= a.b:
            print("=== STEP " + b)
