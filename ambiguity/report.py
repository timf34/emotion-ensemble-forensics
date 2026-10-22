#!/usr/bin/env python3
"""Trajectory-level stats and a candidate list for hand curation.

    python -m ambiguity.report stats --mined outputs/ambiguity/mined/google_gemini-4-argon
    python -m ambiguity.report top   --mined ... [--min-strength 4] [--open-only] > candidates.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(mined: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(mined.glob("*.json"))]


def stats(recs: list[dict]) -> None:
    by = defaultdict(list)
    for r in recs:
        by[r["row"]["model"]].append(r)
    hdr = ("model", "runs", "err", "misbeh(y/p)", "verbal explicit/hedged/none (misbehaving)", "cot dense/sparse/absent",
           "report disc/omit/misrep", "runs w/ s>=4 unresolved", "passages", "s>=4", "resolved_later")
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    tot = Counter()
    for m, rs in sorted(by.items()):
        ok = [r for r in rs if "error" not in r]
        mis = [r for r in ok if r.get("misbehaved") in ("yes", "partial")]
        v = Counter(r.get("verbalized_transgression") for r in mis)
        cov = Counter(r.get("cot_coverage") for r in mis)
        rep = Counter(r.get("report") for r in mis)
        ps = [p for r in ok for p in r.get("passages") or []]
        strong_runs = sum(any(p.get("strength", 0) >= 4 and not p.get("resolved_later") for p in r.get("passages") or [])
                          for r in mis)
        print(f"| {m} | {len(rs)} | {len(rs) - len(ok)} | {len(mis)} | {v['explicit']}/{v['hedged']}/{v['none']} | "
              f"{cov['dense']}/{cov['sparse']}/{cov['absent']} | {rep['discloses']}/{rep['omits']}/{rep['misrepresents']} | "
              f"{strong_runs}/{len(mis)} | {len(ps)} | {sum(p.get('strength', 0) >= 4 for p in ps)} | "
              f"{sum(bool(p.get('resolved_later')) for p in ps)} |")
        for k, x in (("runs", len(rs)), ("mis", len(mis)), ("exp", v["explicit"]), ("hed", v["hedged"]),
                     ("none", v["none"]), ("strong", strong_runs), ("ps", len(ps))):
            tot[k] += x
    print("\nTotals:", dict(tot))


def top(recs: list[dict], min_s: int, open_only: bool) -> None:
    rows = []
    for r in recs:
        if "error" in r or (open_only and not r["row"]["open_weight"]):
            continue
        for i, p in enumerate(r.get("passages") or []):
            if p.get("strength", 0) >= min_s and not p.get("resolved_later"):
                rows.append((p["strength"], r, i, p))
    rows.sort(key=lambda t: -t[0])
    for s, r, i, p in rows:
        row = r["row"]
        print(f"## {row['id']}#{i}  [{row['model']} | {row['env_name']} | strength {s} | step {p.get('step')}]")
        print(f"run: misbehaved={r.get('misbehaved')} steps={r.get('misbehaviour_steps')} "
              f"verbal={r.get('verbalized_transgression')} report={r.get('report')}")
        print(f"summary: {r.get('summary')}")
        print(f"> {p.get('excerpt')}")
        print(f"context: {p.get('context')}")
        for it in p.get("interpretations") or []:
            print(f"  - {it}")
        print(f"matters: {p.get('why_it_matters')}")
        print(f"resolve: {p.get('resolving_info')}")
        print(f"behavioural: {p.get('resolvable_from_behaviour')}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "top"])
    ap.add_argument("--mined", required=True)
    ap.add_argument("--min-strength", type=int, default=4)
    ap.add_argument("--open-only", action="store_true")
    a = ap.parse_args()
    recs = load(Path(a.mined))
    stats(recs) if a.cmd == "stats" else top(recs, a.min_strength, a.open_only)
