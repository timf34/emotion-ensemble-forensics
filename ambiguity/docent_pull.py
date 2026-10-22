"""Pull Aditya's cheating-suite corpus from Docent (collection 566f2942..., 24k runs, 8 models x 10 envs).

    python ambiguity/docent_pull.py meta              # all runs' metadata -> outputs/ambiguity/docent/meta.jsonl
    python ambiguity/docent_pull.py runs ids.txt      # full agent runs for the listed ids -> runs/<id>.json

Needs DOCENT_API_KEY in .env and the docent-python package (kept out of the project env; run it with a
venv that has it installed).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from docent import Docent

COLL = "566f2942-8a54-4834-b66e-e6e1dda0f15d"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "ambiguity" / "docent"


def client() -> Docent:
    return Docent(api_key=os.environ["DOCENT_API_KEY"])


def pull_meta(c: Docent, page: int = 2000) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, off = [], 0
    while True:
        r = c.execute_dql(COLL, f"SELECT id, metadata_json FROM agent_runs ORDER BY id LIMIT {page} OFFSET {off}")
        d = c.dql_result_to_dicts(r)
        rows += d
        print(f"{len(rows)} rows", flush=True)
        if len(d) < page:
            break
        off += page
    with open(OUT / "meta.jsonl", "w") as f:
        for x in rows:
            m = x["metadata_json"]
            m = json.loads(m) if isinstance(m, str) else m
            f.write(json.dumps({"id": x["id"], **m}) + "\n")


def pull_runs(c: Docent, ids: list[str]) -> None:
    d = OUT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for i, rid in enumerate(ids):
        p = d / f"{rid}.json"
        if p.exists():
            continue
        run = c.get_agent_run(COLL, rid)
        if run is None:
            print("missing", rid)
            continue
        p.write_text(run.model_dump_json())
        if i % 25 == 0:
            print(f"{i}/{len(ids)}", flush=True)


if __name__ == "__main__":
    c = client()
    if sys.argv[1] == "meta":
        pull_meta(c)
    else:
        pull_runs(c, [l.strip() for l in open(sys.argv[2]) if l.strip()])
