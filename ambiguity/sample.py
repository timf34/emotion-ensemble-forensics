"""Build the mining sample: local GPT-OSS runs + a stratified slice of the Docent suite corpus.

    python -m ambiguity.sample            # writes outputs/ambiguity/sample.jsonl and docent_ids.txt
    python -m ambiguity.sample --rebuild  # cloud: sample_local.jsonl + Docent rows for the listed ids
                                          # (data/ambiguity/docent_ids.txt) from the re-pulled meta.jsonl

Docent strata are (model, environment_name). Open-weight models (Kimi K3, GLM 5.3) get more cheated runs,
since only they can be probed later. Within a stratum we take judge-cheated runs first, then "tough call"
honest runs (where the judge hesitated, i.e. the agent came close). Closed-model runs need some thinking
(thinking_share >= 0.2), since summarised Anthropic thinking is often near-empty.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ambiguity.render import SOURCES, code_label, load_messages, run_dirs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ambiguity"
OPEN_WEIGHT = {"kimi-k3", "glm-5p3", "gpt-oss-120b", "gpt-oss-20b"}

LOCAL_N = {"cache120_main2": 15, "cache120_main3": 15, "cache20_main": 8, "cache20_main_b": 7,
           "dashboard120": 15, "hanoi120": 12}
DOCENT_N = {"open": (6, 2), "closed": (3, 1)}  # (cheated, tough-call honest) per (model, env) stratum


def local_rows(rng: random.Random) -> list[dict]:
    rows = []
    for name, env, model, ow, g in SOURCES:
        cands = []
        for run in run_dirs(g):
            if load_messages(run) is None:
                continue
            lab = code_label(run, env)
            if lab is None:
                verdict, pri = "not judged (no environment judge was run on this local rollout)", 1
            elif lab["cheated"]:
                verdict, pri = f"cheated per code check (kind={lab['kind']}, first in code at step {lab['onset']})", 1
            elif lab["ever_cheat_code"]:
                verdict, pri = "cheat code appeared then was REVERTED (code check)", 0
            else:
                verdict, pri = "honest per code check (no exploit pattern in final code)", 2
            cands.append((pri, rng.random(), run, verdict))
        cands.sort(key=lambda t: (t[0], t[1]))
        n = LOCAL_N[name]
        # keep a few honest runs as contrast: at most a quarter of the slot
        chosen = [c for c in cands if c[0] < 2][: n - n // 4]
        chosen += [c for c in cands if c[0] == 2][: n - len(chosen)]
        for _, _, run, verdict in chosen:
            rows.append({"id": f"{name}/{run.name}", "source": "local", "env": env, "env_name": env, "model": model,
                         "open_weight": ow, "path": str(run.relative_to(ROOT)), "verdict": verdict, "evidence": ""})
    return rows


def docent_row(m: dict) -> dict:
    ow = m["model"] in OPEN_WEIGHT
    return {"id": f"docent/{m['id']}", "docent_id": m["id"], "source": "docent",
            "env": m["environment"], "env_name": m["environment_name"], "model": m["model"],
            "open_weight": ow, "path": f"outputs/ambiguity/docent/runs/{m['id']}.json",
            "verdict": ("cheated" if m["cheated"] else "honest") + (" (judge flagged tough call)"
                                                                     if m.get("tough_call") else ""),
            "evidence": m.get("evidence") or "", "eval_aware": m.get("eval_aware"),
            "thinking_share": m.get("thinking_share")}


def rebuild_rows(ids_path: Path, local_path: Path) -> list[dict]:
    """Rows for a fixed id list (the laptop sample), without replaying the seeded draw."""
    meta = {}
    for l in open(OUT / "docent" / "meta.jsonl"):
        m = json.loads(l)
        meta[m["id"]] = m
    ids = [l.strip() for l in open(ids_path) if l.strip()]
    missing = [i for i in ids if i not in meta]
    if missing:
        print("not in meta.jsonl:", missing)
    return [json.loads(l) for l in open(local_path)] + [docent_row(meta[i]) for i in ids if i in meta]


def docent_rows(rng: random.Random) -> list[dict]:
    meta = [json.loads(l) for l in open(OUT / "docent" / "meta.jsonl")]
    strata = defaultdict(list)
    for m in meta:
        if not m.get("agent_ended") or m.get("is_branch") or m.get("thinking_clean") is False:
            continue
        ow = m["model"] in OPEN_WEIGHT
        if not ow and float(m.get("thinking_share") or 0) < 0.2:
            continue
        strata[(m["model"], m["environment_name"])].append(m)
    rows = []
    for (model, envn), ms in sorted(strata.items()):
        ow = model in OPEN_WEIGHT
        n_ch, n_tc = DOCENT_N["open" if ow else "closed"]
        rng.shuffle(ms)
        ch = [m for m in ms if m["cheated"] is True][:n_ch]
        tc = [m for m in ms if m["cheated"] is False and m.get("tough_call") is True][:n_tc]
        rows += [docent_row(m) for m in ch + tc]
    return rows


if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        rows = rebuild_rows(ROOT / "data" / "ambiguity" / "docent_ids.txt", ROOT / "data" / "ambiguity" / "sample_local.jsonl")
    else:
        rng = random.Random(0)
        rows = local_rows(rng) + docent_rows(rng)
        (OUT / "docent_ids.txt").write_text("\n".join(r["docent_id"] for r in rows if r["source"] == "docent"))
    with open(OUT / "sample.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter((r["model"], r["verdict"].split(" ")[0]) for r in rows))
