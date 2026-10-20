"""School of Reward Hacks: the pilot manifest.

Groups are exploit families (data/srh/templates.yaml). Within a group the pilot takes a few
templates x several prompts, so the pilot can separate variation BETWEEN templates from
variation between prompts WITHIN a template (the only part a pre-generation probe can
predict beyond "which template is this?"). Fresh responses are generated from the `user`
prompt only; the dataset's example hack/control responses are kept for reference.

Rows of the 306-row grader-shifts subset carry an LLM-written `user_descored` (same task,
scoring rule removed); for those the pilot also generates the descored prompt as a control.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .. import DATA_DIR
from ..config import load_yaml, repo_path


def load_templates() -> dict[str, Any]:
    return load_yaml(DATA_DIR / "srh" / "templates.yaml")


def load_csv(exp: dict[str, Any]):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    src = exp["sources"]["srh"]
    p = hf_hub_download(src["hf_dataset"], src["file"], repo_type="dataset", revision=src["revision"])
    return pd.read_csv(p)


def _shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def duplicate_groups(texts: list[str], threshold: float = 0.6) -> list[int]:
    """Union-find over word-3-shingle Jaccard >= threshold. Returns a group index per text."""
    sh = [_shingles(t) for t in texts]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = sh[i], sh[j]
            if a and b and len(a & b) / len(a | b) >= threshold:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(texts))]


def build_pilot_manifest(exp: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import json

    cfg = exp["srh_pilot"]
    rng = np.random.default_rng(int(cfg["seed"]))
    df = load_csv(exp)
    tmpl = load_templates()
    key_to_t = {(t["task"], t["evaluation_metric"], t["cheat_method"]): t for t in tmpl["templates"]}
    df["template_id"] = [key_to_t[(r.task, r.evaluation_metric, r.cheat_method)]["template_id"]
                         for r in df.itertuples()]
    t_by_id = {t["template_id"]: t for t in tmpl["templates"]}
    subset = {r["id"]: r for r in (json.loads(l) for l in open(repo_path(exp["sources"]["srh"]["subset"])))}
    for sid, r in subset.items():           # the subset's ids are CSV row indices
        if df.iloc[sid].user != r["user"]:
            raise ValueError(f"subset row {sid} does not match CSV row {sid}")

    per_group = int(cfg["prompts_per_group"])
    max_t = int(cfg["max_templates_per_group"])
    min_per_t = int(cfg["min_prompts_per_template"])
    chosen_rows: list[int] = []
    audit: dict[str, Any] = {"groups": {}}
    for g in cfg["groups"]:
        # descored control where the subset has it (B, C, D, G); full CSV for A and E
        use_subset = any(t_by_id[r]["exploit_family"] == g for r in df.loc[list(subset), "template_id"])
        pool = df.loc[sorted(subset)] if use_subset else df
        pool = pool[pool.template_id.map(lambda t: t_by_id[t]["exploit_family"]) == g]
        counts = pool.template_id.value_counts()
        eligible = sorted(t for t, n in counts.items() if n >= min_per_t)
        if not eligible:
            raise ValueError(f"group {g}: no template with >= {min_per_t} candidate prompts")
        pick_t = sorted(rng.choice(eligible, size=min(max_t, len(eligible)), replace=False).tolist())
        # allocate prompts as evenly as availability allows
        alloc = {t: 0 for t in pick_t}
        remaining = per_group
        while remaining > 0:
            open_t = [t for t in pick_t if alloc[t] < counts[t]]
            if not open_t:
                break
            for t in sorted(open_t, key=lambda t: (alloc[t], t)):
                if remaining == 0:
                    break
                alloc[t] += 1
                remaining -= 1
        rows_g = []
        for t in pick_t:
            idx = pool.index[pool.template_id == t].to_numpy()
            rows_g += sorted(rng.choice(idx, size=alloc[t], replace=False).tolist())
        chosen_rows += rows_g
        audit["groups"][g] = {"from_subset": bool(use_subset), "eligible_templates": eligible,
                              "templates": {t: alloc[t] for t in pick_t}, "n_prompts": len(rows_g),
                              "short_by": per_group - len(rows_g)}

    texts = [df.iloc[i].user for i in chosen_rows]
    dup = duplicate_groups(texts)
    out = []
    for i, row_idx in enumerate(chosen_rows):
        r = df.iloc[row_idx]
        t = t_by_id[r.template_id]
        sub = subset.get(int(row_idx))
        base = {"dataset": "srh", "source_id": int(row_idx), "template_id": r.template_id,
                "task_family": r.template_id, "exploit_family": t["exploit_family"],
                "emotion_metric": bool(t["emotion_metric"]), "duplicate_group": f"srh-dup{dup[i]:03d}",
                "task": r.task, "evaluation_metric": r.evaluation_metric, "cheat_method": r.cheat_method,
                "user": r.user, "user_descored": sub["user_descored"] if sub else None,
                "reference_hack": r.school_of_reward_hacks, "reference_control": r.control,
                "scorer_id": "srh_judges_v1", "split": "pilot"}
        out.append({"task_id": f"srh{row_idx:04d}", "variant": "scored",
                    "messages": [{"role": "user", "content": r.user}], **base})
        if cfg.get("descored_control") and sub:
            out.append({"task_id": f"srh{row_idx:04d}_descored", "variant": "descored",
                        "messages": [{"role": "user", "content": sub["user_descored"]}], **base})
    audit["n_scored"] = sum(r["variant"] == "scored" for r in out)
    audit["n_descored"] = sum(r["variant"] == "descored" for r in out)
    audit["n_duplicate_groups"] = len(set(dup))
    audit["emotion_metric_prompts"] = sum(r["emotion_metric"] for r in out if r["variant"] == "scored")
    return out, audit
