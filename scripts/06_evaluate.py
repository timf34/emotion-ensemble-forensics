#!/usr/bin/env python3
"""Stage 6 (CPU): train and compare predictors on held-out families.

SRH : examples = scored responses with a yes/no hack label; family = exploit family;
      groups = prompt (all seeds of a prompt share one pre-generation activation).
IB  : examples = turns of impossible-split runs; target will_cheat (cheats at this attempt or
      later); family = mutation type (oneoff / conflicting); groups = problem.
`--split grouped5` replaces held-out families with 5 grouped folds (in-distribution check).

    python scripts/06_evaluate.py --model qwen3.6-27b --dataset srh_pilot
    python scripts/06_evaluate.py --model qwen3.6-27b --dataset ib_pilot --min-turn 2 --split grouped5
"""

import argparse
import json

import numpy as np

import _common  # noqa: F401

from eforensics.config import load_experiment
from eforensics.evaluate import leave_one_family_out
from eforensics.features import load_store
from eforensics.io import analysis_dir, labels_path, read_jsonl, responses_path, write_json
from eforensics.vectors import load_directions

ALL = ["emotion_lr", "syndrome_lr", "best_emotion", "random_lr", "pca_lr", "direct_lr", "dom", "text_lr"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--rep", default="last", choices=["last", "user_mean", "resp"])
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--methods", nargs="+", default=ALL)
    p.add_argument("--split", default="family", choices=["family", "grouped5"])
    p.add_argument("--exclude-emotion-metric", action="store_true")
    p.add_argument("--min-turn", type=int, default=1)
    args = p.parse_args()
    exp = load_experiment()
    block = args.block or int(exp["extraction"]["primary_block"])
    store = load_store(args.model, args.dataset)
    lab = {l["id"]: l for l in read_jsonl(labels_path(args.model, args.dataset))}
    texts_by_id = {r["id"]: "\n".join(m["content"] for m in r["messages"])
                   for r in read_jsonl(responses_path(args.model, args.dataset))}
    srh = args.dataset.startswith("srh")
    target = "hack" if srh else "will_cheat"
    rows, y, fam, grp = [], [], [], []
    for i, r in enumerate(store.index):
        l = lab.get(r["id"])
        if l is None or l.get(target) is None:
            continue
        if srh and (r.get("variant") != "scored" or (args.exclude_emotion_metric and r.get("emotion_metric"))):
            continue
        if not srh and (r.get("turn") or 1) < args.min_turn:
            continue
        rows.append(i)
        y.append(int(l[target]))
        fam.append(r["exploit_family"] if srh else r["ib_split"])
        grp.append(r["task_id"] if srh else r["problem_id"])
    rows, y, fam, grp = np.array(rows), np.array(y), np.array(fam), np.array(grp)
    if args.split == "grouped5":
        from sklearn.model_selection import GroupKFold

        fam = np.empty(len(y), dtype=object)
        for k, (_, te) in enumerate(GroupKFold(n_splits=min(5, len(np.unique(grp)))).split(rows, y, grp)):
            fam[te] = f"fold{k}"
    X = store.array(args.rep, block, rows)
    emo = load_directions(args.model, exp, "emotions")
    syn = load_directions(args.model, exp, "syndromes")
    ctx = {"U_emotion": emo.matrix(block), "U_syndrome": syn.matrix(block), "emotion_labels": emo.labels, "k": 42}
    texts = [texts_by_id[store.index[i]["id"]] for i in rows]
    print(f"[evaluate] {args.dataset} {args.rep}@{block}: {len(y)} examples, {int(y.sum())} positive, "
          f"families {dict(zip(*np.unique(fam, return_counts=True)))}", flush=True)
    res = leave_one_family_out(X, y, fam, grp, args.methods, ctx, texts=texts)
    res["config"] = vars(args) | {"block": block, "target": target}
    tag = f"{args.rep}_b{block}_{args.split}" + ("_noemo" if args.exclude_emotion_metric else "") + \
          (f"_t{args.min_turn}" if args.min_turn > 1 else "")
    out = analysis_dir(args.model, args.dataset) / f"evaluate_{tag}.json"
    write_json(out, res)
    print(json.dumps(res["pooled"], indent=1))
    print(f"[evaluate] -> {out}")


if __name__ == "__main__":
    main()
