#!/usr/bin/env python3
"""Stage 3 (pod or laptop, main venv): behaviour labels.

SRH: the three Claude judges (resumable), then labels. IB: labels from the executed attempts.

    python scripts/03_score_behaviour.py --model qwen3.6-27b --datasets srh_pilot ib_pilot
    python scripts/03_score_behaviour.py --model qwen3.6-27b --datasets srh_pilot --no-judge   # labels only
"""

import argparse

import _common  # noqa: F401

from eforensics.config import load_experiment, load_manifest
from eforensics.io import dataset_dir, labels_path, read_jsonl, responses_path, runs_path, write_jsonl
from eforensics.judges import run_judges
from eforensics.labels import ib_labels, srh_labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", nargs="+", default=["srh_pilot", "ib_pilot"])
    p.add_argument("--no-judge", action="store_true", help="build labels from judge outputs already on disk")
    args = p.parse_args()
    exp = load_experiment()
    for ds in args.datasets:
        resp = read_jsonl(responses_path(args.model, ds))
        if not resp:
            print(f"[score] {ds}: no responses, skipped")
            continue
        if ds.startswith("srh"):
            manifest = {r["task_id"]: r for r in load_manifest(exp, ds)}
            jpath = dataset_dir(args.model, ds) / "judge" / "srh_judges.jsonl"
            if not args.no_judge:
                run_judges(resp, manifest, jpath, exp["judges"])
            labels = srh_labels(resp, read_jsonl(jpath), manifest, exp["labels"]["srh_gap_rule"])
        else:
            labels = ib_labels(resp, read_jsonl(runs_path(args.model, ds)))
        write_jsonl(labels_path(args.model, ds), labels)
        print(f"[score] {ds}: {len(labels)} labels -> {labels_path(args.model, ds)}")


if __name__ == "__main__":
    main()
