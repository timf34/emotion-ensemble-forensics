#!/usr/bin/env python3
"""Stage 4 (pod, main venv, one GPU per worker): HF teacher-forced replay of the stored token
ids; activations at the extraction blocks for the last / user_mean / resp representations.

    python scripts/04_extract_features.py --model qwen3.6-27b --datasets srh_pilot ib_pilot
    python scripts/04_extract_features.py --model qwen3.6-27b --merge-only
"""

import argparse

import _common  # noqa: F401

from eforensics.config import get_model, load_experiment
from eforensics.extract import Extractor, extract_dataset, load_hf_model, merge_shards
from eforensics.generate import config_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", nargs="+", default=["srh_pilot", "ib_pilot"])
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true", help="multi-worker runs merge once, after all workers")
    args = p.parse_args()
    exp = load_experiment()
    spec = get_model(args.model)
    blocks = list(exp["extraction"]["blocks"])
    if not args.merge_only:
        model, tok = load_hf_model(spec)
        ex = Extractor(model, tok, blocks, bool(exp["generation"]["enable_thinking"]), spec.chat_template_kwargs)
        for ds in args.datasets:
            n = extract_dataset(ex, spec.key, ds, worker=args.worker, num_workers=args.num_workers,
                                save_dtype=exp["extraction"]["save_dtype"])
            print(f"[extract] {ds}: {n} records extracted by worker {args.worker}/{args.num_workers}", flush=True)
    if not args.no_merge and (args.merge_only or args.num_workers == 1):
        for ds in args.datasets:
            cid, _ = config_identity(spec, exp, ds)
            n = merge_shards(spec.key, ds, blocks, {"config_id": cid, "model": spec.hf_id, "revision": spec.revision})
            print(f"[extract] {ds}: merged {n} records", flush=True)


if __name__ == "__main__":
    main()
