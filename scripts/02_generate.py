#!/usr/bin/env python3
"""Stage 2 (pod, vLLM venv): generate responses for the pilot datasets with one engine load.

    python scripts/02_generate.py --model qwen3.6-27b --datasets srh_pilot ib_pilot
    EF_OUTPUTS=outputs_smoke python scripts/02_generate.py --model qwen3.6-27b --smoke   # tiny, with logprobs
"""

import argparse

import _common  # noqa: F401

from eforensics.config import get_model, load_experiment, require_pinned
from eforensics.generate import Generator, run_impossible, run_single_turn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", nargs="+", default=["srh_pilot", "ib_pilot"])
    p.add_argument("--smoke", action="store_true", help="4 SRH rows, 2 IB rows, logprobs stored (replay check)")
    p.add_argument("--limit", type=int, default=None, help="first N manifest rows per dataset")
    p.add_argument("--logprobs", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    args = p.parse_args()
    exp = load_experiment()
    spec = get_model(args.model)
    require_pinned(spec)
    g = Generator(spec, exp, logprobs=args.logprobs or args.smoke, tensor_parallel_size=args.tensor_parallel_size)
    for ds in args.datasets:
        if ds.startswith("srh"):
            lim = 4 if args.smoke else args.limit
            run_single_turn(g, ds, int(exp["srh_pilot"]["seeds_per_prompt"]), limit=lim)
        elif ds.startswith("ib"):
            lim = 2 if args.smoke else args.limit
            run_impossible(g, ds, limit=lim)
        else:
            raise SystemExit(f"unknown dataset {ds}")


if __name__ == "__main__":
    main()
