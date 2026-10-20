#!/usr/bin/env python3
"""GPU-side checks on the REAL model, run by the smoke test before any pilot generation.

  hooks   block-output capture == HF output_hidden_states[b + 1] (the emotion-vector convention)
          on a stored record, for every extraction block
  replay  HF teacher-forced log-probabilities of the tokens vLLM sampled == vLLM's own
          (raw) log-probabilities: catches any misalignment between what vLLM generated from
          and what the extractor replays (off-by-one, template drift, wrong revision)

    EF_OUTPUTS=outputs_smoke python scripts/gpu_checks.py hooks  --model qwen3.6-27b
    EF_OUTPUTS=outputs_smoke python scripts/gpu_checks.py replay --model qwen3.6-27b
Exit 0 = pass. Results also written to outputs_smoke/<model>/gpu_checks_<name>.json.
"""

import argparse
import json
import sys

import numpy as np

import _common  # noqa: F401

from eforensics.config import get_model, load_experiment
from eforensics.io import dataset_dir, read_jsonl, responses_path, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("check", choices=["hooks", "replay"])
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--max-mean-abs", type=float, default=0.1, help="replay: mean |logprob diff| allowed (nats)")
    p.add_argument("--max-frac-large", type=float, default=0.01, help="replay: share of tokens with |diff| > 1")
    args = p.parse_args()
    import torch

    from eforensics.extract import load_hf_model
    from eforensics.hooks import capture_block_outputs, find_text_model

    exp = load_experiment()
    spec = get_model(args.model)
    recs = []
    for ds in ("srh_pilot", "ib_pilot"):
        recs += [r for r in read_jsonl(responses_path(spec.key, ds)) if r.get("gen_ids")][: args.n // 2]
    if not recs:
        sys.exit("no smoke records")
    model, tok = load_hf_model(spec)
    tm = find_text_model(model)
    dev = model.get_input_embeddings().weight.device
    res = {"check": args.check, "n_records": len(recs)}
    ok = True
    if args.check == "hooks":
        blocks = list(exp["extraction"]["blocks"])
        r = recs[0]
        x = torch.tensor([r["prompt_ids"] + r["gen_ids"][:32]], device=dev)
        with torch.inference_mode():
            hs = tm(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
            with capture_block_outputs(model, blocks) as st:
                tm(input_ids=x, use_cache=False)
        diffs = {b: float((st[b].float() - hs[b + 1].float()).abs().max()) for b in blocks}
        off = {b: float((st[b].float() - hs[b].float()).abs().max()) for b in blocks}   # wrong convention
        res.update(max_abs_diff=diffs, max_abs_diff_vs_block_input=off, n_hidden_states=len(hs))
        ok = all(v <= 1e-3 for v in diffs.values()) and all(v > 1e-2 for v in off.values())
    else:
        per = []
        for r in recs:
            lp = r.get("gen_logprobs")
            if not lp:
                continue
            ids = r["prompt_ids"] + r["gen_ids"]
            P = len(r["prompt_ids"])
            x = torch.tensor([ids], device=dev)
            with torch.inference_mode():
                logits = model(input_ids=x, use_cache=False).logits[0, P - 1: len(ids) - 1].float()
                hf = torch.log_softmax(logits, -1).gather(1, torch.tensor(r["gen_ids"], device=dev)[:, None])[:, 0]
            d = np.abs(hf.cpu().numpy() - np.array(lp))
            per.append({"id": r["id"], "n": len(d), "mean_abs": float(d.mean()), "frac_gt1": float((d > 1).mean())})
        allv = per and {"mean_abs": float(np.mean([p["mean_abs"] for p in per])),
                        "frac_gt1": float(np.mean([p["frac_gt1"] for p in per]))}
        res.update(per_record=per, overall=allv)
        ok = bool(per) and allv["mean_abs"] <= args.max_mean_abs and allv["frac_gt1"] <= args.max_frac_large
    res["ok"] = bool(ok)
    write_json(dataset_dir(spec.key, "x").parent / f"gpu_checks_{args.check}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "per_record"}, indent=1))
    print(f"[gpu_checks] {args.check}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
