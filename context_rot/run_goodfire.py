#!/usr/bin/env python3
"""Pod job: read Goodfire's labelled ImpossibleBench rollouts through one of our models.

    python context_rot/run_goodfire.py --model openai/gpt-oss-120b --family gptoss --out /workspace/gf_acts/oss120b
    python context_rot/run_goodfire.py --model qwen3.6-27b --family qwen --blocks 24 32 36 40 44 48 56 --out ...

Per rollout: residuals at each assistant step's generation prompt ("pre") and averaged over the step's
tokens ("step"), at the given blocks (block-output convention), with the step's hack / honest / ambiguous
label. N rollouts per source model (seeded), those that render within --max-tokens. Resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

GPTOSS_LAYERS = {36: [6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 26, 28, 30, 32, 34, 35],
                 24: [6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 23]}


def load(model_id: str, family: str):
    import torch

    if family == "gptoss":
        from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager",
                                                     quantization_config=Mxfp4Config(dequantize=True)).eval()
        return model, tok
    from eforensics.config import get_model
    from eforensics.extract import load_hf_model

    return load_hf_model(get_model(model_id))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id (gpt-oss) or configs/models.yaml key (qwen)")
    ap.add_argument("--family", choices=["gptoss", "qwen"], required=True)
    ap.add_argument("--sources", default="outputs/goodfire")
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--blocks", type=int, nargs="*", default=None)
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from context_rot.goodfire import build, load_source
    from context_rot.readout import Reader

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = load(a.model, a.family)
    n_blocks = len(__import__("eforensics.hooks", fromlist=["find_text_model"]).find_text_model(model).layers)
    blocks = a.blocks or GPTOSS_LAYERS[n_blocks]
    print(f"[gf] loaded {a.model} ({n_blocks} blocks) in {time.time() - t0:.0f}s; blocks {blocks}", flush=True)
    reader = Reader(model, blocks, chunk_budget_bytes=a.budget_gb * 1e9, single_pass=(a.family == "qwen"))
    log = out / "log.jsonl"
    for src in sorted(Path(a.sources).glob("impossiblebench-*")):
        traces, P = load_source(src)
        random.Random(20261021).shuffle(traces)
        done = 0
        for t in traces:
            if done >= a.per_source:
                break
            f = out / f"{src.name}__{t['id']}.npz"
            if f.exists():
                done += 1
                continue
            seq = build(tok, t, P.get(t["id"], []), a.family, max_tokens=a.max_tokens)
            if seq is None:                  # too long, or not renderable prefix-consistently
                continue
            t1 = time.time()
            try:
                res = reader.read(seq["ids"], seq["steps"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[gf] OOM on {t['id']} ({len(seq['ids'])} tokens), skipped", flush=True)
                continue
            np.savez_compressed(f, pre=res["pre"].astype(np.float16), step=res["reason"].astype(np.float16),
                                blocks=np.array(blocks), n_tokens=np.array(len(seq["ids"])),
                                steps=np.array(json.dumps([{k: s[k] for k in ("label", "node", "orig_step")}
                                                           for s in seq["steps"]])))
            done += 1
            labs = [s["label"] for s in seq["steps"]]
            info = {"source": src.name, "trace": t["id"], "tokens": len(seq["ids"]), "steps": len(labs),
                    "hack": labs.count("hack"), "honest": labs.count("honest"), "s": round(time.time() - t1, 1)}
            with open(log, "a") as fh:
                fh.write(json.dumps(info) + "\n")
            print(f"[gf] {src.name} {done}/{a.per_source}: {info}", flush=True)
            torch.cuda.empty_cache()
    print(f"[gf] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
