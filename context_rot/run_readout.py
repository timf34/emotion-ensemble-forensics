#!/usr/bin/env python3
"""Pod job: replay agent-interp-envs rollouts through GPT-OSS and save per-step residuals.

    python context_rot/run_readout.py --runs <dir with run-*/final/messages.json> --out <dir>

Model loading follows dprobe (emotion-concepts src/dprobe/models.py), the recipe the gpt-oss-120b
emotion vectors were extracted with: MXFP4 dequantized to bf16, eager attention (sinks), split
over the visible GPUs. Blocks = the vectors' analysis layers. Resumable per run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

ANALYSIS_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 26, 28, 30, 32, 34, 35]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--budget-gb", type=float, default=4.0, help="attention scratch per chunk (eager)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--blocks", type=int, nargs="*", default=None, help="default: the 120b vector layers")
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    from context_rot.readout import Reader, read_run

    runs = sorted(Path(args.runs).glob("run-*"), key=lambda r: int(r.name.split("-")[1]))
    runs = [r for r in runs if (r / "final" / "messages.json").exists()][: args.limit or None]
    out = Path(args.out)
    todo = [r for r in runs if not (out / f"{r.name}.npz").exists()]
    print(f"[readout] {len(runs)} runs, {len(todo)} to do", flush=True)
    if not todo:
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True)).eval()
    print(f"[readout] loaded {args.model} in {time.time() - t0:.0f}s on {torch.cuda.device_count()} GPU(s)", flush=True)
    reader = Reader(model, args.blocks or ANALYSIS_LAYERS, chunk_budget_bytes=args.budget_gb * 1e9)
    log = out / "readout_log.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    for r in todo:
        t1 = time.time()
        try:
            info = read_run(reader, tok, r, out / f"{r.name}.npz")
            info.update(seconds=round(time.time() - t1, 1), ok=True)
        except Exception as e:  # keep going; the failure is logged
            info = {"run": str(r), "ok": False, "error": repr(e)[:500]}
        with open(log, "a") as f:
            f.write(json.dumps(info) + "\n")
        print(f"[readout] {r.name}: {info}", flush=True)
        torch.cuda.empty_cache()
    print(f"[readout] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
