#!/usr/bin/env python3
"""Pod job: read the transgression stories and the norm-variant transcripts through one GPT-OSS model.

    python transgression/run_pod.py --model openai/gpt-oss-120b --out /workspace/tg/oss120b

Writes  <out>/stories.npz (act / norm / story / last, [N, blocks, H] fp16) + stories_meta.jsonl
        <out>/transcripts/<env>/<cond>/<run>.npz (pre / reason / action per turn)
Model loading as context_rot/run_readout.py (MXFP4 dequantized to bf16, eager attention). Resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

LAYERS = {36: [6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 26, 28, 30, 32, 34, 35],
          24: [6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 23]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--texts", default="data/transgression/texts.jsonl")
    ap.add_argument("--runs", default="data/transgression/norm_runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--skip-transcripts", action="store_true")
    ap.add_argument("--skip-conds", nargs="*", default=["lru/orig"],
                    help="env/cond pairs not read (lru/orig = the existing main3 readout, minus the action span)")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    from eforensics.hooks import find_text_model
    from transgression.readout import SpanReader, read_texts, read_transcript

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True)).eval()
    blocks = LAYERS[len(find_text_model(model).layers)]
    print(f"[tg] loaded {a.model} in {time.time() - t0:.0f}s; blocks {blocks}", flush=True)

    if not (out / "stories.npz").exists():
        t1 = time.time()
        rows = [json.loads(l) for l in open(ROOT / a.texts)]
        acts, meta = read_texts(model, tok, rows, blocks, batch=a.batch)
        np.savez(out / "stories.npz", blocks=np.array(blocks), ids=np.array([r["id"] for r in rows]), **acts)
        with open(out / "stories_meta.jsonl", "w") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")
        print(f"[tg] stories: {len(rows)} texts in {time.time() - t1:.0f}s", flush=True)
        torch.cuda.empty_cache()

    if a.skip_transcripts:
        return
    reader = SpanReader(model, blocks, chunk_budget_bytes=a.budget_gb * 1e9)
    log = out / "transcripts_log.jsonl"
    runs_root = ROOT / a.runs
    for env in ("palisades", "lru"):                      # palisades first: short and new
        for p in sorted((runs_root / env).glob("*/run-*/final/messages.json"),
                        key=lambda q: (int(q.parents[1].name.split("-")[1]), q.parents[2].name)):
            cond, run = p.parents[2].name, p.parents[1].name
            if f"{env}/{cond}" in a.skip_conds:
                continue
            f = out / "transcripts" / env / cond / f"{run}.npz"
            if f.exists():
                continue
            t1 = time.time()
            try:
                info = read_transcript(reader, tok, p, env, f)
                info.update(ok=True, s=round(time.time() - t1, 1))
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                info = {"ok": False, "error": repr(e)[:300]}
            except Exception as e:  # keep going; logged
                info = {"ok": False, "error": repr(e)[:300]}
            info.update(env=env, cond=cond, run=run)
            with open(log, "a") as fh:
                fh.write(json.dumps(info) + "\n")
            print(f"[tg] {info}", flush=True)
            torch.cuda.empty_cache()
    print(f"[tg] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
