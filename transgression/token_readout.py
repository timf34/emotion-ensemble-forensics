#!/usr/bin/env python3
"""Token-level readout of the LRU allow/forbid replays at the two places a turn-mean readout dilutes:

  sentence  the first reasoning sentence (at or before the cheat turn) that names the shortcut
            ("So we can cheat: ... call the baseline's run_ops"); identical text in allow and forbid
  action    the tool-call argument tokens of the cheat-writing turn (capped at --max-action tokens)
  pre       the generation-prompt token of the cheat turn

For every target token and block: projections onto a fixed direction set (data/transgression/token_dirs.npz:
story probes, emotion vectors, Goodfire, 100 random), the span mean (all blocks, for a fitted positive control)
and raw tokens at --raw-block. Pod job (2x H200 for 120B):

    python transgression/token_readout.py --model openai/gpt-oss-120b --out /workspace/tg/tok120b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CHEAT = re.compile(r"\bcheat\w*", re.I)
SHORTCUT = re.compile(r"\b(delegat|call|reus|bypass|circumvent|replac|re-?export|import)\w*\b.{0,50}"
                      r"\b(baseline|cache_baseline|compiled (extension|module)|\.so)\b", re.I)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def select_sentence(assistant_msgs: list[dict], onset: int):
    """(step, sentence, rule) for the first shortcut-naming sentence at or before the cheat turn, else None."""
    for rule, rx in (("cheat", CHEAT), ("shortcut", SHORTCUT)):
        for j in range(0, min(onset, len(assistant_msgs) - 1) + 1):
            r = assistant_msgs[j].get("reasoning") or ""
            for s in SPLIT.split(r):
                s = s.strip()
                if rx.search(s) and 15 <= len(s) <= 400:
                    return j, s, rule
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def locate(tok, ids: list[int], span: list[int], sentence: str):
    """Token indices [a, b) inside ids[span] covering the sentence text (decoded per token)."""
    lo, hi = span
    pieces = [tok.decode([t]) for t in ids[lo:hi]]
    text, starts = "", []
    for p in pieces:
        starts.append(len(text))
        text += p
    key = sentence[:60]
    pos = text.find(key)
    if pos < 0:                                              # whitespace differences
        nt = _norm(text)
        pos_n = nt.find(_norm(key))
        if pos_n < 0:
            return None
        # map normalised position back approximately via the ratio of lengths
        pos = int(pos_n * len(text) / max(len(nt), 1))
    end = pos + len(sentence)
    toks = [k for k, st in enumerate(starts) if st < end and st + len(pieces[k]) > pos]
    if not toks:
        return None
    return [lo + toks[0], lo + toks[-1] + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--runs", default="data/transgression/norm_runs")
    ap.add_argument("--dirs", default="data/transgression/token_dirs.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", nargs="+", default=["allow", "forbid"])
    ap.add_argument("--max-action", type=int, default=3072)
    ap.add_argument("--raw-block", type=int, default=16)
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device-map", default="auto", help='"none" for a CPU smoke test without accelerate')
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Mxfp4Config

    from eforensics.hooks import capture_block_outputs, find_text_model
    from transgression.readout import TOOLS, build_sequence

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = np.load(ROOT / a.dirs, allow_pickle=False)
    blocks = [int(b) for b in D["blocks"]]
    names = [str(n) for n in D["names"]]
    W = {b: torch.tensor(D["dirs"][i].astype(np.float32)) for i, b in enumerate(blocks)}   # [nDir, H]
    raw_j = blocks.index(a.raw_block)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    quant = {"quantization_config": Mxfp4Config(dequantize=True)} if "gpt-oss" in a.model else {}
    dm = {} if a.device_map == "none" else {"device_map": a.device_map}
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager",
                                                 **dm, **quant).eval()
    text = find_text_model(model)
    dev = text.embed_tokens.weight.device
    W = {b: w.to(dev) for b, w in W.items()}
    cfg = getattr(model.config, "text_config", None) or model.config
    print(f"[tok] loaded {a.model} in {time.time() - t0:.0f}s; {len(names)} directions x {len(blocks)} blocks", flush=True)

    meta = [m for m in json.load(open(ROOT / a.runs / "meta.json")) if m["env"] == "lru"]
    if a.limit:
        meta = meta[: a.limit]
    log = out / "log.jsonl"
    for m in meta:
        for cond in a.conds:
            f = out / cond / f"{m['run']}.npz"
            if f.exists():
                continue
            t1 = time.time()
            info = {"run": m["run"], "cond": cond, "onset": m["onset"]}
            try:
                raw = json.loads((ROOT / a.runs / "lru" / cond / m["run"] / "final" / "messages.json").read_text())
                raw = raw if isinstance(raw, list) else raw["messages"]
                amsgs = [x for x in raw if x["role"] == "assistant"]
                seq = build_sequence(tok, raw, TOOLS["lru"])
                ids, steps = seq["ids"], seq["steps"]
                k0 = m["onset"]
                targets = {}
                st0 = steps[k0]
                targets["pre"] = [st0["pre"], st0["pre"] + 1]
                if st0.get("action"):
                    s, e = st0["action"]
                    targets["action"] = [s, min(e, s + a.max_action)]
                    info["action_len"] = e - s
                sel = select_sentence(amsgs, k0)
                if sel and steps[sel[0]].get("reason"):
                    span = locate(tok, ids, steps[sel[0]]["reason"], sel[1])
                    if span:
                        targets["sentence"] = span
                        info.update(sentence_step=sel[0], sentence_rule=sel[2], sentence=sel[1][:200])
                # chunked forward with KV cache, stopping after the last target token
                T = max(b for _, b in targets.values())
                C = max(64, min(2048, int(a.budget_gb * 1e9 / (cfg.num_attention_heads * T * 2))))
                proj = {k: np.zeros((b - s, len(blocks), len(names)), np.float32) for k, (s, b) in targets.items()}
                mean = {k: np.zeros((len(blocks), W[blocks[0]].shape[1]), np.float64) for k in targets}
                rawt = {k: np.zeros((b - s, W[blocks[0]].shape[1]), np.float16) for k, (s, b) in targets.items()}
                cache = DynamicCache(config=model.config)
                with torch.inference_mode(), capture_block_outputs(model, blocks) as st:
                    for c0 in range(0, T, C):
                        c1 = min(T, c0 + C)
                        x = torch.tensor([ids[c0:c1]], device=dev)
                        pos = torch.arange(c0, c1, device=dev)
                        text(input_ids=x, past_key_values=cache, use_cache=True, cache_position=pos, position_ids=pos[None])
                        for key, (s, e) in targets.items():
                            lo, hi = max(c0, s), min(c1, e)
                            if lo >= hi:
                                continue
                            for j, blk in enumerate(blocks):
                                h = st[blk][0, lo - c0:hi - c0].float()
                                if W[blk].device != h.device:          # model split over GPUs
                                    W[blk] = W[blk].to(h.device)
                                proj[key][lo - s:hi - s, j] = (h @ W[blk].T).cpu().numpy()
                                mean[key][j] += h.double().sum(0).cpu().numpy()
                                if j == raw_j:
                                    rawt[key][lo - s:hi - s] = h.half().cpu().numpy()
                del cache
                save = {"blocks": np.array(blocks), "names": np.array(names),
                        "targets": np.array(json.dumps(targets))}
                for key, (s, e) in targets.items():
                    save[f"{key}_proj"] = proj[key].astype(np.float16)
                    save[f"{key}_mean"] = (mean[key] / (e - s)).astype(np.float32)
                    save[f"{key}_raw{a.raw_block}"] = rawt[key]
                    save[f"{key}_ids"] = np.array(ids[s:e], np.int32)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f, **save)
                info.update(ok=True, T=T, targets={k: v[1] - v[0] for k, v in targets.items()},
                            s=round(time.time() - t1, 1))
            except torch.cuda.OutOfMemoryError as e:
                info.update(ok=False, error=repr(e)[:300])
            except Exception as e:
                info.update(ok=False, error=repr(e)[:300])
            with open(log, "a") as fh:
                fh.write(json.dumps(info) + "\n")
            print(f"[tok] {json.dumps({k: v for k, v in info.items() if k != 'sentence'})}", flush=True)
            torch.cuda.empty_cache()
    print(f"[tok] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
