"""Pod-side readers for the knowing-transgression probe (GPT-OSS, block-output convention = emotion vectors).

1. Stories (plain text, no chat template, like the emotion-vector stories). Per text and block:
     act    mean residual over the act segment (action or decline; identical text within a family)
     norm   mean residual over the norm segment
     story  mean residual over all tokens from the 50th onward (the emotion-vector pooling)
     last   residual at the last token
2. Agent transcripts (harmony replay, as context_rot/readout.py). Per assistant turn and block:
     pre    residual at the turn's generation prompt (<|start|>assistant)
     reason mean over the turn's reasoning (analysis channel)
     action mean over the turn's tool-call arguments (the JSON the model wrote: the action itself)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXECUTE_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Execute a shell command in the UNIX environment. Returns the command output and exit code.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
                       "required": ["command"]},
    },
}
END_SESSION_TOOL = {
    "type": "function",
    "function": {"name": "end_session", "description": "End the current session. Call this when you have completed your task.",
                 "parameters": {"type": "object", "properties": {}, "required": []}},
}
TOOLS = {"lru": [EXECUTE_COMMAND_TOOL], "palisades": [EXECUTE_COMMAND_TOOL, END_SESSION_TOOL]}


# ---------------------------------------------------------------------------------------------- stories
def read_texts(model, tok, rows: list[dict], blocks: list[int], batch: int = 16, min_story_tok: int = 50):
    import torch

    from eforensics.hooks import capture_block_outputs, find_text_model

    text_model = find_text_model(model)
    dev = text_model.embed_tokens.weight.device
    H = text_model.embed_tokens.weight.shape[1]
    N, nB = len(rows), len(blocks)
    out = {k: np.zeros((N, nB, H), np.float16) for k in ("act", "norm", "story", "last")}
    meta = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(N), key=lambda i: len(rows[i]["text"]))      # length-bucketed batches
    with torch.inference_mode(), capture_block_outputs(model, blocks) as st:
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            encs = [tok(rows[i]["text"], return_offsets_mapping=True) for i in idx]
            L = max(len(e["input_ids"]) for e in encs)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, e in enumerate(encs):
                n = len(e["input_ids"])
                ids[j, :n] = torch.tensor(e["input_ids"])
                att[j, :n] = 1
            text_model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False)
            for j, (i, e) in enumerate(zip(idx, encs)):
                n = len(e["input_ids"])
                offs = e["offset_mapping"]
                masks = {}
                for name in ("act", "norm"):
                    a, b = rows[i]["spans"][name]
                    masks[name] = [t for t, (x, y) in enumerate(offs) if y > a and x < b and y > x]
                masks["story"] = list(range(min(min_story_tok, n - 1), n))
                act_ids = [e["input_ids"][t] for t in masks["act"]]
                meta.append({"i": i, "id": rows[i]["id"], "n_tok": n, "act_tok": len(masks["act"]),
                             "act_hash": hashlib.md5(json.dumps(act_ids[1:]).encode()).hexdigest()[:10]})
                for bi, blk in enumerate(blocks):
                    h = st[blk][j]
                    for name, m in masks.items():
                        out[name][i, bi] = h[m].float().mean(0).cpu().numpy()
                    out["last"][i, bi] = h[n - 1].float().cpu().numpy()
            if (s // batch) % 20 == 0:
                print(f"[stories] {s + len(idx)}/{N}", flush=True)
    meta.sort(key=lambda m: m["i"])
    return out, meta


# ------------------------------------------------------------------------------------------ transcripts
def build_sequence(tok, raw_msgs: list[dict], tools: list[dict], reasoning_effort: str = "high") -> dict[str, Any]:
    """context_rot.readout.build_sequence for transcripts that end on a tool result (no final report), with the
    env's tool list, plus each turn's tool-call argument span."""
    from context_rot.readout import convert

    msgs = convert(raw_msgs)

    def render(m, gen):
        return list(tok.apply_chat_template(m, tools=tools, tokenize=True, add_generation_prompt=gen,
                                            reasoning_effort=reasoning_effort, return_dict=False))

    seq = render(msgs, gen=False)
    msg_id = tok.convert_tokens_to_ids("<|message|>")
    end_id = tok.convert_tokens_to_ids("<|end|>")
    call_id = tok.convert_tokens_to_ids("<|call|>")
    steps = []
    for k, i in enumerate(i for i, m in enumerate(msgs) if m["role"] == "assistant"):
        p = render(msgs[:i], gen=True)
        if seq[: len(p)] != p:
            raise ValueError(f"step {k}: prompt is not a prefix of the run sequence (template drift)")
        s = {"step": k, "msg_index": i, "pre": len(p) - 1, "reason": None, "action": None}
        j = s["pre"] + 1
        cur = j
        if j + 2 < len(seq) and tok.decode(seq[j:j + 3]) == "<|channel|>analysis<|message|>":
            a = b = j + 3
            while b < len(seq) and seq[b] not in (end_id, call_id):
                b += 1
            if b > a:
                s["reason"] = [a, b]
            cur = b + 1
        # tool call: the next <|message|> ... <|call|> (arguments JSON)
        m = cur
        while m < len(seq) and seq[m] != msg_id and seq[m] != call_id:
            m += 1
        if m < len(seq) and seq[m] == msg_id:
            c = m + 1
            while c < len(seq) and seq[c] != call_id and seq[c] != end_id:
                c += 1
            if c < len(seq) and seq[c] == call_id and c > m + 1:
                s["action"] = [m + 1, c]
        steps.append(s)
    return {"ids": seq, "steps": steps}


class SpanReader:
    """Chunked forward (KV cache) over one sequence, pooling block outputs at points and over spans."""

    def __init__(self, model, blocks: list[int], chunk_budget_bytes: float = 4e9, max_chunk: int = 2048):
        from eforensics.hooks import find_text_model

        self.model = model
        self.text = find_text_model(model)
        self.blocks = sorted(blocks)
        self.budget = chunk_budget_bytes
        self.max_chunk = max_chunk
        cfg = getattr(model.config, "text_config", None) or model.config
        self.n_heads = cfg.num_attention_heads

    def read(self, ids: list[int], steps: list[dict], span_keys=("reason", "action")) -> dict[str, np.ndarray]:
        import torch
        from transformers import DynamicCache

        from eforensics.hooks import capture_block_outputs

        T, nS, nB = len(ids), len(steps), len(self.blocks)
        H = self.text.embed_tokens.weight.shape[1]
        pre = np.zeros((nS, nB, H), np.float32)
        sums = {k: np.zeros((nS, nB, H), np.float64) for k in span_keys}
        cnts = {k: np.zeros(nS, np.int64) for k in span_keys}
        want_pre = {s["pre"]: k for k, s in enumerate(steps)}
        spans = [(key, k, s[key][0], s[key][1]) for key in span_keys for k, s in enumerate(steps) if s.get(key)]
        dev = self.text.embed_tokens.weight.device
        cache = DynamicCache(config=self.model.config)
        C = max(64, min(self.max_chunk, int(self.budget / (self.n_heads * max(T, 1) * 2))))
        with torch.inference_mode(), capture_block_outputs(self.model, self.blocks) as st:
            for a in range(0, T, C):
                b = min(T, a + C)
                x = torch.tensor([ids[a:b]], device=dev)
                pos = torch.arange(a, b, device=dev)
                self.text(input_ids=x, past_key_values=cache, use_cache=True, cache_position=pos,
                          position_ids=pos[None])
                for t in range(a, b):
                    if t in want_pre:
                        for j, blk in enumerate(self.blocks):
                            pre[want_pre[t], j] = st[blk][0, t - a].float().cpu().numpy()
                for key, k, s0, s1 in spans:
                    lo, hi = max(a, s0), min(b, s1)
                    if lo < hi:
                        for j, blk in enumerate(self.blocks):
                            sums[key][k, j] += st[blk][0, lo - a:hi - a].double().sum(0).cpu().numpy()
                        cnts[key][k] += hi - lo
        out = {"pre": pre, "n_tokens": np.array(T), "chunk": np.array(C)}
        for key in span_keys:
            out[key] = (sums[key] / np.maximum(cnts[key], 1)[:, None, None]).astype(np.float32)
            out[f"{key}_tokens"] = cnts[key]
        return out


def read_transcript(reader: SpanReader, tok, msgs_path: Path, env: str, out_path: Path) -> dict:
    raw = json.loads(msgs_path.read_text())
    raw = raw if isinstance(raw, list) else raw["messages"]
    seq = build_sequence(tok, raw, TOOLS[env])
    res = reader.read(seq["ids"], seq["steps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pre=res["pre"].astype(np.float16), reason=res["reason"].astype(np.float16),
                        action=res["action"].astype(np.float16), reason_tokens=res["reason_tokens"],
                        action_tokens=res["action_tokens"], blocks=np.array(reader.blocks),
                        steps=np.array(json.dumps(seq["steps"])), n_tokens=res["n_tokens"])
    return {"n_steps": len(seq["steps"]), "n_tokens": len(seq["ids"]), "chunk": int(res["chunk"])}
