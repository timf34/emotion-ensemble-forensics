"""Emotion-probe readout along agent-interp-envs rollouts (GPT-OSS, harmony format).

For a run's final/messages.json (OpenAI chat format from the OpenRouter provider) we rebuild the
token sequence the model conditioned on at every step and run ONE chunked forward pass over it:

  * step k's prompt = harmony render of messages[:i_k] with the generation prompt; until the final
    report the template keeps every earlier analysis (reasoning) message, so step prompts are
    prefixes of each other (checked) and one pass over the last prompt covers all of them;
  * the final report turn (analysis + final channel) is appended as text, because rendering it
    through the template would drop all earlier reasoning.

Saved per step, at each block of `blocks` (block b = output of decoder block b = HF hidden_states[b+1],
the emotion-vector convention):
  pre     residual at the last token of step k's prompt (`<|start|>assistant`), before step k is written
  reason  mean residual over step k's reasoning (analysis-channel) tokens; zeros if none

Chunked prefill with a KV cache keeps attention memory at chunk x context instead of context^2.
"""

from __future__ import annotations

import json
import re
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
TOOLS = [EXECUTE_COMMAND_TOOL]


def strip_think(s: str | None) -> str:
    s = (s or "").strip()
    s = re.sub(r"^<think>\s*", "", s)
    s = re.sub(r"\s*</think>$", "", s)
    return s


def convert(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenRouter/OpenAI messages -> the fields the gpt-oss chat template reads."""
    out = []
    for m in msgs:
        r = m["role"]
        if r == "assistant":
            x: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            th = strip_think(m.get("reasoning") or m.get("reasoning_content"))
            if th:
                x["thinking"] = th
            tcs = []
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args = {"command": args}
                tcs.append({"type": "function", "function": {"name": tc["function"]["name"], "arguments": args}})
            if tcs:
                x["tool_calls"] = tcs
                # the gpt-oss template refuses visible content next to thinking in a tool-call turn;
                # such text is a commentary preamble the model wrote while thinking: fold it in
                if x["content"].strip() and x.get("thinking"):
                    x["thinking"] = x["thinking"] + "\n\n" + x["content"].strip()
                    x["content"] = ""
                    x["_folded_content"] = True
            out.append(x)
        elif r == "tool":
            out.append({"role": "tool", "content": m.get("content") or ""})
        else:
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict))
            out.append({"role": r, "content": c or ""})
    return out


def render(tok, msgs, gen: bool, reasoning_effort: str = "high") -> list[int]:
    ids = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=True, add_generation_prompt=gen,
                                  reasoning_effort=reasoning_effort, return_dict=False)
    return list(ids)


def build_sequence(tok, raw_msgs: list[dict[str, Any]], reasoning_effort: str = "high") -> dict[str, Any]:
    """Token ids + per-step (prompt end, reasoning span) for one run."""
    msgs = convert(raw_msgs)
    steps_i = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    final_i = steps_i[-1] if steps_i and not msgs[steps_i[-1]].get("tool_calls") else None
    tool_steps = [i for i in steps_i if i != final_i]
    upto = final_i if final_i is not None else len(msgs)
    seq = render(tok, msgs[:upto], gen=final_i is not None)          # every tool step, reasoning kept
    if final_i is None:
        seq = render(tok, msgs[:upto], gen=False)
    msg_id = tok.convert_tokens_to_ids("<|message|>")
    end_id = tok.convert_tokens_to_ids("<|end|>")
    call_id = tok.convert_tokens_to_ids("<|call|>")
    steps = []
    for k, i in enumerate(tool_steps + ([final_i] if final_i is not None else [])):
        p = render(tok, msgs[:i], gen=True)
        if seq[: len(p)] != p:
            raise ValueError(f"step {k}: prompt is not a prefix of the run sequence (template drift)")
        steps.append({"step": k, "msg_index": i, "pre": len(p) - 1, "reason": None})
    if final_i is not None:     # append the final report turn as generated
        fm = msgs[final_i]
        tail = ""
        if fm.get("thinking"):
            tail += f"<|channel|>analysis<|message|>{fm['thinking']}<|end|><|start|>assistant"
        tail += f"<|channel|>final<|message|>{fm.get('content') or ''}<|return|>"
        seq = seq + tok(tail, add_special_tokens=False)["input_ids"]
    # reasoning spans: the first analysis message after each step's prompt
    for s in steps:
        j = s["pre"] + 1
        if j + 2 < len(seq) and tok.decode(seq[j:j + 3]) == "<|channel|>analysis<|message|>":
            a = j + 3
            b = a
            while b < len(seq) and seq[b] not in (end_id, call_id):
                b += 1
            if b > a:
                s["reason"] = [a, b]
    return {"ids": seq, "steps": steps, "n_tool_steps": len(tool_steps), "has_final": final_i is not None}


class Reader:
    """Chunked forward over one long sequence, keeping block outputs only where needed."""

    def __init__(self, model, blocks: list[int], chunk_budget_bytes: float = 3e9, max_chunk: int = 2048):
        from eforensics.hooks import find_text_model

        self.model = model
        self.text = find_text_model(model)
        self.blocks = sorted(blocks)
        self.budget = chunk_budget_bytes
        self.max_chunk = max_chunk
        cfg = getattr(model.config, "text_config", None) or model.config
        self.n_heads = cfg.num_attention_heads

    def chunk_size(self, total: int) -> int:
        # eager attention materialises [heads, chunk, context] in bf16
        c = int(self.budget / (self.n_heads * max(total, 1) * 2))
        return max(64, min(self.max_chunk, c))

    def read(self, ids: list[int], steps: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        import torch
        from transformers import DynamicCache

        from eforensics.hooks import capture_block_outputs

        T = len(ids)
        nB = len(self.blocks)
        H = self.text.embed_tokens.weight.shape[1]
        pre = np.zeros((len(steps), nB, H), np.float32)
        rsum = np.zeros((len(steps), nB, H), np.float64)
        rcnt = np.zeros(len(steps), np.int64)
        want_pre = {s["pre"]: k for k, s in enumerate(steps)}
        spans = [(k, s["reason"][0], s["reason"][1]) for k, s in enumerate(steps) if s["reason"]]
        dev = self.text.embed_tokens.weight.device
        cache = DynamicCache(config=self.model.config)
        C = self.chunk_size(T)
        with torch.inference_mode(), capture_block_outputs(self.model, self.blocks) as st:
            for a in range(0, T, C):
                b = min(T, a + C)
                x = torch.tensor([ids[a:b]], device=dev)
                pos = torch.arange(a, b, device=dev)
                self.text(input_ids=x, past_key_values=cache, use_cache=True, cache_position=pos,
                          position_ids=pos[None])
                for t in range(a, b):
                    if t in want_pre:
                        k = want_pre[t]
                        for j, blk in enumerate(self.blocks):
                            pre[k, j] = st[blk][0, t - a].float().cpu().numpy()
                for k, s0, s1 in spans:
                    lo, hi = max(a, s0), min(b, s1)
                    if lo < hi:
                        for j, blk in enumerate(self.blocks):
                            rsum[k, j] += st[blk][0, lo - a:hi - a].double().sum(0).cpu().numpy()
                        rcnt[k] += hi - lo
        reason = (rsum / np.maximum(rcnt, 1)[:, None, None]).astype(np.float32)
        return {"pre": pre, "reason": reason, "reason_tokens": rcnt, "chunk": np.array(C), "n_tokens": np.array(T)}


def read_run(reader: Reader, tok, run_dir: Path, out_path: Path, reasoning_effort: str = "high") -> dict[str, Any]:
    raw = json.loads((run_dir / "final" / "messages.json").read_text())
    raw = raw if isinstance(raw, list) else raw["messages"]
    seq = build_sequence(tok, raw, reasoning_effort)
    res = reader.read(seq["ids"], seq["steps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pre=res["pre"].astype(np.float16), reason=res["reason"].astype(np.float16),
                        reason_tokens=res["reason_tokens"], blocks=np.array(reader.blocks),
                        steps=np.array(json.dumps(seq["steps"])), n_tokens=res["n_tokens"])
    return {"run": str(run_dir), "n_steps": len(seq["steps"]), "n_tokens": len(seq["ids"]), "chunk": int(res["chunk"])}
