"""An independent reward-hacking direction from Goodfire's labelled rollouts (arXiv 2609.19101,
huggingface.co/datasets/Goodfire/reward-hack-data), read through OUR models.

Their ImpossibleBench rollouts (Kimi K3, GLM 5.2, Qwen 3.8 Max) carry passage-level judge labels
(3 votes; consensus positive / negative / ambiguous) on every reasoning / content / tool-call channel
of every sampled assistant step. We:
  1. rebuild each rollout's actual conversation (the node chain to the leaf; each assistant node is
     replaced by its SAMPLED sibling, which holds the structured reasoning / content / tool calls and
     is the node the labels point at); turns with several tool calls are split one call per turn;
  2. label each assistant step: hack = any passage positive, honest = all passages negative, else skip;
  3. render the conversation in the TARGET model's chat template (thinking kept) up to the last tool
     step and locate every step's tokens: [end of its generation prompt, end of the step);
  4. read the target model's residuals (block outputs) at the step's generation prompt ("pre") and
     averaged over the step's tokens ("step"), and save them with the labels.
The reward-hacking direction = mean over hack steps minus mean over honest steps (computed later, on
CPU, so it can be done per layer, per source model, with held-out splits).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def _tool_calls(raw) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw)
    out = []
    for tc in raw:
        fn = tc.get("function") or tc
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        out.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args or {}})
    return out


def conversation(trace: dict) -> tuple[list[dict], list[int | None]]:
    """(messages in the target-neutral form, sampled node index per message (assistant) or None)."""
    nodes = trace["nodes"]
    children = {}
    for i, n in enumerate(nodes):
        children.setdefault(n.get("parent"), []).append(i)
    # the leaf of the actual conversation: the last node; walk parents to the root
    chain, i = [], len(nodes) - 1
    while i is not None:
        chain.append(i)
        i = nodes[i].get("parent")
    chain.reverse()
    msgs, src = [], []
    for i in chain:
        m = nodes[i]["message"]
        role = m["role"]
        if role == "assistant":
            sib = [j for j in children.get(nodes[i].get("parent"), []) if nodes[j].get("sampled")
                   and nodes[j]["message"]["role"] == "assistant"]
            j = sib[0] if sib else i
            sm = nodes[j]["message"]
            msgs.append({"role": "assistant", "content": (sm.get("content") or "").strip(),
                         "reasoning": (sm.get("reasoning_content") or "").strip(),
                         "tool_calls": _tool_calls(sm.get("tool_calls"))})
            src.append(j if nodes[j].get("sampled") else None)
        elif role == "tool":
            msgs.append({"role": "tool", "content": m.get("content") or "", "tool_call_id": m.get("tool_call_id"),
                         "name": m.get("name")})
            src.append(None)
        else:
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict))
            msgs.append({"role": role, "content": c or ""})
            src.append(None)
    return msgs, src


def step_labels(passages: list[dict]) -> dict[int, str]:
    """node_index -> hack / honest / ambiguous."""
    by = {}
    for p in passages:
        by.setdefault(int(p["node_index"]), []).append(p["consensus"])
    out = {}
    for n, cs in by.items():
        out[n] = "hack" if "positive" in cs else ("honest" if all(c == "negative" for c in cs) else "ambiguous")
    return out


def to_template(msgs: list[dict], family: str) -> list[dict]:
    """Target-template messages; assistant turns with k tool calls become k turns (reasoning / text on
    the first), each followed by its tool result, so every template can render them."""
    out = []
    pending: dict[str, dict] = {}
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] != "assistant":
            out.append({"role": m["role"], "content": m["content"]})
            i += 1
            continue
        calls = m["tool_calls"]
        results = []
        j = i + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            results.append(msgs[j])
            j += 1
        if not calls:
            out.append(_assistant(m["reasoning"], m["content"], [], family, step_of=i))
            i = j
            continue
        for k, tc in enumerate(calls):
            out.append(_assistant(m["reasoning"] if k == 0 else "", m["content"] if k == 0 else "", [tc], family,
                                  step_of=i))
            res = next((r for r in results if r.get("tool_call_id") == tc["id"]), results[k] if k < len(results) else None)
            out.append({"role": "tool", "content": (res or {}).get("content", ""), "name": tc["name"],
                        "tool_call_id": tc["id"]})
        i = j
    return out


def _assistant(reasoning: str, content: str, calls: list[dict], family: str, step_of: int) -> dict:
    x: dict[str, Any] = {"role": "assistant", "_step_of": step_of}
    tcs = [{"type": "function", "id": c["id"], "function": {"name": c["name"], "arguments": c["arguments"]}}
           for c in calls]
    if family == "gptoss":
        th = reasoning
        if tcs and content:                     # the gpt-oss template refuses content + thinking with tools
            th = (th + "\n\n" + content).strip()
            content = ""
        if th:
            x["thinking"] = th
        x["content"] = content
    else:                                       # qwen: reasoning_content
        x["content"] = content
        if reasoning:
            x["reasoning_content"] = reasoning
    if tcs:
        x["tool_calls"] = tcs
    return x


def openai_tools(trace: dict) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("parameters", {"type": "object"})}}
            for t in trace.get("tools") or []]


def render(tok, msgs, tools, gen: bool, family: str) -> str:
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]
    # qwen: keep every turn's reasoning (inside one tool chain the model sees it anyway)
    kw = {"reasoning_effort": "high"} if family == "gptoss" else {"enable_thinking": True, "preserve_thinking": True}
    return tok.apply_chat_template(clean, tools=tools, tokenize=False, add_generation_prompt=gen, **kw)


def build(tok, trace: dict, passages: list[dict], family: str, max_tokens: int = 60000) -> dict | None:
    """Token ids + steps [{pre, reason=[a,b], label, node}] for one rollout, or None if it cannot be
    rendered prefix-consistently. Steps are located by CHARACTER offsets in the rendered text and mapped to
    tokens (a template can merge the generation prompt's last newline with the step's first token, e.g.
    Qwen's empty "<think>\\n\\n</think>"): pre = the last token that starts before the step's text,
    step tokens = tokens starting inside [prompt end, step end)."""
    import bisect

    msgs, src = conversation(trace)
    labels = step_labels(passages)
    tmsgs = to_template(msgs, family)
    tools = openai_tools(trace)
    # cut after the last assistant turn that has a tool call (a final answer would drop reasoning)
    last = max((i for i, m in enumerate(tmsgs) if m["role"] == "assistant" and m.get("tool_calls")), default=None)
    if last is None:
        return None
    tmsgs = tmsgs[: last + 1]
    text = render(tok, tmsgs, tools, gen=False, family=family)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, starts = list(enc["input_ids"]), [a for a, _ in enc["offset_mapping"]]
    if len(ids) > max_tokens:
        return None
    steps = []
    for i, m in enumerate(tmsgs):
        if m["role"] != "assistant":
            continue
        p = render(tok, tmsgs[:i], tools, gen=True, family=family)
        e = render(tok, tmsgs[: i + 1], tools, gen=False, family=family)
        # the prompt may end inside a token the step shares (newline merging): compare up to its last \n
        p_cmp = p.rstrip("\n")
        if not text.startswith(p_cmp) or not text.startswith(e) or len(e) <= len(p_cmp):
            return None
        a = bisect.bisect_left(starts, len(p_cmp))          # first token starting at/after the prompt
        b = bisect.bisect_left(starts, len(e))               # first token after the step
        if a == 0 or b <= a:
            return None
        node = src[m["_step_of"]]
        steps.append({"pre": a - 1, "reason": [a, b], "node": node,
                      "label": labels.get(node, "ambiguous") if node is not None else "ambiguous",
                      "orig_step": m["_step_of"]})
    return {"ids": ids, "steps": steps}


def load_source(d: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    traces = [json.loads(l)["traces"][0] for l in open(d / "traces.jsonl")]
    passages: dict[str, list[dict]] = {}
    for l in open(d / "labels" / "judge_passages.jsonl"):
        p = json.loads(l)
        passages.setdefault(p["trace_id"], []).append(p)
    return traces, passages
