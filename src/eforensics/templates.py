"""Chat-template rendering and splitting a generation into reasoning/answer (copied from
timf34/grader-shifts-persona-space@0ad65c4 src/gshift/templates.py, plus last_user_span).

The exact token ids that vLLM generated from are rendered here once, stored with every
record, and re-used verbatim by the activation extractor. Nothing downstream re-applies
the chat template to generated text (Qwen chat templates rewrite past assistant turns,
e.g. strip reasoning, so re-templating would not reproduce what the model saw).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class TemplateInfo:
    """What the chat template does in thinking mode, recorded with every run."""

    think_open_id: int | None
    think_close_id: int | None
    end_ids: list[int]                 # tokens that terminate a turn (eos, <|im_end|>, ...)
    special_ids: list[int]             # every special/added-special token id
    generation_prompt_suffix: str      # what add_generation_prompt appends (thinking on)
    prompt_opens_think: bool           # generation prompt already ends inside <think>
    chat_template_sha256: str
    notes: list[str] = field(default_factory=list)
    thinking: bool = True              # inspected with enable_thinking (False: no reasoning expected)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_id(tokenizer, text: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(text)
    if tid is None or (tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id):
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tid if ids == [tid] else None


def render_prompt(tokenizer, messages: Sequence[dict[str, str]], enable_thinking: bool = True,
                  template_kwargs: dict[str, Any] | None = None) -> str:
    """Chat-templated prompt with the generation prompt appended. `template_kwargs` are
    model-specific template switches (e.g. Qwen3.8 reasoning_effort) from models.yaml."""
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        **(template_kwargs or {}),
    )


def prompt_token_ids(tokenizer, messages: Sequence[dict[str, str]], enable_thinking: bool = True,
                     template_kwargs: dict[str, Any] | None = None) -> list[int]:
    text = render_prompt(tokenizer, messages, enable_thinking, template_kwargs)
    # Same convention as public-steering-vectors: tokenizer(templated) with no extra BOS.
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def inspect_template(tokenizer, enable_thinking: bool = True,
                     template_kwargs: dict[str, Any] | None = None) -> TemplateInfo:
    import hashlib

    kw = dict(template_kwargs or {})
    msgs = [{"role": "user", "content": "Hello."}]
    with_gen = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking, **kw
    )
    without = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking, **kw
    )
    suffix = with_gen[len(without):] if with_gen.startswith(without) else with_gen[-64:]
    open_id = _token_id(tokenizer, THINK_OPEN)
    close_id = _token_id(tokenizer, THINK_CLOSE)
    end_ids = set()
    for t in ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>"):
        tid = _token_id(tokenizer, t)
        if tid is not None:
            end_ids.add(tid)
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        end_ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        end_ids.update(int(x) for x in eos)
    special = set(int(x) for x in getattr(tokenizer, "all_special_ids", []))
    # added special tokens (e.g. <|im_start|>, <think>) are not always in all_special_ids
    for tid, tok in getattr(tokenizer, "added_tokens_decoder", {}).items():
        if getattr(tok, "special", False):
            special.add(int(tid))
    special.update(end_ids)
    notes = []
    stripped = suffix.rstrip("\n")
    opens = stripped.endswith(THINK_OPEN)
    if enable_thinking and not opens:
        notes.append(
            "generation prompt does not end with <think>: the model is expected to emit "
            "<think> itself (handled by split_generation)"
        )
    if with_gen.startswith("<|im_start|>system"):
        notes.append("template injects a system message into a user-only conversation "
                     f"(template_kwargs={kw}); check models.yaml chat_template_kwargs")
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(tmpl, dict):
        tmpl = "\n".join(f"{k}:{v}" for k, v in sorted(tmpl.items()))
    return TemplateInfo(
        thinking=bool(enable_thinking),
        think_open_id=open_id,
        think_close_id=close_id,
        end_ids=sorted(end_ids),
        special_ids=sorted(special),
        generation_prompt_suffix=suffix,
        prompt_opens_think=opens,
        chat_template_sha256=hashlib.sha256(tmpl.encode()).hexdigest(),
        notes=notes,
    )


@dataclass
class Split:
    reasoning: list[int]        # positions (relative to generation start) of reasoning tokens
    answer: list[int]           # positions of final-answer tokens
    think_closed: bool          # saw </think>
    has_answer: bool            # think closed AND at least one answer token
    n_generated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reasoning_tokens": len(self.reasoning),
            "n_answer_tokens": len(self.answer),
            "think_closed": self.think_closed,
            "has_answer": self.has_answer,
            "n_generated": self.n_generated,
        }


def _trim_whitespace(positions: list[int], gen_ids: Sequence[int], tokenizer) -> list[int]:
    """Drop whitespace-only tokens at both ends (e.g. the "\\n\\n" after </think>)."""
    lo, hi = 0, len(positions)
    while lo < hi and tokenizer.decode([gen_ids[positions[lo]]]).strip() == "":
        lo += 1
    while hi > lo and tokenizer.decode([gen_ids[positions[hi - 1]]]).strip() == "":
        hi -= 1
    return positions[lo:hi]


def split_generation(gen_ids: Sequence[int], tokenizer, info: TemplateInfo) -> Split:
    """Locate reasoning and final-answer tokens inside a thinking-mode generation.

    Reasoning: tokens strictly between the (prompt- or model-emitted) <think> and </think>.
    Answer: tokens after </think>, excluding special tokens and boundary whitespace.
    If </think> never appears (truncated mid-reasoning) there is no answer.
    """
    gen_ids = list(gen_ids)
    n = len(gen_ids)
    special = set(info.special_ids)
    end = n
    while end > 0 and gen_ids[end - 1] in special:
        end -= 1
    start = 0
    if info.think_open_id is not None and info.think_open_id in gen_ids[:end]:
        start = gen_ids.index(info.think_open_id) + 1
    close = None
    if info.think_close_id is not None:
        for i in range(start, end):
            if gen_ids[i] == info.think_close_id:
                close = i
                break
    if close is None:
        if not info.thinking and (info.think_open_id is None or info.think_open_id not in gen_ids[:end]):
            # thinking OFF (prompt already closed an empty <think></think>): everything is answer
            answer = _trim_whitespace([i for i in range(0, end) if gen_ids[i] not in special], gen_ids, tokenizer)
            return Split([], answer, True, len(answer) > 0, n)
        reasoning = [i for i in range(start, end) if gen_ids[i] not in special]
        reasoning = _trim_whitespace(reasoning, gen_ids, tokenizer)
        return Split(reasoning, [], False, False, n)
    reasoning = [i for i in range(start, close) if gen_ids[i] not in special]
    reasoning = _trim_whitespace(reasoning, gen_ids, tokenizer)
    answer = [i for i in range(close + 1, end) if gen_ids[i] not in special]
    answer = _trim_whitespace(answer, gen_ids, tokenizer)
    return Split(reasoning, answer, True, len(answer) > 0, n)


def answer_text(gen_ids: Sequence[int], tokenizer, split: Split) -> str:
    if not split.answer:
        return ""
    return tokenizer.decode([gen_ids[i] for i in range(split.answer[0], split.answer[-1] + 1)],
                            skip_special_tokens=True).strip()


def reasoning_text(gen_ids: Sequence[int], tokenizer, split: Split) -> str:
    if not split.reasoning:
        return ""
    return tokenizer.decode([gen_ids[i] for i in range(split.reasoning[0], split.reasoning[-1] + 1)],
                            skip_special_tokens=True).strip()


def last_user_span(prompt_ids: Sequence[int], tokenizer) -> tuple[int, int]:
    """[start, end) positions of the CONTENT tokens of the last user message in a rendered
    ChatML prompt: after "<|im_start|>user\\n", up to (not including) its "<|im_end|>"."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = list(prompt_ids)
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start and i + 2 < len(ids) and tokenizer.decode([ids[i + 1]]) == "user":
            start = i + 2
            if tokenizer.decode([ids[start]]) == "\n":
                start += 1
            end = ids.index(im_end, start) if im_end in ids[start:] else len(ids)
            if end <= start:
                raise ValueError("empty last user message")
            return start, end
    raise ValueError("no user message in prompt")
