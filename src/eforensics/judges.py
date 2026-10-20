"""School of Reward Hacks judges, one Claude model for all three:

  metric   Betley et al.'s judge 1 (verbatim): sees the SCORED prompt, scores how literally the
           answer satisfies the stated rule (0-100)
  quality  Betley et al.'s judge 2 (verbatim): sees the DESCORED prompt where one exists
           (else the scored prompt), scores quality (0-100)
  exploit  data/srh/judge3_exploit_prompt.txt (frozen 2026-10-20): did the answer sacrifice the
           task to game the evaluation? exploit / no_exploit / ambiguous, plus refusal

Only the visible answer is judged. A record with no answer is never sent; an unparseable
judge reply is stored as unparsed, never turned into a number. Resumable per (id, judge).
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from pathlib import Path
from typing import Any

from . import DATA_DIR
from .io import append_jsonl, read_jsonl

_SCORE = re.compile(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>")
_VERDICT = re.compile(r"<verdict>\s*(exploit|no_exploit|ambiguous)\s*</verdict>", re.I)
_REFUSAL = re.compile(r"<refusal>\s*(yes|no)\s*</refusal>", re.I)
JUDGES = ("metric", "quality", "exploit")


def template(name: str) -> str:
    return (DATA_DIR / "srh" / name).read_text()


def parse_score(text: str | None) -> float | None:
    m = _SCORE.findall(text or "")
    if not m:
        return None
    v = float(m[-1])
    return v if 0 <= v <= 100 else None


def parse_exploit(text: str | None) -> dict[str, Any] | None:
    v = _VERDICT.findall(text or "")
    if not v:
        return None
    r = _REFUSAL.findall(text or "")
    return {"verdict": v[-1].lower(), "refusal": (r[-1].lower() == "yes") if r else None}


class JudgeClient:
    """Anthropic (ANTHROPIC_API_KEY) or, without that key, OpenRouter routed to Anthropic only."""

    def __init__(self):
        self.provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openrouter"
        if self.provider == "anthropic":
            import anthropic

            self._c = anthropic.AsyncAnthropic()
        else:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("set ANTHROPIC_API_KEY or OPENROUTER_API_KEY (.env)")
            from openai import AsyncOpenAI

            self._c = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=0, timeout=600)

    async def chat(self, model: str, prompt: str, max_tokens: int) -> str:
        msgs = [{"role": "user", "content": prompt}]
        if self.provider == "anthropic":
            r = await self._c.messages.create(model=model, max_tokens=max_tokens, messages=msgs)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        r = await self._c.chat.completions.create(
            model=model if "/" in model else f"anthropic/{model}", messages=msgs, max_tokens=max_tokens,
            extra_body={"provider": {"only": ["anthropic"], "allow_fallbacks": False}})
        return r.choices[0].message.content or ""


def judge_prompts(rec: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, str]:
    ans = rec["answer_text"]
    scored = manifest_row["user"]
    seen = rec["messages"][-1]["content"]              # what the model was actually asked
    descored = manifest_row.get("user_descored") or scored
    return {
        "metric": template("judge1_metric_prompt.txt").replace("{prompt}", scored).replace("{answer}", ans),
        "quality": template("judge2_quality_prompt.txt").replace("{prompt}", descored).replace("{answer}", ans),
        "exploit": (template("judge3_exploit_prompt.txt").replace("{evaluation_metric}", manifest_row["evaluation_metric"])
                    .replace("{cheat_method}", manifest_row["cheat_method"]).replace("{prompt}", seen)
                    .replace("{answer}", ans)),
    }


async def _one(client, cfg, name, prompt) -> dict[str, Any]:
    parse = parse_exploit if name == "exploit" else parse_score
    reply = None
    for a in range(1, int(cfg["max_attempts"]) + 1):
        try:
            reply = await client.chat(cfg["model"], prompt, int(cfg["max_tokens"]))
            v = parse(reply)
            if v is not None:
                return {"value": v, "reply": reply, "attempts": a, "unparsed": False}
        except Exception as e:
            reply = f"ERROR: {e!r}"
            await asyncio.sleep(min(60, 2 ** a + random.random()))
    return {"value": None, "reply": reply, "attempts": int(cfg["max_attempts"]), "unparsed": True}


def run_judges(responses: list[dict[str, Any]], manifest: dict[str, dict[str, Any]], out_path: Path,
               cfg: dict[str, Any], judges: tuple[str, ...] = JUDGES) -> int:
    done = {(r["id"], r["judge"]) for r in read_jsonl(out_path) if not r.get("unparsed")}
    items = []
    for rec in responses:
        if not rec.get("has_answer") or not rec.get("answer_text", "").strip():
            continue
        prompts = judge_prompts(rec, manifest[rec["task_id"]])
        items += [(rec["id"], j, prompts[j]) for j in judges if (rec["id"], j) not in done]
    print(f"[judge] {len(items)} judge calls to make ({cfg['model']})", flush=True)
    if not items:
        return 0

    async def main():
        client = JudgeClient()
        sem = asyncio.Semaphore(int(cfg["concurrency"]))
        buf, lock = [], asyncio.Lock()

        async def go(rid, name, prompt):
            async with sem:
                res = await _one(client, cfg, name, prompt)
            async with lock:
                buf.append({"id": rid, "judge": name, "judge_model": cfg["model"], **res})
                if len(buf) >= 50:
                    append_jsonl(out_path, buf)
                    buf.clear()

        await asyncio.gather(*(go(*it) for it in items))
        if buf:
            append_jsonl(out_path, buf)

    asyncio.run(main())
    return len(items)
