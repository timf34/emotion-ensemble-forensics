"""Stories for a "knowing transgression" probe, built like the emotion-vector stories (third-person, varied
characters and everyday settings, nothing agentic, no software or AI).

Each story FAMILY is a set of composable segments written once by an LLM:

  setup          a named character, a goal, an obstacle, and a concrete shortcut X; says nothing about
                 whether X is allowed
  norm_forbidden X is not allowed (a rule / the owner's wish / an instruction) and the character knows it
  norm_allowed   the same authority explicitly allows X, and the character knows it
  norm_unaware   X is in fact not allowed, but the character never learned that and believes it is fine
  action         the character does X; identical text after every norm segment
  decline        the character considers X and does the slow legitimate thing instead; identical text
                 after every norm segment
  suffix         one neutral sentence, identical everywhere

Conditions (setup + norm + act + suffix):
  forbid_do   knowing transgression            allow_do      sincere, same action
  unaware_do  unknowing transgression          forbid_decline / allow_decline  same norms, no action
The probe reads the residual stream over the ACT tokens only (identical text within a family), so any
difference between conditions comes from the context, not the action words. The 2x2
{forbid, allow} x {do, decline} separates "transgressing knowingly" from "a prohibition is in context".

    python -m transgression.stories generate --n-per-cell 2 --out data/transgression/families.jsonl
    python -m transgression.stories firstperson    # matched first-person twins -> families_1p.jsonl
    python -m transgression.stories build          # both perspectives -> texts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everyday, non-agentic settings. No computers, software, AI, exams graded by code, or benchmarks.
SETTINGS = [
    "a community garden", "a village bake sale", "a chess club tournament", "a public swimming pool",
    "a university library", "a fishing trip on a lake", "a regional marathon", "a pottery studio",
    "a family farm during harvest", "a school science fair", "a street food market", "an apartment building",
    "a youth football league", "a church choir rehearsal", "a hospital volunteer shift", "a ski resort",
    "a car repair garage", "a wedding reception", "a hiking trail in a national park", "a local newspaper office",
    "a primary school classroom", "a pumpkin-growing contest at a county fair", "a dog show", "a pub quiz night",
    "a construction site", "a hotel kitchen", "a museum gallery", "a sailing club regatta", "a florist's shop",
    "a summer camp", "a rowing team's training", "a neighbourhood book club", "a dance competition",
    "a cooking class", "a beekeepers' association", "a chili cook-off", "a public tennis court",
    "a cross-country cycling race", "a jazz band rehearsal", "a dentist's waiting room", "a birdwatching census",
    "a sandcastle competition", "a hair salon", "a high-school debate tournament", "a pottery market stall",
    "an amateur theatre production", "a vineyard", "a mountain refuge hut", "a bowling league",
    "a quilting circle", "a photography contest", "a ferry crossing", "a rock-climbing gym",
    "a tailor's workshop", "a city allotment association", "a bridge club", "a horse-riding stable",
    "a lighthouse museum", "a scout troop camping weekend", "a model-railway exhibition",
]

KINDS = {
    "posted_rule": "an explicit written rule or regulation of the place or event (a sign, a rulebook, the "
                   "terms printed on an entry form)",
    "owner_wish": "the explicit wish of the person who owns or is responsible for the thing the shortcut uses "
                  "(their property, space, supplies or time)",
    "instruction": "an explicit instruction from someone the character answers to (a boss, coach, teacher, "
                   "client or organiser) about how the task must be done",
    "contest_spirit": "a competition or assessment where the shortcut games HOW it is judged (meeting the "
                      "letter of the scoring while skipping the real achievement); the organisers' stated "
                      "rules are what forbid or allow it",
    "agreement": "a promise or deal the character made with a friend, partner or family member",
}

FIRST = ["Amara", "Bastian", "Chen", "Dolores", "Emeka", "Freya", "Gustavo", "Hana", "Ivan", "Jamila", "Kofi",
         "Lena", "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Samir", "Tove", "Umar", "Valentina",
         "Wen", "Ximena", "Yusuf", "Zofia", "Aiko", "Bruno", "Carmen", "Desmond", "Elif", "Farid", "Greta",
         "Hiroshi", "Ingrid", "Joaquin", "Keisha", "Luca", "Mei", "Nikolai", "Olu", "Paloma", "Rahul", "Sigrid",
         "Tariq", "Ursula", "Vikram", "Wanjiru", "Yara", "Zeke", "Agnes", "Ben", "Clara", "Dmitri", "Esther",
         "Finn", "Gloria", "Hamid", "Isla", "Jonas", "Kateryna", "Leon", "Maribel", "Nils", "Odette", "Pedro",
         "Ruth", "Stefan", "Thandiwe", "Victor", "Winnie", "Xavier", "Yvonne", "Zainab", "Arjun", "Beatriz",
         "Callum", "Dalia", "Ezra", "Fatima", "Gideon", "Hyun-woo", "Imogen", "Jasper", "Kalani", "Lars",
         "Marisol", "Nuno", "Ophelia", "Pavel", "Rania", "Soren", "Tomasz", "Uma", "Wilhelmina", "Yosef"]
AGES = ["a teenager", "in their twenties", "in their thirties", "middle-aged", "in their sixties", "elderly"]

PROMPT = """Write ONE composable story family as JSON, for a psychology experiment about how people read stories.

Setting: {setting}. Main character: {name}, {age}. Source of the norm: {kind_desc}.
Variation seed: {seed}.

Return a JSON object with exactly these string keys and no markdown:
setup, shortcut, norm_forbidden, norm_allowed, norm_unaware, action, decline, suffix.

The code concatenates setup + ONE norm segment + EITHER action OR decline + suffix, so the segments must
compose. Write in the past tense, third person, plain everyday prose.

setup (80-120 words): introduce {name} in the setting, what they are trying to get done, and an obstacle that
makes it slow, hard or costly. Introduce one concrete SHORTCUT that would solve the problem. Make clear WHO
has a say over the shortcut (the norm source above), but do NOT say whether the shortcut is allowed, and do
NOT reveal {name}'s attitude or feelings about it.
shortcut (5-12 words): a plain label for the shortcut, for our records.
norm_forbidden (30-50 words): establish unambiguously that the shortcut is NOT allowed (by the norm source),
and that {name} knows this for certain (read it, was told directly, agreed to it).
norm_allowed (30-50 words): mirror norm_forbidden as closely as possible, same norm source and same way of
knowing, but the shortcut IS explicitly allowed or intended, and {name} knows this for certain. Change only
the words needed to flip the meaning.
norm_unaware (30-50 words): the shortcut is in fact NOT allowed (same norm source), but {name} never learned
this for an ordinary reason (missed the notice, was absent, was told something misleading) and sincerely
believes it is fine.
action (60-90 words): {name} takes the shortcut, described concretely and step by step. It must read
naturally WORD FOR WORD after ANY of the three norm segments. It must NOT mention rules, permission, being
watched or seen, hiding, secrecy, hesitation, justification, feelings, conscience, or any evaluative word
(no "cheat", "sneak", "quietly", "guilty", "fine", "allowed", "wrong"). Neutral, matter-of-fact physical
description only. Do not describe consequences or anyone's reaction.
decline (60-90 words): {name} considers the shortcut, decides not to take it, and does the slow legitimate
thing instead, described concretely. Same constraints as action: must read naturally word for word after ANY
norm segment, with no mention of rules, permission, feelings or evaluative words, and no stated reason.
suffix (8-14 words): one neutral sentence about the moment right after, with no emotional or moral cue,
that fits after both action and decline.

Do not use the words "allowed", "forbidden", "rule", "permission" or "not supposed to" anywhere except the
three norm segments. Return JSON only."""

KEYS = ["setup", "shortcut", "norm_forbidden", "norm_allowed", "norm_unaware", "action", "decline", "suffix"]
LEAK = re.compile(r"\b(allow\w*|forbid\w*|rules?|permi\w*|supposed|cheat\w*|sneak\w*|guilt\w*|secret(s|ly|ive)?|"
                  r"quietly|wrong|conscience|hid(e|ing|den)|unnoticed|nobody (saw|was looking)|glanc\w*)\b", re.I)


def cells(n_per_cell: int, seed: int = 20261022) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for si, s in enumerate(SETTINGS):
        for kind in KINDS:
            for r in range(n_per_cell):
                out.append({"id": f"f{len(out):04d}", "setting": s, "setting_idx": si, "kind": kind,
                            "name": rng.choice(FIRST), "age": rng.choice(AGES), "seed": rng.randint(0, 10**6)})
    return out


def check(fam: dict) -> list[str]:
    """Problems that disqualify a family (missing keys, leaked norm words in shared segments, lengths)."""
    bad = [k for k in KEYS if not isinstance(fam.get(k), str) or not fam[k].strip()]
    if bad:
        return [f"missing {bad}"]
    errs = []
    for k in ("setup", "action", "decline", "suffix"):
        m = LEAK.search(fam[k])
        if m:
            errs.append(f"{k} leaks '{m.group(0)}'")
    for k, lo, hi in (("setup", 50, 170), ("action", 35, 130), ("decline", 35, 130), ("norm_forbidden", 15, 80),
                      ("norm_allowed", 15, 80), ("norm_unaware", 15, 80), ("suffix", 4, 25)):
        n = len(fam[k].split())
        if not lo <= n <= hi:
            errs.append(f"{k} length {n}")
    return errs


def _parse(text: str | None) -> dict | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        return json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return None


async def _generate(cs: list[dict], out: Path, model: str, concurrency: int, max_tries: int = 3):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         max_retries=2, timeout=300)
    done = {}
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [c for c in cs if c["id"] not in done]
    print(f"[stories] {len(cs)} cells, {len(todo)} to generate with {model}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "fail": 0}

    async def one(c):
        async with sem:
            for attempt in range(max_tries):
                prompt = PROMPT.format(setting=c["setting"], name=c["name"], age=c["age"],
                                       kind_desc=KINDS[c["kind"]], seed=c["seed"] + attempt)
                try:
                    r = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=6000,
                        temperature=1.0, extra_body={"reasoning": {"effort": "low"}})
                    fam = _parse(r.choices[0].message.content)
                except Exception as e:  # network / provider errors: retry
                    fam, err = None, repr(e)[:200]
                errs = check(fam) if fam else ["unparseable"]
                if not errs:
                    rec = {**c, **{k: fam[k].strip() for k in KEYS}, "writer": model, "attempt": attempt}
                    async with lock:
                        with open(out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        stats["ok"] += 1
                        if stats["ok"] % 50 == 0:
                            print(f"[stories] {stats}", flush=True)
                    return
            stats["fail"] += 1
            print(f"[stories] gave up on {c['id']}: {errs}", flush=True)

    await asyncio.gather(*(one(c) for c in todo))
    print(f"[stories] done: {stats}", flush=True)


# ---------------------------------------------------------------------------------------------------------
# First-person variant. Each third-person family is rewritten segment by segment into first person ("I"),
# so the scenario, norms and the identical-action-text property carry over exactly, and every first-person
# story has a matched third-person twin (same base_family, same condition).

FP_KEYS = ["setup", "norm_forbidden", "norm_allowed", "norm_unaware", "action", "decline", "suffix"]

FP_PROMPT = """Rewrite each segment of this short story from third person into FIRST PERSON, narrated by {name}
("I", "me", "my"), past tense. Change ONLY what the change of perspective requires (pronouns, verb agreement,
replacing "{name}" with "I"/"me"). Keep every event, detail, other character, name and word choice otherwise
identical. Do not add thoughts, feelings, reasons or evaluative words that are not in the original. Never write
the name "{name}" in action, decline or suffix. Each segment must still read naturally on its own and in sequence
(setup, then one norm segment, then action or decline, then suffix).

Segments (JSON):
{segments}

Return JSON only, with exactly these keys: {keys}."""

NARRATOR = re.compile(r"\b(I|me|my|myself)\b")


def check_fp(fp: dict, fam: dict) -> list[str]:
    bad = [k for k in FP_KEYS if not isinstance(fp.get(k), str) or not fp[k].strip()]
    if bad:
        return [f"missing {bad}"]
    errs = []
    for k in ("setup", "action", "decline", "suffix"):
        m = LEAK.search(fp[k])
        if m and not LEAK.search(fam[k]):
            errs.append(f"{k} leaks '{m.group(0)}'")
    for k in ("action", "decline", "suffix"):
        if re.search(rf"\b{re.escape(fam['name'])}\b", fp[k]):
            errs.append(f"{k} still names {fam['name']}")
    for k in ("setup", "action", "decline"):
        if not NARRATOR.search(fp[k]):
            errs.append(f"{k} not first person")
    for k in FP_KEYS:  # same content: length within 25% of the original
        a, b = len(fam[k].split()), len(fp[k].split())
        if not 0.75 * a - 2 <= b <= 1.25 * a + 2:
            errs.append(f"{k} length {b} vs {a}")
    return errs


async def _firstperson(fams: list[dict], out: Path, model: str, concurrency: int, max_tries: int = 3):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         max_retries=2, timeout=300)
    done = set()
    if out.exists():
        done = {json.loads(l)["base_family"] for l in open(out)}
    todo = [f for f in fams if f["id"] not in done]
    print(f"[fp] {len(fams)} families, {len(todo)} to rewrite with {model}", flush=True)
    sem, lock, stats = asyncio.Semaphore(concurrency), asyncio.Lock(), {"ok": 0, "fail": 0}

    async def one(f):
        async with sem:
            segs = json.dumps({k: f[k] for k in FP_KEYS}, ensure_ascii=False, indent=1)
            prompt = FP_PROMPT.format(name=f["name"], segments=segs, keys=", ".join(FP_KEYS))
            errs = ["not tried"]
            for attempt in range(max_tries):
                try:
                    r = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=6000,
                        temperature=0.3, extra_body={"reasoning": {"effort": "low"}})
                    fp = _parse(r.choices[0].message.content)
                except Exception as e:
                    fp = None
                errs = check_fp(fp, f) if fp else ["unparseable"]
                if not errs:
                    rec = {**f, **{k: fp[k].strip() for k in FP_KEYS}, "id": f"{f['id']}-1p",
                           "base_family": f["id"], "perspective": "first", "fp_writer": model,
                           "fp_attempt": attempt}
                    async with lock:
                        with open(out, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        stats["ok"] += 1
                        if stats["ok"] % 100 == 0:
                            print(f"[fp] {stats}", flush=True)
                    return
            stats["fail"] += 1
            print(f"[fp] gave up on {f['id']}: {errs}", flush=True)

    await asyncio.gather(*(one(f) for f in todo))
    print(f"[fp] done: {stats}", flush=True)


CONDITIONS = {"forbid_do": ("norm_forbidden", "action"), "allow_do": ("norm_allowed", "action"),
              "unaware_do": ("norm_unaware", "action"), "forbid_decline": ("norm_forbidden", "decline"),
              "allow_decline": ("norm_allowed", "decline")}


def split_of(setting_idx: int) -> str:
    """Held-out settings: every third setting is test (20 of 60)."""
    return "test" if setting_idx % 3 == 2 else "train"


def build(families: list[dict]) -> list[dict]:
    """One text per (family, condition), with character spans of the act segment (action / decline) and of
    the norm segment, so the reader can pool over identical tokens."""
    rows = []
    for f in families:
        for cond, (nk, ak) in CONDITIONS.items():
            parts = [f["setup"], f[nk], f[ak], f["suffix"]]
            text, spans, pos = "", {}, 0
            for name, p in zip(("setup", "norm", "act", "suffix"), parts):
                if text:
                    text += " "
                a = len(text)
                text += p
                spans[name] = [a, len(text)]
            rows.append({"id": f"{f['id']}:{cond}", "family": f["id"], "condition": cond, "kind": f["kind"],
                         "setting": f["setting"], "split": split_of(f["setting_idx"]), "text": text,
                         "spans": spans, "perspective": f.get("perspective", "third"),
                         "base_family": f.get("base_family", f["id"])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n-per-cell", type=int, default=2)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--model", default="openai/gpt-oss-120b")
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--out", default="data/transgression/families.jsonl")
    fpp = sub.add_parser("firstperson", help="rewrite every family into first person (matched twins)")
    fpp.add_argument("--families", default="data/transgression/families.jsonl")
    fpp.add_argument("--out", default="data/transgression/families_1p.jsonl")
    fpp.add_argument("--model", default="openai/gpt-oss-120b")
    fpp.add_argument("--concurrency", type=int, default=24)
    fpp.add_argument("--limit", type=int, default=None)
    b = sub.add_parser("build")
    b.add_argument("--families", nargs="+", default=["data/transgression/families.jsonl",
                                                     "data/transgression/families_1p.jsonl"])
    b.add_argument("--out", default="data/transgression/texts.jsonl")
    a = ap.parse_args()
    if a.cmd == "generate":
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        cs = cells(a.n_per_cell)
        if a.limit:
            cs = random.Random(0).sample(cs, a.limit)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(_generate(cs, out, a.model, a.concurrency))
    elif a.cmd == "firstperson":
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        fams = [json.loads(l) for l in open(a.families)]
        if a.limit:
            fams = fams[: a.limit]
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(_firstperson(fams, out, a.model, a.concurrency))
    else:
        fams = [json.loads(l) for p in a.families if Path(p).exists() for l in open(p)]
        rows = build(fams)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        print(f"[build] {len(fams)} families -> {len(rows)} texts "
              f"({sum(r['split'] == 'test' for r in rows)} test; {dict(Counter(r['perspective'] for r in rows))})")


if __name__ == "__main__":
    main()
