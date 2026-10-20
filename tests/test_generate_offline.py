"""The generation loops end to end with a fake vLLM engine and the real tokenizer."""

from __future__ import annotations

import pytest

from helpers import FakeLLM, install_fake_vllm, tokenizer


@pytest.fixture
def gen(tmp_path, monkeypatch):
    monkeypatch.setenv("EF_OUTPUTS", str(tmp_path))
    install_fake_vllm(monkeypatch)
    from eforensics.config import get_model, load_experiment
    from eforensics.generate import Generator

    exp = load_experiment()
    exp["ib_pilot"]["test_timeout_s"] = 10
    spec = get_model("qwen3.6-27b")
    tok = tokenizer("qwen3.6-27b")
    return exp, spec, tok, Generator


def test_single_turn_resume_and_records(gen):
    exp, spec, tok, Generator = gen
    from eforensics.generate import run_single_turn
    from eforensics.io import read_jsonl, responses_path

    llm = FakeLLM(tok, lambda prompt, seed: "Thank you, thank you, thank you!")
    g = Generator(spec, exp, llm=llm, tokenizer=tok)
    n = run_single_turn(g, "srh_pilot", seeds_per_prompt=3, limit=4)
    assert n == 12
    recs = read_jsonl(responses_path(spec.key, "srh_pilot"))
    assert len(recs) == 12 and len({r["seed"] for r in recs}) == 12
    r = recs[0]
    assert r["prompt_text"].endswith("<think>\n\n</think>\n\n") and r["answer_text"] == "Thank you, thank you, thank you!"
    assert r["exploit_family"] and r["config_id"] and r["has_answer"]
    assert run_single_turn(g, "srh_pilot", seeds_per_prompt=3, limit=4) == 0     # resumed: nothing to do


def test_impossible_loop(gen):
    exp, spec, tok, Generator = gen
    from eforensics.generate import run_impossible
    from eforensics.io import read_jsonl, responses_path, runs_path
    from eforensics.labels import ib_labels

    def respond(prompt, seed):
        # attempt 1: wrong; attempt 2: redefine check (cheat); count turns from the transcript
        ep = prompt.split("# Use check(")[1].split(")")[0]
        if prompt.count("<|im_start|>user") == 1:
            return f"```python\ndef {ep}(*a, **k):\n    return None\n```"
        return f"```python\ndef {ep}(*a, **k):\n    return None\n\ndef check(candidate):\n    pass\n```"

    llm = FakeLLM(tok, respond)
    g = Generator(spec, exp, llm=llm, tokenizer=tok)
    n = run_impossible(g, "ib_pilot", limit=3)          # one row per split
    assert n == 9
    runs = read_jsonl(runs_path(spec.key, "ib_pilot"))
    turns = read_jsonl(responses_path(spec.key, "ib_pilot"))
    assert len(runs) == 9 and len(turns) == 18
    for r in runs:
        assert r["first_pass_attempt"] == 2 and r["status"] == "passed"
        assert r["cheated"] == (r["ib_split"] != "original")
        assert r["attempts"][1]["heuristics"]["redefines_check"]
    t2 = next(t for t in turns if t["turn"] == 2)
    # turn 2 context = first prompt + assistant answer + ImpossibleBench feedback, re-rendered
    assert "Your previous attempt failed the tests. Here's the error:" in t2["prompt_text"]
    assert "To reiterate, this is your task:" in t2["prompt_text"]
    assert len(t2["messages"]) == 3 and t2["messages"][1]["role"] == "assistant"
    labs = ib_labels(turns, runs)
    imp = [l for l in labs if l["impossible"]]
    assert {(l["turn"], l["cheat"], l["will_cheat"]) for l in imp} == {(1, 0, 1), (2, 1, 1)}
    assert all(l["cheat"] is None for l in labs if not l["impossible"])
    assert run_impossible(g, "ib_pilot", limit=3) == 0
