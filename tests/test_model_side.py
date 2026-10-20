"""Hooks, templates and extraction on CPU: a tiny random Qwen3.5-architecture model (same classes
as Qwen3.6/3.8-27B) and the real Qwen tokenizers when cached."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import QWEN, tiny_model, tokenizer

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_pinned_transformers_matches_vllm_venv():
    from eforensics import REPO_ROOT

    pins = [l.split("==")[1].strip() for l in (REPO_ROOT / "requirements" / "pod.txt").read_text().splitlines()
            if l.startswith("transformers==")]
    assert len(pins) == 1
    sh = (REPO_ROOT / "scripts" / "pod" / "install_vllm.sh").read_text()
    assert f'TRANSFORMERS_PIN="${{TRANSFORMERS_PIN:-{pins[0]}}}"' in sh


def test_block_output_hook_is_hidden_states_plus_one():
    from eforensics.hooks import capture_block_outputs, find_text_model

    m = tiny_model(vocab=512)
    tm = find_text_model(m)
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        hs = tm(input_ids=ids, output_hidden_states=True).hidden_states
        with capture_block_outputs(m, [0, 3, 6]) as st:
            tm(input_ids=ids)
    for b in (0, 3, 6):          # emotion-vector convention: block b output == hidden_states[b + 1]
        assert torch.allclose(st[b], hs[b + 1], atol=1e-6)
        assert not torch.allclose(st[b], hs[b], atol=1e-3)


@pytest.mark.parametrize("key", list(QWEN))
def test_thinking_off_template_and_user_span(key):
    from eforensics.templates import inspect_template, last_user_span, prompt_token_ids, render_prompt

    tok = tokenizer(key)
    kw = QWEN[key][2]
    msgs = [{"role": "user", "content": "First question?"}, {"role": "assistant", "content": "An answer."},
            {"role": "user", "content": "Your previous attempt failed.\n\nTry again."}]
    text = render_prompt(tok, msgs, False, kw)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    info = inspect_template(tok, False, kw)
    assert not info.notes and info.think_close_id == 248069
    ids = prompt_token_ids(tok, msgs, False, kw)
    a, b = last_user_span(ids, tok)
    assert tok.decode(ids[a:b]) == msgs[-1]["content"]
    a1, b1 = last_user_span(prompt_token_ids(tok, msgs[:1], False, kw), tok)
    assert tok.decode(prompt_token_ids(tok, msgs[:1], False, kw)[a1:b1]) == "First question?"


def test_qwen36_drops_empty_think_from_history_qwen38_keeps_it():
    """Why IB turns are re-rendered per turn instead of extended token-wise (generate.py)."""
    from eforensics.templates import render_prompt

    msgs = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
    t36 = render_prompt(tokenizer("qwen3.6-27b"), msgs, False, {})
    t38 = render_prompt(tokenizer("qwen3.8-27b"), msgs, False, {"reasoning_effort": "medium"})
    assert "assistant\nA1<|im_end|>" in t36
    assert "assistant\n<think>\n\n</think>\n\nA1<|im_end|>" in t38


def test_extractor_matches_hidden_states(tmp_path, monkeypatch):
    """Fake records -> Extractor / shards / merge; saved reps equal HF hidden states."""
    monkeypatch.setenv("EF_OUTPUTS", str(tmp_path))
    from eforensics.extract import Extractor, extract_dataset, merge_shards
    from eforensics.features import load_store
    from eforensics.hooks import find_text_model
    from eforensics.io import responses_path, write_jsonl
    from eforensics.templates import last_user_span, prompt_token_ids

    tok = tokenizer("qwen3.6-27b")
    m = tiny_model()
    blocks = [1, 4, 6]
    ex = Extractor(m, tok, blocks, enable_thinking=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    recs = []
    for i in range(5):
        msgs = [{"role": "user", "content": f"Write note number {i} for my friend."}]
        p = prompt_token_ids(tok, msgs, False, {})
        g = tok(f"Dear friend, note {i}.", add_special_tokens=False)["input_ids"] + [im_end]
        recs.append({"id": f"r{i}", "run_id": f"r{i}", "task_id": f"t{i}", "prompt_ids": p, "gen_ids": g,
                     "messages": msgs, "dataset": "srh", "variant": "scored", "exploit_family": "B"})
    write_jsonl(responses_path("tiny", "srh_pilot"), recs)
    extract_dataset(ex, "tiny", "srh_pilot", shard_size=2, save_dtype="float32")
    merge_shards("tiny", "srh_pilot", blocks)
    st = load_store("tiny", "srh_pilot")
    assert st.ids == [f"r{i}" for i in range(5)]
    tm = find_text_model(m)
    for i, r in enumerate(recs):
        P = len(r["prompt_ids"])
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"]])
        with torch.no_grad():
            hs = tm(input_ids=ids, output_hidden_states=True).hidden_states
        u0, u1 = last_user_span(r["prompt_ids"], tok)
        n_ans = len(r["gen_ids"]) - 1                  # trailing <|im_end|> is not an answer token
        for b in blocks:
            h = hs[b + 1][0].numpy()
            np.testing.assert_allclose(st.array("last", b)[i], h[P - 1], atol=1e-5)
            np.testing.assert_allclose(st.array("user_mean", b)[i], h[u0:u1].mean(0), atol=1e-5)
            np.testing.assert_allclose(st.array("resp", b)[i], h[P:P + n_ans].mean(0), atol=1e-5)
