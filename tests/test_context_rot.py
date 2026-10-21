"""context_rot readout: harmony sequence building on a real rollout, and chunked cached capture ==
a plain full forward (tiny random GPT-OSS model, sliding-window + full layers, attention sinks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("openai/gpt-oss-120b", local_files_only=True)
    except Exception:
        pytest.skip("gpt-oss tokenizer not cached")


def a_run():
    runs = sorted(ROOT.glob("outputs/context_rot/*/optimize_cache_c/*/*/run-*/final/messages.json"))
    runs = [r for r in runs if len(json.loads(r.read_text())) >= 6]
    if not runs:
        pytest.skip("no rollout on disk")
    return json.loads(runs[0].read_text())


def tiny(vocab: int):
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.manual_seed(0)
    cfg = GptOssConfig(vocab_size=vocab, hidden_size=32, intermediate_size=32, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=4,
                       num_experts_per_tok=2, sliding_window=16, max_position_embeddings=1 << 17,
                       layer_types=["sliding_attention", "full_attention"] * 2)
    cfg._attn_implementation = "eager"
    return GptOssForCausalLM(cfg).eval()


def test_sequence_on_real_rollout():
    from context_rot.readout import build_sequence

    t = tok()
    raw = a_run()
    seq = build_sequence(t, raw)
    ids, steps = seq["ids"], seq["steps"]
    assert len(steps) == sum(m["role"] == "assistant" for m in raw)
    for s in steps:
        assert t.decode(ids[s["pre"] - 1: s["pre"] + 1]) == "<|start|>assistant"
    with_r = [s for s in steps if s["reason"]]
    assert with_r, "no reasoning spans found"
    s = with_r[0]
    first = next(m for m in raw if m["role"] == "assistant")
    from context_rot.readout import strip_think

    assert t.decode(ids[s["reason"][0]:s["reason"][1]]).strip()[:80] == strip_think(first["reasoning"])[:80]


def test_chunked_read_matches_full_forward():
    from context_rot.readout import Reader, build_sequence

    t = tok()
    seq = build_sequence(t, a_run())
    ids = seq["ids"][:900]
    steps = [s for s in seq["steps"] if s["pre"] < 900]
    for s in steps:
        if s["reason"]:
            s["reason"] = [s["reason"][0], min(s["reason"][1], 900)]
    m = tiny(len(t) + 64)
    r = Reader(m, [0, 1, 2], max_chunk=97)          # odd chunk: crosses spans and the sliding window
    r.chunk_size = lambda T: 97
    out = r.read(ids, steps)
    with torch.no_grad():
        hs = m.model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states
    for k, s in enumerate(steps):
        for j, b in enumerate([0, 1, 2]):
            np.testing.assert_allclose(out["pre"][k, j], hs[b + 1][0, s["pre"]].numpy(), atol=2e-4, rtol=1e-3)
            if s["reason"]:
                a, e = s["reason"]
                np.testing.assert_allclose(out["reason"][k, j], hs[b + 1][0, a:e].mean(0).numpy(), atol=2e-4,
                                           rtol=1e-3)


def _gf_trace(max_chars=None):
    from context_rot.goodfire import load_source

    d = ROOT / "outputs" / "goodfire" / "impossiblebench-kimi-k3"
    if not d.exists():
        pytest.skip("Goodfire data not downloaded")
    traces, P = load_source(d)
    return traces, P


@pytest.mark.parametrize("family", ["gptoss", "qwen"])
def test_goodfire_steps_and_readout(family):
    """Real Goodfire rollout -> target-template sequence with step spans; the reader (chunked for gpt-oss,
    single pass for qwen) equals a plain full forward at every pre position and step span."""
    from context_rot.goodfire import build
    from context_rot.readout import Reader
    from helpers import tiny_model as tiny_qwen
    from helpers import tokenizer as qwen_tok

    traces, P = _gf_trace()
    t = tok() if family == "gptoss" else qwen_tok("qwen3.6-27b")
    seq = next(s for s in (build(t, tr, P.get(tr["id"], []), family) for tr in traces[:40]) if s is not None)
    labels = {s["label"] for s in seq["steps"]}
    assert labels & {"hack", "honest"}
    for s in seq["steps"]:
        a, b = s["reason"]
        assert 0 < a < b <= len(seq["ids"]) and s["pre"] == a - 1
    ids = seq["ids"][:1200]
    steps = [dict(s, reason=[s["reason"][0], min(s["reason"][1], 1200)]) for s in seq["steps"] if s["reason"][0] < 1199]
    m = tiny(len(t) + 64) if family == "gptoss" else tiny_qwen(vocab=len(t) + 64)
    r = Reader(m, [0, 1, 2], max_chunk=97, single_pass=(family == "qwen"))
    if family == "gptoss":
        r.chunk_size = lambda T: 97
    out = r.read(ids, steps)
    from eforensics.hooks import find_text_model

    with torch.no_grad():
        hs = find_text_model(m)(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states
    for k, s in enumerate(steps):
        a, b = s["reason"]
        for j, blk in enumerate([0, 1, 2]):
            np.testing.assert_allclose(out["pre"][k, j], hs[blk + 1][0, s["pre"]].numpy(), atol=2e-4, rtol=1e-3)
            np.testing.assert_allclose(out["reason"][k, j], hs[blk + 1][0, a:b].mean(0).numpy(), atol=2e-4, rtol=1e-3)
