"""Shared test helpers: a tiny Qwen3.5-architecture model, cached real tokenizers, a fake vLLM."""

from __future__ import annotations

import sys
import types

import pytest

QWEN = {"qwen3.6-27b": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", {}),
        "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", {"reasoning_effort": "medium"})}


def tokenizer(key: str = "qwen3.6-27b"):
    transformers = pytest.importorskip("transformers")
    hf, rev, _ = QWEN[key]
    try:
        return transformers.AutoTokenizer.from_pretrained(hf, revision=rev, local_files_only=True)
    except Exception:
        pytest.skip(f"{hf} tokenizer not cached locally")


def tiny_model(vocab: int = 248320, n_layers: int = 8):
    torch = pytest.importorskip("torch")
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    tc = dict(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
              num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
              full_attention_interval=4)
    cfg = Qwen3_5Config(text_config=tc, vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                                                           num_heads=2, out_hidden_size=32))
    return Qwen3_5ForConditionalGeneration(cfg).eval()


class _Out:
    def __init__(self, token_ids, text, finish_reason="stop"):
        self.token_ids, self.text, self.finish_reason, self.logprobs = token_ids, text, finish_reason, None


class _Res:
    def __init__(self, out):
        self.outputs = [out]


def install_fake_vllm(monkeypatch):
    """Minimal stand-in for the vllm API used by eforensics.generate."""
    vllm = types.ModuleType("vllm")
    inputs = types.ModuleType("vllm.inputs")

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def TokensPrompt(prompt_token_ids):
        return {"prompt_token_ids": prompt_token_ids}

    vllm.SamplingParams = SamplingParams
    inputs.TokensPrompt = TokensPrompt
    vllm.inputs = inputs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


class FakeLLM:
    """Answers with `respond(prompt_text, seed) -> text`; tokenises with the real tokenizer."""

    def __init__(self, tok, respond):
        self.tok, self.respond, self.calls = tok, respond, []

    def generate(self, prompts, params, use_tqdm=False):
        out = []
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        for p, sp in zip(prompts, params):
            text = self.respond(self.tok.decode(p["prompt_token_ids"]), sp.seed)
            ids = self.tok(text, add_special_tokens=False)["input_ids"] + [im_end]
            self.calls.append(p["prompt_token_ids"])
            out.append(_Res(_Out(ids, text)))
        return out
