"""Stage 4a: residual-stream summaries from a teacher-forced HF forward pass over the EXACT
token ids stored by stage 2 (batch size 1: no padding ever reaches the Gated-DeltaNet state).

Per record, at each block of the sweep (block-OUTPUT convention, hooks.py):
  last       residual at the last prompt position (the prediction point: the "\\n\\n" after
             the empty <think></think>, before any answer token)
  user_mean  mean over the content tokens of the last user message (pre-registered alternative)
  resp       mean over the generated answer tokens (a post-hoc readout, never a prediction)

Written in shards of `shard_size` records (resumable), then merged into
activations/{last,user_mean,resp}.npy [n, n_blocks, hidden] (float16) + index.jsonl + meta.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelSpec
from .io import activations_dir, environment_info, read_jsonl, responses_path, write_json, write_jsonl
from .templates import inspect_template, last_user_span, split_generation

REPS = ("last", "user_mean", "resp")
INDEX_FIELDS = ("id", "run_id", "task_id", "seed_idx", "turn", "dataset", "variant", "template_id", "task_family",
                "exploit_family", "emotion_metric", "duplicate_group", "problem_id", "ib_split", "impossible",
                "config_id", "truncated", "n_prompt_tokens", "n_gen_tokens")


def text_config(config):
    return getattr(config, "text_config", None) or config


def load_hf_model(spec: ModelSpec, dtype: str = "bfloat16", device_map: str | dict = "auto"):
    """Load the decoder with all weights present; refuse a partially initialised model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = dict(revision=spec.revision, dtype=getattr(torch, dtype), device_map=device_map,
                  trust_remote_code=spec.trust_remote_code)
    model, info = AutoModelForCausalLM.from_pretrained(spec.hf_id, output_loading_info=True, **common)
    missing = [k for k in sorted(info.get("missing_keys", [])) if "layers." in k or "embed_tokens" in k]
    if missing:
        print(f"[extract] AutoModelForCausalLM left {len(missing)} decoder weights uninitialised; "
              "falling back to AutoModelForImageTextToText", flush=True)
        from transformers import AutoModelForImageTextToText

        del model
        model, info = AutoModelForImageTextToText.from_pretrained(spec.hf_id, output_loading_info=True, **common)
        missing = [k for k in sorted(info.get("missing_keys", [])) if "layers." in k or "embed_tokens" in k]
        if missing:
            raise RuntimeError(f"decoder weights missing after fallback: {missing[:5]}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.tokenizer_revision or spec.revision,
                                        trust_remote_code=spec.trust_remote_code)
    tc = text_config(model.config)
    if spec.n_layers is not None and tc.num_hidden_layers != spec.n_layers:
        raise RuntimeError(f"{spec.hf_id}: {tc.num_hidden_layers} layers, configs/models.yaml says {spec.n_layers}")
    if spec.hidden_size is not None and tc.hidden_size != spec.hidden_size:
        raise RuntimeError(f"{spec.hf_id}: hidden {tc.hidden_size}, configs/models.yaml says {spec.hidden_size}")
    return model, tok


class Extractor:
    def __init__(self, model, tokenizer, blocks: list[int], enable_thinking: bool = False,
                 template_kwargs: dict | None = None):
        from .hooks import find_text_model

        self.model = model
        self.text_model = find_text_model(model)       # text decoder only: no LM head, no vision tower
        self.tok = tokenizer
        self.blocks = sorted(set(int(b) for b in blocks))
        n = len(self.text_model.layers)
        bad = [b for b in self.blocks if not 0 <= b < n]
        if bad:
            raise ValueError(f"blocks {bad} out of range for a {n}-block model")
        self.info = inspect_template(tokenizer, enable_thinking, template_kwargs)

    def forward(self, ids: list[int]) -> dict[int, Any]:
        import torch

        from .hooks import capture_block_outputs

        dev = self.model.get_input_embeddings().weight.device
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.inference_mode(), capture_block_outputs(self.model, self.blocks) as store:
            self.text_model(input_ids=x, use_cache=False)
        return {b: store[b][0] for b in self.blocks}

    def record(self, rec: dict[str, Any]) -> dict[str, Any]:
        p_ids, g_ids = rec["prompt_ids"], rec.get("gen_ids") or []
        P = len(p_ids)
        split = split_generation(g_ids, self.tok, self.info)
        u0, u1 = last_user_span(p_ids, self.tok)
        ids = list(p_ids) + list(g_ids[: (split.answer[-1] + 1) if split.answer else 0])
        hs = self.forward(ids)
        H = next(iter(hs.values())).shape[-1]
        out = {rep: np.zeros((len(self.blocks), H), dtype=np.float32) for rep in REPS}
        for j, b in enumerate(self.blocks):
            h = hs[b]
            out["last"][j] = h[P - 1].float().cpu().numpy()
            out["user_mean"][j] = h[u0:u1].float().mean(0).cpu().numpy()
            if split.answer:
                idx = [P + i for i in split.answer]
                out["resp"][j] = h[idx].float().mean(0).cpu().numpy()
        return {"acts": out, "user_span": [u0, u1], "n_answer_tokens": len(split.answer),
                "has_answer": bool(split.answer), "n_input_tokens": len(ids)}


def _shard_dir(model_key: str, dataset: str) -> Path:
    return activations_dir(model_key, dataset) / "shards"


def extract_dataset(ex: Extractor, model_key: str, dataset: str, shard_size: int = 128, worker: int = 0,
                    num_workers: int = 1, save_dtype: str = "float16", only_ids: set[str] | None = None) -> int:
    """Extract this worker's shards (shard k belongs to worker k % num_workers)."""
    recs = read_jsonl(responses_path(model_key, dataset))
    if only_ids is not None:
        recs = [r for r in recs if r["id"] in only_ids]
    recs.sort(key=lambda r: r["id"])
    sd = _shard_dir(model_key, dataset)
    sd.mkdir(parents=True, exist_ok=True)
    n_done, t0 = 0, time.time()
    for k in range(0, (len(recs) + shard_size - 1) // shard_size):
        if k % num_workers != worker:
            continue
        chunk = recs[k * shard_size:(k + 1) * shard_size]
        f = sd / f"{k:05d}.npz"
        ids = [r["id"] for r in chunk]
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                if list(z["ids"]) == ids:
                    continue
        arrays = {rep: [] for rep in REPS}
        index = []
        for r in chunk:
            res = ex.record(r)
            for rep in REPS:
                arrays[rep].append(res["acts"][rep])
            index.append({**{k2: r.get(k2) for k2 in INDEX_FIELDS}, "user_span": res["user_span"],
                          "n_answer_tokens": res["n_answer_tokens"], "has_answer": res["has_answer"]})
        np.savez(f.with_suffix(".tmp.npz"), ids=np.array(ids), index=np.array(json.dumps(index)),
                 **{rep: np.stack(v).astype(save_dtype) for rep, v in arrays.items()})
        f.with_suffix(".tmp.npz").replace(f)
        n_done += len(chunk)
        print(f"[extract] {dataset} shard {k}: {len(chunk)} records ({(time.time() - t0) / max(n_done, 1):.2f}s/record)",
              flush=True)
    return n_done


def merge_shards(model_key: str, dataset: str, blocks: list[int], meta: dict[str, Any] | None = None) -> int:
    """Concatenate all shards (in id order) into the final arrays. Refuses if any record is missing."""
    recs = read_jsonl(responses_path(model_key, dataset))
    want = sorted(r["id"] for r in recs)
    sd = _shard_dir(model_key, dataset)
    files = sorted(f for f in sd.glob("*.npz") if ".tmp" not in f.name)
    arrays = {rep: [] for rep in REPS}
    index: list[dict] = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            for rep in REPS:
                arrays[rep].append(z[rep])
            index += json.loads(str(z["index"]))
    got = [r["id"] for r in index]
    if got != want:
        missing = sorted(set(want) - set(got))
        raise RuntimeError(f"{dataset}: shards cover {len(got)} of {len(want)} records (missing e.g. {missing[:3]})")
    d = activations_dir(model_key, dataset)
    for rep in REPS:
        np.save(d / f"{rep}.npy", np.concatenate(arrays[rep]))
    write_json(d / "meta.json", {"blocks": blocks, "reps": list(REPS),
                                 "block_convention": "block b = residual stream at the OUTPUT of decoder block b "
                                                     "(0-based) = HF hidden_states[b + 1] (emotion-vector convention)",
                                 "environment": environment_info(), **(meta or {})})
    for i, r in enumerate(index):
        r["row"] = i
    write_jsonl(d / "index.jsonl", index)
    return len(index)
