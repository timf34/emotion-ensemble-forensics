"""Experiment config (configs/pilot.yaml), model specs (configs/models.yaml) and manifests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import CONFIG_DIR, REPO_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "pilot.yaml")


@dataclass(frozen=True)
class ModelSpec:
    key: str                     # short name used in output paths, e.g. "qwen3.6-27b"
    hf_id: str
    revision: str | None
    tokenizer_revision: str | None
    n_layers: int | None
    hidden_size: int | None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 128
    trust_remote_code: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def load_models(path: str | Path | None = None) -> dict[str, ModelSpec]:
    raw = load_yaml(path or CONFIG_DIR / "models.yaml")
    out = {}
    for key, m in raw["models"].items():
        m = dict(m)
        known = {k: m.pop(k) for k in list(m) if k in ModelSpec.__dataclass_fields__}
        for k in ("chat_template_kwargs", "vllm_kwargs"):
            known[k] = dict(known.get(k) or {})
        out[key] = ModelSpec(key=key, extra=m, **known)
    return out


def get_model(key: str, path: str | Path | None = None) -> ModelSpec:
    models = load_models(path)
    if key not in models:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(models)}")
    return models[key]


def require_pinned(spec: ModelSpec) -> None:
    if spec.revision is None and os.environ.get("EF_ALLOW_UNPINNED") != "1":
        raise RuntimeError(f"{spec.hf_id} has no pinned revision in configs/models.yaml")


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def manifest_path(exp: dict[str, Any], dataset: str) -> Path:
    return repo_path(exp["manifests"][dataset])


def load_manifest(exp: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    path = manifest_path(exp, dataset)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate task ids in {path}")
    return rows


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()[:16]


def vector_folder(exp: dict[str, Any], model_key: str) -> str:
    return exp["sources"]["emotion_vectors"]["folders"][model_key]
