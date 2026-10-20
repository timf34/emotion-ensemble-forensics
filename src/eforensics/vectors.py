"""Pinned emotion / syndrome vectors (timf34/emotion-vectors, denoised `_dn` files).

    vecs = load_directions("qwen3.6-27b", exp)          # Directions for emotions
    U = vecs.matrix(40)                                  # [42, hidden], unit-normalised rows

Vectors are {label: {block: Tensor[hidden]}} in the block-OUTPUT convention (hooks.py).
Scores are dot products with UNIT-normalised directions (scale-free across labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EMOTIONS_42 = (
    "depressed", "sad", "gloomy", "melancholy", "miserable", "dispirited", "listless", "weary", "worn out", "tired",
    "resigned", "lonely", "heartbroken", "grief-stricken", "worthless", "self-critical", "ashamed", "guilty",
    "humiliated", "embarrassed", "regretful", "sorry", "desperate", "panicked", "hysterical", "overwhelmed",
    "distressed", "tormented", "trapped", "stuck", "frustrated", "exasperated", "angry", "anxious", "afraid",
    "nervous", "stressed", "calm", "self-confident", "content", "happy", "hopeful")
SYNDROMES_6 = ("clinical_depression", "acute_grief", "burnout_exhaustion", "physical_illness_fatigue",
               "anxiety_panic", "frustration_blocked_goal")


@dataclass
class Directions:
    labels: list[str]
    vectors: dict[str, dict[int, np.ndarray]]      # label -> block -> float32 [hidden]
    source: dict[str, Any]

    def blocks(self) -> list[int]:
        return sorted(set.intersection(*(set(v) for v in self.vectors.values())))

    def matrix(self, block: int, normalise: bool = True) -> np.ndarray:
        M = np.stack([self.vectors[l][block] for l in self.labels]).astype(np.float64)
        if normalise:
            M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return M


def _from_torch(obj) -> dict[str, dict[int, np.ndarray]]:
    out = {}
    for label, by_block in obj.items():
        out[label] = {int(b): v.float().cpu().numpy() for b, v in by_block.items()}
    return out


def load_directions(model_key: str, exp: dict[str, Any], label_set: str = "emotions",
                    local_path: str | Path | None = None) -> Directions:
    """label_set: "emotions" (42) or "syndromes" (6). Downloads the pinned HF revision unless
    `local_path` (a vectors_*_dn.pt file) is given."""
    import torch

    src = exp["sources"]["emotion_vectors"]
    folder = src["folders"][model_key]
    fname = f"{folder}/vectors_{label_set}_dn.pt"
    if local_path is None:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(src["hf_dataset"], fname, repo_type="dataset", revision=src["revision"])
    raw = torch.load(local_path, map_location="cpu", weights_only=True)
    vecs = _from_torch(raw)
    expected = EMOTIONS_42 if label_set == "emotions" else SYNDROMES_6
    missing = [l for l in expected if l not in vecs]
    if missing:
        raise RuntimeError(f"{fname}: missing labels {missing}")
    return Directions(labels=list(expected), vectors={l: vecs[l] for l in expected},
                      source={"hf_dataset": src["hf_dataset"], "revision": src["revision"], "file": fname})
