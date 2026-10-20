"""Loading saved activations and turning them into probe features (CPU).

Arrays are memory-mapped and one (rep, block) slice is read at a time:
n x hidden x 4 B = 2,000 x 5,120 x 4 B = 41 MB per slice for the pilot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import activations_dir, read_jsonl


@dataclass
class ActStore:
    path: Path
    index: list[dict[str, Any]]
    blocks: list[int]
    reps: list[str]

    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.index]

    def array(self, rep: str, block: int, rows: np.ndarray | None = None) -> np.ndarray:
        a = np.load(self.path / f"{rep}.npy", mmap_mode="r")
        j = self.blocks.index(block)
        x = a[:, j, :] if rows is None else a[rows, j, :]
        return np.asarray(x, dtype=np.float32)


def load_store(model_key: str, dataset: str, root: Path | None = None) -> ActStore:
    d = root or activations_dir(model_key, dataset)
    meta = json.loads((d / "meta.json").read_text())
    return ActStore(path=d, index=read_jsonl(d / "index.jsonl"), blocks=list(meta["blocks"]), reps=list(meta["reps"]))


def project(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Scores on unit directions: [n, H] @ [k, H].T -> [n, k]."""
    return X.astype(np.float64) @ U.T


def random_directions(hidden: int, k: int, seed: int) -> np.ndarray:
    R = np.random.default_rng(seed).standard_normal((k, hidden))
    return R / np.linalg.norm(R, axis=1, keepdims=True)


def effective_rank(U: np.ndarray) -> dict[str, float]:
    """Participation ratio and components for 80/90% of the spectrum of unit directions."""
    s = np.linalg.svd(U / np.linalg.norm(U, axis=1, keepdims=True), compute_uv=False) ** 2
    frac = np.cumsum(s) / s.sum()
    return {"participation_ratio": float(s.sum() ** 2 / (s ** 2).sum()),
            "k80": int(np.searchsorted(frac, 0.8) + 1), "k90": int(np.searchsorted(frac, 0.9) + 1)}
