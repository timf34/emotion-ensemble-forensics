"""Small IO helpers: jsonl, run metadata, output paths.

Layout (one directory per model x dataset):
  outputs/<model>/<dataset>/responses.jsonl          one record per generation turn
  outputs/<model>/<dataset>/runs.jsonl               IB only: one record per multi-turn run
  outputs/<model>/<dataset>/run_config/<cid>.json    identity, rendered template, environment
  outputs/<model>/<dataset>/judge/<name>.jsonl       raw judge outputs (SRH)
  outputs/<model>/<dataset>/labels.jsonl             behaviour labels (03_score_behaviour.py)
  outputs/<model>/<dataset>/activations/             {last,user_mean,resp}.npy + index.jsonl + meta.json
  outputs/<model>/<dataset>/analysis/                features, predictor results, report
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import REPO_ROOT


def outputs_root() -> Path:
    return Path(os.environ.get("EF_OUTPUTS", REPO_ROOT / "outputs"))


def dataset_dir(model_key: str, dataset: str) -> Path:
    return outputs_root() / model_key / dataset


def responses_path(model_key: str, dataset: str) -> Path:
    return dataset_dir(model_key, dataset) / "responses.jsonl"


def runs_path(model_key: str, dataset: str) -> Path:
    return dataset_dir(model_key, dataset) / "runs.jsonl"


def labels_path(model_key: str, dataset: str) -> Path:
    return dataset_dir(model_key, dataset) / "labels.jsonl"


def activations_dir(model_key: str, dataset: str) -> Path:
    return dataset_dir(model_key, dataset) / "activations"


def analysis_dir(model_key: str, dataset: str) -> Path:
    return dataset_dir(model_key, dataset) / "analysis"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "-C", str(REPO_ROOT), "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "repo_commit": git_commit(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
    }
    for mod in ("torch", "transformers", "vllm", "numpy", "sklearn", "fla"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:
            info[f"{mod}_version"] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info
