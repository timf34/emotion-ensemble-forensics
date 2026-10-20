"""eforensics: emotion-probe ensembles for predicting model misbehaviour without chain-of-thought."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "configs"

__all__ = ["REPO_ROOT", "DATA_DIR", "CONFIG_DIR"]
