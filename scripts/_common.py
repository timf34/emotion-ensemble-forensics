"""Shared script setup: src/ on the path, .env loaded (EF_ENV_FILE when the pod moved it)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("EF_ENV_FILE") or ROOT / ".env")
except Exception:
    pass
