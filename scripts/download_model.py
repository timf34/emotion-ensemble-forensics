#!/usr/bin/env python3
"""Download a model's weights at its pinned revision into HF_HOME (hf_transfer if enabled).

    HF_HUB_ENABLE_HF_TRANSFER=1 python -u scripts/download_model.py --model qwen3.6-27b
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from eforensics.config import get_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    spec = get_model(args.model)
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            path = snapshot_download(spec.hf_id, revision=spec.revision,
                                     allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
            print(f"[download] {spec.hf_id}@{spec.revision} -> {path} in {time.time() - t0:.0f}s", flush=True)
            return
        except Exception as e:  # transient network errors: retry rather than disabling hf_transfer
            print(f"[download] attempt {attempt} failed: {e!r}", flush=True)
            time.sleep(10 * attempt)
    sys.exit(1)


if __name__ == "__main__":
    main()
