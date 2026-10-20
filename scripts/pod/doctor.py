#!/usr/bin/env python3
"""Fail-fast pod preflight (main venv), run before any GPU work by scripts/pod/_lib.sh
(adapted from timf34/grader-shifts-persona-space@0ad65c4): GPUs usable, disk, pinned model
revisions reachable, emotion vectors downloadable at the pinned revision, results dataset
writable, judge key live (one tiny call), self-stop configured, and the code sandbox
actually isolates (model-written code cannot read .env or see API keys; network status).

    python scripts/pod/doctor.py --models qwen3.6-27b --need-gb 100 [--judge] [--shutdown stop]

Exit 0 = safe to leave running; otherwise every failed check is listed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--need-gb", type=float, default=100.0)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--shutdown", default=os.environ.get("SHUTDOWN", ""))
    p.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    args = p.parse_args()
    ok, bad, warn = [], [], []

    def check(name, fn):
        try:
            msg = fn()
            (warn if str(msg).startswith("WARNING") else ok).append(f"{name}: {msg}")
        except Exception as e:
            bad.append(f"{name}: {e}")

    def gpus():
        import torch

        n = torch.cuda.device_count()
        assert n > 0, "no CUDA device visible"
        out = []
        for i in range(n):
            x = torch.ones(16, device=f"cuda:{i}")
            assert (x @ x).item() == 16
            out.append(f"{torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GB")
        return f"{n} x " + ", ".join(sorted(set(out))) + f" (torch {torch.__version__}, cuda {torch.version.cuda})"

    def disk():
        hf = Path(os.environ.get("HF_HOME", "/workspace/hf"))
        hf.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(hf).free / 2**30
        assert free >= args.need_gb, f"{free:.0f} GB free under {hf}, need {args.need_gb:.0f}"
        return f"{free:.0f} GB free under {hf}"

    def hf_models():
        from huggingface_hub import HfApi

        from eforensics.config import get_model

        out = []
        for key in args.models:
            spec = get_model(key)
            info = HfApi().model_info(spec.hf_id, revision=spec.revision)
            assert info.sha == spec.revision, f"{spec.hf_id}: hub resolved {info.sha}, pinned {spec.revision}"
            out.append(f"{spec.hf_id}@{spec.revision[:8]}")
        return ", ".join(out)

    def vectors():
        from eforensics.config import load_experiment
        from eforensics.vectors import load_directions

        exp = load_experiment()
        out = []
        for key in args.models:
            for ls in ("emotions", "syndromes"):
                d = load_directions(key, exp, ls)
                need = set(exp["extraction"]["blocks"])
                assert need <= set(d.blocks()), f"{key} {ls}: blocks {sorted(need - set(d.blocks()))} missing"
                out.append(f"{key}:{ls}={len(d.labels)}")
        return ", ".join(out) + f" @ {exp['sources']['emotion_vectors']['revision'][:8]}"

    def manifests():
        from eforensics.config import file_sha, load_experiment, load_manifest

        exp = load_experiment()
        return ", ".join(f"{k}={len(load_manifest(exp, k))} rows sha {file_sha(v)}" for k, v in exp["manifests"].items())

    def hf_dataset():
        repo = os.environ.get("HF_DATASET_REPO")
        if not repo:
            return "WARNING HF_DATASET_REPO unset: results stay on this pod only (no uploads; terminate -> stop)"
        from huggingface_hub import HfApi

        return f"{repo} (token of {HfApi().whoami().get('name')})"

    def judge():
        import asyncio

        from eforensics.config import load_experiment
        from eforensics.judges import JudgeClient

        cfg = load_experiment()["judges"]
        c = JudgeClient()
        reply = asyncio.run(c.chat(cfg["model"], "Reply with the single digit 3.", 16))
        assert reply and reply.strip(), "empty reply"
        return f"{c.provider}:{cfg['model']} answered {reply.strip()[:20]!r}"

    def sandbox():
        from eforensics.sandbox import selfcheck

        env_file = ROOT / ".env"
        r = selfcheck(str(env_file) if env_file.exists() else None)
        assert r.get("ok"), f"sandbox probe failed: {r}"
        assert not r.get("secret_readable"), f"model-written code can read {env_file} ({r['mode']})"
        assert not r.get("env_keys"), f"API keys visible in the sandbox environment: {r['env_keys']}"
        msg = f"mode={r['mode']} uid={r.get('uid')} secret_readable={r.get('secret_readable')} network={r.get('network')}"
        return ("WARNING " + msg + " (network reachable: socket prelude only)") if r.get("network") else msg

    def self_stop():
        if not args.shutdown:
            return "WARNING SHUTDOWN unset: the pod keeps billing after the job (rp down it yourself)"
        assert args.shutdown in ("stop", "terminate"), f"SHUTDOWN={args.shutdown!r}"
        assert os.environ.get("RUNPOD_API_KEY"), "RUNPOD_API_KEY unset (.env): self-stop would be a no-op"
        pid = args.pod_id
        if not pid and os.path.exists("/proc/1/environ"):
            env = Path("/proc/1/environ").read_bytes().split(b"\0")
            pid = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"RUNPOD_POD_ID=")), "")
        assert pid, "RUNPOD_POD_ID not found"
        return f"SHUTDOWN={args.shutdown} via REST API for pod {pid}"

    for name, fn in [("gpus", gpus), ("disk", disk), ("hf models", hf_models), ("emotion vectors", vectors),
                     ("manifests", manifests), ("hf dataset", hf_dataset), ("sandbox", sandbox),
                     ("self-stop", self_stop)] + ([("judge", judge)] if args.judge else []):
        check(name, fn)
    for line in ok:
        print("  ok  ", line)
    for line in warn:
        print("  WARN", line)
    for line in bad:
        print("  FAIL", line)
    if bad:
        print(f"!! {len(bad)} preflight check(s) failed")
        return 1
    print("ALL PREFLIGHT CHECKS PASSED - safe to leave it running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
