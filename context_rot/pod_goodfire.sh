#!/usr/bin/env bash
# Pod job (2x H200): independent reward-hacking readouts from Goodfire's labelled rollouts, for three
# models, plus the GPT-OSS-20B LRU rollout readout. Results go to $HF_DATASET_REPO under
# context_rot/goodfire_<model>/ and context_rot/gptoss20b_lru_main/; the pod stops itself at the end
# (or after MAX_HOURS) when SHUTDOWN=stop.
#   expects: the repo at its own root with .env, the 20B rollouts at /workspace/cr20_runs/run-*/final/
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python3}"
MAX_HOURS="${MAX_HOURS:-4}"
POD_ID="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)}"
stop_pod() {
  [ "${SHUTDOWN:-}" = stop ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "$POD_ID" ] && \
    curl -s -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
}
( sleep "$(awk "BEGIN{print int($MAX_HOURS*3600)}")"; echo "!! MAX_HOURS reached"; stop_pod ) &
CEIL=$!
upload() {  # upload <local dir> <path in repo>
  $PY - "$1" "$2" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(); repo = os.environ["HF_DATASET_REPO"]
api.create_repo(repo, repo_type="dataset", private=os.environ.get("HF_DATASET_PUBLIC") != "1", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=sys.argv[1], path_in_repo=sys.argv[2],
                  commit_message=f"context-rot {sys.argv[2]}")
print("uploaded", sys.argv[1], "->", sys.argv[2])
PY
}

echo "== $(date -u +%T) data + weights"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Goodfire/reward-hack-data", repo_type="dataset", local_dir="outputs/goodfire",
                  allow_patterns=["impossiblebench-*/traces.jsonl", "impossiblebench-*/labels/*"])
for m in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
    snapshot_download(m, allow_patterns=["*.json", "model-*.safetensors", "*.jinja"], ignore_patterns=["original/*", "metal/*"])
print("data + gpt-oss weights ready")
PY
code=0

echo "== $(date -u +%T) gpt-oss-120b goodfire"
$PY -u context_rot/run_goodfire.py --model openai/gpt-oss-120b --family gptoss --out /workspace/gf/oss120b || code=1
upload /workspace/gf/oss120b context_rot/goodfire_gptoss120b || code=1

echo "== $(date -u +%T) gpt-oss-20b goodfire + LRU readout"
$PY -u context_rot/run_goodfire.py --model openai/gpt-oss-20b --family gptoss --out /workspace/gf/oss20b || code=1
upload /workspace/gf/oss20b context_rot/goodfire_gptoss20b || code=1
if [ -d /workspace/cr20_runs ]; then
  $PY -u context_rot/run_readout.py --model openai/gpt-oss-20b --runs /workspace/cr20_runs --out /workspace/cr20_acts \
    --blocks 6 8 10 12 14 15 16 18 20 22 23 || code=1
  upload /workspace/cr20_acts context_rot/gptoss20b_lru_main || code=1
fi

echo "== $(date -u +%T) qwen3.6-27b goodfire"
$PY -u context_rot/run_goodfire.py --model qwen3.6-27b --family qwen --blocks 24 32 36 40 44 48 56 \
  --out /workspace/gf/qwen36 || code=1
upload /workspace/gf/qwen36 context_rot/goodfire_qwen36 || code=1

echo "EXIT=$code $(date -u +%T)"
kill "$CEIL" 2>/dev/null
stop_pod
exit "$code"
