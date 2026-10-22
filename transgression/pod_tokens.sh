#!/usr/bin/env bash
# Pod job (2x H200): token-level readout of the LRU allow/forbid replays through GPT-OSS-120B
# (transgression/token_readout.py). Uploads to $HF_DATASET_REPO context_rot/transgression_gptoss120b_tokens/ and
# stops the pod at the end (or after MAX_HOURS) when SHUTDOWN=stop.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python3}"
MAX_HOURS="${MAX_HOURS:-1.5}"
POD_ID="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)}"
stop_pod() {
  [ "${SHUTDOWN:-}" = stop ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "$POD_ID" ] && \
    curl -s -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
}
( sleep "$(awk "BEGIN{print int($MAX_HOURS*3600)}")"; echo "!! MAX_HOURS reached"; stop_pod ) &
CEIL=$!
[ -f data/transgression/token_dirs.npz ] && [ -d data/transgression/norm_runs ] || { echo "!! inputs missing"; exit 2; }

echo "== $(date -u +%T) weights"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openai/gpt-oss-120b", allow_patterns=["*.json", "model-*.safetensors", "*.jinja"], ignore_patterns=["original/*", "metal/*"])
print("120b ready")
PY
code=0
echo "== $(date -u +%T) token readout"
$PY -u transgression/token_readout.py --model openai/gpt-oss-120b --out /workspace/tg/tok120b || code=1
cp data/transgression/token_dirs.npz /workspace/tg/tok120b/ 2>/dev/null
$PY - <<'PY' || code=1
import os
from huggingface_hub import HfApi
api = HfApi(); repo = os.environ["HF_DATASET_REPO"]
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path="/workspace/tg/tok120b",
                  path_in_repo="context_rot/transgression_gptoss120b_tokens", commit_message="transgression token readout 120b")
print("uploaded tokens")
PY
echo "EXIT=$code $(date -u +%T)"
kill "$CEIL" 2>/dev/null
stop_pod
exit "$code"
