#!/usr/bin/env bash
# Pod job for the context-rot probe readout (run with `rp run ... -- bash context_rot/pod_readout.sh`).
# Expects the rollouts at /workspace/cr_runs/run-*/final/messages.json (rp scp'd from the laptop).
# Weights download in the background (hf_transfer) while nothing else is needed; activations go to
# /workspace/cr_acts and are uploaded to $HF_DATASET_REPO under context_rot/<tag>/. Stops the pod at
# the end (or after MAX_HOURS) through the RunPod REST API when SHUTDOWN=stop.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG="${TAG:-gptoss120b_lru_main3}"
MAX_HOURS="${MAX_HOURS:-3}"
PY="${PY:-python3}"
POD_ID="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)}"

stop_pod() {
  [ "${SHUTDOWN:-}" = stop ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "$POD_ID" ] && \
    curl -s -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
}
( sleep "$(awk "BEGIN{print int($MAX_HOURS*3600)}")"; echo "!! MAX_HOURS reached"; stop_pod ) &
CEIL=$!

echo "== $(date -u +%T) download weights"
$PY - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("openai/gpt-oss-120b", allow_patterns=["*.json", "model-*.safetensors", "*.jinja"], ignore_patterns=["original/*", "metal/*"])
print("weights at", p)
PY
echo "== $(date -u +%T) readout"
$PY -u context_rot/run_readout.py --runs /workspace/cr_runs --out /workspace/cr_acts
code=$?
echo "== $(date -u +%T) readout exit $code; uploading"
$PY - <<PY
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.environ["HF_DATASET_REPO"]
api.create_repo(repo, repo_type="dataset", private=os.environ.get("HF_DATASET_PUBLIC") != "1", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path="/workspace/cr_acts",
                  path_in_repo="context_rot/${TAG}", commit_message="context-rot readout ${TAG}")
print("uploaded to", repo)
PY
echo "EXIT=$code"
kill "$CEIL" 2>/dev/null
stop_pod
exit "$code"
