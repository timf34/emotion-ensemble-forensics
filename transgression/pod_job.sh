#!/usr/bin/env bash
# Pod job (2x H200): knowing-transgression stories + norm-variant transcripts through GPT-OSS-120B, then
# GPT-OSS-20B. Results go to $HF_DATASET_REPO under context_rot/transgression_gptoss{120b,20b}/; the pod stops
# itself at the end (or after MAX_HOURS) when SHUTDOWN=stop.
#   expects: the repo at its own root with .env, plus transgression/ and data/transgression/ (texts.jsonl,
#   norm_runs/) copied in from the laptop.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python3}"
MAX_HOURS="${MAX_HOURS:-3.5}"
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
                  commit_message=f"transgression {sys.argv[2]}")
print("uploaded", sys.argv[1], "->", sys.argv[2])
PY
}
[ -f data/transgression/texts.jsonl ] && [ -d data/transgression/norm_runs ] || { echo "!! data/transgression missing"; exit 2; }

echo "== $(date -u +%T) weights (20b in the background)"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openai/gpt-oss-120b", allow_patterns=["*.json", "model-*.safetensors", "*.jinja"], ignore_patterns=["original/*", "metal/*"])
print("120b ready")
PY
( $PY -c 'from huggingface_hub import snapshot_download as s; s("openai/gpt-oss-20b", allow_patterns=["*.json","model-*.safetensors","*.jinja"], ignore_patterns=["original/*","metal/*"]); print("20b ready")' ) &
DL20=$!
code=0

echo "== $(date -u +%T) gpt-oss-120b"
$PY -u transgression/run_pod.py --model openai/gpt-oss-120b --out /workspace/tg/oss120b || code=1
cp data/transgression/texts.jsonl data/transgression/norm_runs/meta.json /workspace/tg/oss120b/ 2>/dev/null
upload /workspace/tg/oss120b context_rot/transgression_gptoss120b || code=1

echo "== $(date -u +%T) gpt-oss-20b"
wait "$DL20"
$PY -u transgression/run_pod.py --model openai/gpt-oss-20b --out /workspace/tg/oss20b --batch 32 || code=1
upload /workspace/tg/oss20b context_rot/transgression_gptoss20b || code=1

echo "EXIT=$code $(date -u +%T)"
kill "$CEIL" 2>/dev/null
stop_pod
exit "$code"
