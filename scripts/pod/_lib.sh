# shellcheck shell=bash
# Shared plumbing for scripts/pod/run_*.sh (copied from timf34/grader-shifts-persona-space@0ad65c4,
# EasySteer server parts removed). Source it, then call pod_init:
#
#   source "$(dirname "$0")/_lib.sh"
#   pod_init pilot "$MODEL"       # job name = the `rp run --job` name (its log is /workspace/<job>.log)
#
# Every pod job gets (runpod-runner CLAUDE.md + the sibling repos' issue logs):
#   * repo root as cwd, .env exported, HF_HOME set explicitly (rp run presets its own);
#   * a status file $LOGDIR/STATE.md (one timestamped line per step) and a final DONE or
#     FAILED marker; step logs + the rp job log are uploaded next to the results;
#   * an EXIT trap that always runs: kills background children, stops a vLLM server this
#     job started, writes the marker, uploads the logs, then stops billing (SHUTDOWN=stop|
#     terminate through the RunPod REST API -- `runpodctl` and $RUNPOD_POD_ID are not
#     available inside rp's tmux session, so the pod id is read from PID 1's environment).
#     terminate is downgraded to stop on a failure or a failed upload, so nothing is lost;
#   * a hard ceiling (MAX_HOURS) that stops the pod even if a step hangs;
#   * uploads that record (not hide) failures, a fail-fast preflight (doctor.py), per-GPU
#     fan-out, and the vLLM venv env for vLLM processes (venv bin on PATH for ninja,
#     CUDA compat libraries only there).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
set -a; [ -f .env ] && . ./.env; set +a
# Storage: without a network volume /workspace can be a FUSE or NFS network mount (pip took ~70 min
# for a 12 GB venv there: emotion-concepts ISSUES_LOG #10). Then the weights cache and the
# vLLM venv go on the container disk (fast; wiped on stop, so a restarted pod
# re-downloads); results, logs and the rp venv stay on /workspace. EF_HF_HOME /
# ES_VENV_ROOT override. rp run presets HF_HOME, so it is SET here, never defaulted.
WS_FSTYPE=$(findmnt -n -o FSTYPE /workspace 2>/dev/null || stat -f -c %T /workspace 2>/dev/null || echo "?")
case "$WS_FSTYPE" in   # local block filesystems keep everything on /workspace; anything else
  xfs|ext4|ext3|btrfs|overlay|zfs) _HF_DEFAULT=/workspace/hf; export ES_VENV_ROOT="${ES_VENV_ROOT:-/workspace}" ;;
  *) _HF_DEFAULT=/hf_cache; export ES_VENV_ROOT="${ES_VENV_ROOT:-/}" ;;   # fuse, nfs4, ceph, ...: container disk
esac
export HF_HOME="${EF_HF_HOME:-$_HF_DEFAULT}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python}"
T0=$(date +%s)
UPLOAD_FAILED=0

elapsed() { echo "$(( ($(date +%s) - T0) / 60 )) min"; }
state() { echo "$(date -u +%FT%TZ) [$(elapsed)] $*" | tee -a "$LOGDIR/STATE.md"; }
step() { echo; state "=== $*"; }

pod_init() {
  # job name = the `rp run --job` name: rp runs the job in a tmux session of that name (it
  # exports no variable for it); RP_JOB overrides, the script's default is the fallback
  local tm=""
  [ -n "${TMUX:-}" ] && tm=$(tmux display-message -p '#S' 2>/dev/null)
  JOB="${RP_JOB:-${tm:-${1:?job name}}}"; MODEL="${2:?model key}"
  case "$JOB" in *.*|*:*) echo "!! job names may not contain '.' or ':' (rp run refuses them)"; exit 2 ;; esac
  # cross-pod sharding: SHARDS pods split the roles; this pod is SHARD (0-based)
  SHARD="${SHARD:-0}"; SHARDS="${SHARDS:-1}"
  SHARD_TAG=""; [ "$SHARDS" -gt 1 ] && SHARD_TAG="/shard$SHARD"
  LOGDIR="${LOGDIR:-/workspace/logs/$JOB-$MODEL}"; mkdir -p "$LOGDIR"
  rm -f "$LOGDIR/DONE" "$LOGDIR/FAILED"
  NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
  GEN_TP="${GEN_TP:-1}"
  POD_ID="$(pod_id)"
  trap on_exit EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  state "storage: /workspace is $WS_FSTYPE -> HF_HOME=$HF_HOME, vLLM venv under $ES_VENV_ROOT; $(df -h / /workspace 2>/dev/null | awk 'NR>1{printf "%s %s free; ", $6, $4}')"
  state "job $JOB model $MODEL shard $SHARD/$SHARDS on pod ${POD_ID:-?}: $NGPU GPU(s), repo $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo -dirty), SHUTDOWN=${SHUTDOWN:-unset}, MAX_HOURS=${MAX_HOURS:-}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^/    /'
  arm_ceiling
}

pod_id() {
  local id="${RUNPOD_POD_ID:-}"
  if [ -z "$id" ] && [ -r /proc/1/environ ]; then
    id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)
  fi
  echo "$id"
}

self_stop() {  # SHUTDOWN=stop|terminate: stop billing through the RunPod REST API
  [ -n "${SHUTDOWN:-}" ] || { echo "(SHUTDOWN unset: pod left running -- rp down it when done)"; return 0; }
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${POD_ID:-}" ]; then
    echo "!! self-stop SKIPPED: RUNPOD_API_KEY or RUNPOD_POD_ID missing -- the pod is still billing; rp down it"
    return 0
  fi
  if [ "$SHUTDOWN" = terminate ]; then
    echo "== SHUTDOWN=terminate -> DELETE pod $POD_ID =="
    curl -s -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"
  else
    echo "== SHUTDOWN=stop -> stop pod $POD_ID =="
    curl -s -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
  fi
  echo
}

arm_ceiling() {  # hard stop after MAX_HOURS even if a step hangs (only when SHUTDOWN is set)
  [ -n "${SHUTDOWN:-}" ] && [ -n "${MAX_HOURS:-}" ] || return 0
  local secs; secs=$(awk "BEGIN{print int(${MAX_HOURS} * 3600)}")
  (
    sleep "$secs"
    state "!! hard ceiling of ${MAX_HOURS} h reached: uploading logs and stopping the pod"
    echo "FAILED ceiling ${MAX_HOURS}h" > "$LOGDIR/FAILED"
    timeout 600 bash -c "$(declare -f upload_logs sync_logs); PY='$PY' JOB='$JOB' MODEL='$MODEL' LOGDIR='$LOGDIR' SHARD_TAG='$SHARD_TAG' upload_logs" || true
    [ "${SHUTDOWN}" = terminate ] && SHUTDOWN=stop     # keep the disk for inspection
    self_stop
  ) &
  CEILING_PID=$!
  disown "$CEILING_PID"     # no job-control noise; killed explicitly in on_exit
}

sync_logs() {  # the rp job log next to the step logs
  [ -f "/workspace/$JOB.log" ] && cp "/workspace/$JOB.log" "$LOGDIR/rp_job.log"
  return 0
}

upload_logs() {
  [ -n "${HF_DATASET_REPO:-}" ] || return 0
  sync_logs
  $PY -u scripts/upload_hf.py --folder "$LOGDIR" --path-in-repo "logs/$JOB/$MODEL$SHARD_TAG" --message "logs $JOB $MODEL$SHARD_TAG" \
    > "$LOGDIR/upload_logs.out" 2>&1 || { echo "!! log upload failed (see $LOGDIR/upload_logs.out)"; return 1; }
}

upload() {  # upload <message> [scripts/upload_hf.py args; default --model $MODEL]
  local msg="$1"; shift
  [ -n "${HF_DATASET_REPO:-}" ] || { state "upload skipped (HF_DATASET_REPO unset): $msg"; return 0; }
  [ $# -gt 0 ] || set -- --model "$MODEL"
  if $PY -u scripts/upload_hf.py --message "$msg" "$@" >> "$LOGDIR/upload.log" 2>&1; then
    state "uploaded: $msg"
  else
    UPLOAD_FAILED=1
    state "!! upload FAILED: $msg (tail of upload.log below; continuing, the pod will not be terminated)"
    tail -n 5 "$LOGDIR/upload.log"
  fi
  upload_logs || true
}

on_exit() {
  local code=$?
  trap - EXIT TERM INT
  [ -n "${CEILING_PID:-}" ] && kill "$CEILING_PID" 2>/dev/null
  # shellcheck disable=SC2046
  kill $(jobs -p) 2>/dev/null
  if [ "$code" = 0 ]; then
    echo "DONE $(date -u +%FT%TZ)" > "$LOGDIR/DONE"; state "DONE after $(elapsed)"
  else
    echo "FAILED code=$code $(date -u +%FT%TZ)" > "$LOGDIR/FAILED"; state "FAILED code=$code after $(elapsed)"
  fi
  upload_logs || UPLOAD_FAILED=1
  if [ "${SHUTDOWN:-}" = terminate ] && { [ "$code" != 0 ] || [ "$UPLOAD_FAILED" = 1 ] || [ -z "${HF_DATASET_REPO:-}" ]; }; then
    state "SHUTDOWN=terminate downgraded to stop (exit $code, upload_failed=$UPLOAD_FAILED): the disk is kept"
    SHUTDOWN=stop
  fi
  self_stop
  exit "$code"
}

finish() { exit "${1:-0}"; }   # the EXIT trap does the rest

judge_ready() {  # 0 if a key for the Claude judges is set (Anthropic, or OpenRouter routed to Anthropic)
  [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${OPENROUTER_API_KEY:-}" ]
}

doctor() {  # doctor <need_gb> [--judge]
  local need="$1"; shift
  $PY -u scripts/pod/doctor.py --models "$MODEL" --need-gb "$need" --shutdown "${SHUTDOWN:-}" --pod-id "${POD_ID:-}" "$@" \
    2>&1 | tee "$LOGDIR/doctor.log"
  return "${PIPESTATUS[0]}"
}

need_gb() {  # free disk needed: ~60 GB weights (unless cached) + ~25 GB venvs/outputs + margin
  if [ -d "$HF_HOME/hub/models--$(model_hf_dir)" ]; then echo 40; else echo 100; fi
}

model_hf_dir() {
  $PY - "$MODEL" <<'PY'
import sys, yaml
print(yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]["hf_id"].replace("/", "--"))
PY
}

start_download() {  # weights in the background, in parallel with the venv install
  $PY -u scripts/download_model.py --model "$MODEL" > "$LOGDIR/download.log" 2>&1 &
  DL_PID=$!
}

wait_download() {
  wait "$DL_PID" || { state "!! weight download failed"; tail -n 20 "$LOGDIR/download.log"; finish 4; }
  state "weights ready ($(tail -n 1 "$LOGDIR/download.log"))"
}

es_install() {  # always re-run: idempotent, re-applies container-disk parts after a stop
  bash scripts/pod/install_vllm.sh > "$LOGDIR/install_es.log" 2>&1 \
    || { state "!! vLLM venv install failed"; tail -n 30 "$LOGDIR/install_es.log"; finish 3; }
  # shellcheck disable=SC1091
  . /workspace/.es_env
  state "vLLM venv $ES_VENV ($ES_CUDA_LINE): $(grep -E '^vllm ' "$LOGDIR/install_es.log" | tail -n 1)"
  $PY -m pip install -q --no-deps "fla-core==0.5.2" "flash-linear-attention==0.5.2" einops 2>/dev/null \
    && echo "flash-linear-attention installed" || echo "(flash-linear-attention unavailable: HF uses the torch fallback)"
}

es_py() {  # python of the vLLM venv, with its bin on PATH (ninja) and the compat libs
  PATH="$ES_VENV/bin:$PATH" LD_LIBRARY_PATH="${ES_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ES_VENV/bin/python" "$@"
}

fanout() {  # fanout <log-prefix> <tp> <python-runner> <script> [args...]: one worker per TP group of GPUs
  # (global worker index SHARD x nw + i of SHARDS x nw, so several pods can split one grid)
  local prefix="$1" tp="$2" runner="$3"; shift 3
  local nw=$(( NGPU / tp )); [ "$nw" -ge 1 ] || nw=1
  local pids=() i devs fail=0
  for ((i = 0; i < nw; i++)); do
    devs=$(seq -s, $((i * tp)) $((i * tp + tp - 1)))
    CUDA_VISIBLE_DEVICES=$devs $runner -u "$@" --worker "$((SHARD * nw + i))" --num-workers "$((SHARDS * nw))" \
      > "$LOGDIR/${prefix}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [ $fail != 0 ]; then
    state "!! a $prefix worker failed"
    tail -n 20 "$LOGDIR/${prefix}"_*.log
    return 1
  fi
  return 0
}
