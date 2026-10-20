#!/usr/bin/env bash
# Feasibility pilot for ONE model (configs/pilot.yaml): SRH (119 scored + 79 descored prompts
# x 3 seeds) and Impossible-LiveCodeBench (50 problems x 3 splits x 3 seeds, <= 5 attempts),
# thinking off. Idempotent and resumable: re-running skips finished work.
#
# Launch (laptop, from the repo root):
#   rp up --name ef-pilot --gpu h200 --gpus 1 --volume none --volume-size 200 --disk 80
#   rp bootstrap ef-pilot --repo https://github.com/timf34/emotion-ensemble-forensics --env .env --req requirements/pod.txt
#   rp run ef-pilot --job pilot --env SHUTDOWN=stop -- bash scripts/pod/run_pilot.sh qwen3.6-27b
#   rp logs ef-pilot --job pilot -n 40
#
# Knobs (env): SKIP_SMOKE=1 | DOCTOR_ONLY=1 | SMOKE_ONLY=1 | DATASETS="srh_pilot ib_pilot" |
#   GEN_TP=2 | SHUTDOWN=stop|terminate | MAX_HOURS=8
#
# Steps: preflight -> weights || vLLM venv -> smoke (tiny generation, real-model hook check,
# vLLM->HF replay check) -> generation (vLLM) -> SRH judges (API, background) || extraction
# (HF) -> labels -> pilot report; results uploaded to $HF_DATASET_REPO under pilot/<model>/.
MODEL="${1:?usage: run_pilot.sh <model-key>}"
MAX_HOURS="${MAX_HOURS:-8}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init pilot "$MODEL"
DATASETS="${DATASETS:-srh_pilot ib_pilot}"
JUDGE_FLAG=""; judge_ready && JUDGE_FLAG="--judge"
TP_FLAG=""; [ "$GEN_TP" -gt 1 ] && TP_FLAG="--tensor-parallel-size $GEN_TP"

step "preflight"
doctor "$(need_gb)" $JUDGE_FLAG || finish 2
[ "${DOCTOR_ONLY:-0}" = 1 ] && finish 0
[ -n "$JUDGE_FLAG" ] || state "!! no ANTHROPIC_API_KEY / OPENROUTER_API_KEY: SRH labels will be missing (judge later)"

step "weights download (background) + vLLM venv"
start_download
es_install
wait_download

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  step "smoke test (outputs_smoke/)"
  # shellcheck disable=SC2086
  EF_OUTPUTS=outputs_smoke es_py -u scripts/02_generate.py --model "$MODEL" --smoke $TP_FLAG \
    > "$LOGDIR/smoke_generate.log" 2>&1 || { tail -n 30 "$LOGDIR/smoke_generate.log"; finish 5; }
  EF_OUTPUTS=outputs_smoke $PY -u scripts/04_extract_features.py --model "$MODEL" > "$LOGDIR/smoke_extract.log" 2>&1 \
    || { tail -n 30 "$LOGDIR/smoke_extract.log"; finish 6; }
  for c in hooks replay; do
    EF_OUTPUTS=outputs_smoke CUDA_VISIBLE_DEVICES=0 $PY -u scripts/gpu_checks.py "$c" --model "$MODEL" \
      | tee "$LOGDIR/smoke_$c.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || { state "!! smoke check $c FAILED"; finish 6; }
  done
  EF_OUTPUTS=outputs_smoke $PY -u scripts/03_score_behaviour.py --model "$MODEL" --datasets ib_pilot \
    > "$LOGDIR/smoke_labels.log" 2>&1 || { tail -n 30 "$LOGDIR/smoke_labels.log"; finish 6; }
  state "smoke test passed"
  upload "smoke outputs" --folder "outputs_smoke/$MODEL" --path-in-repo "smoke/$MODEL"
  [ "${SMOKE_ONLY:-0}" = 1 ] && finish 0
fi

step "generation: $DATASETS"
# shellcheck disable=SC2086
es_py -u scripts/02_generate.py --model "$MODEL" --datasets $DATASETS $TP_FLAG > "$LOGDIR/generate.log" 2>&1 \
  || { tail -n 30 "$LOGDIR/generate.log"; finish 8; }
grep "^\[generate\]" "$LOGDIR/generate.log" | tail -n 6
upload "pilot responses" --folder "outputs/$MODEL" --path-in-repo "pilot/$MODEL"

JUDGE=""
if [ -n "$JUDGE_FLAG" ] && [[ " $DATASETS " == *" srh_pilot "* ]]; then
  $PY -u scripts/03_score_behaviour.py --model "$MODEL" --datasets srh_pilot > "$LOGDIR/judge.log" 2>&1 &
  JUDGE=$!
fi

step "extraction (HF replay)"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/04_extract_features.py --model "$MODEL" --datasets $DATASETS \
  > "$LOGDIR/extract.log" 2>&1 || { tail -n 30 "$LOGDIR/extract.log"; finish 10; }
if [ -n "$JUDGE" ]; then
  wait "$JUDGE" || state "!! SRH judges failed (see judge.log)"
fi

step "labels + pilot report"
# shellcheck disable=SC2086
$PY -u scripts/03_score_behaviour.py --model "$MODEL" --datasets $DATASETS --no-judge > "$LOGDIR/labels.log" 2>&1 \
  || { tail -n 30 "$LOGDIR/labels.log"; finish 11; }
# shellcheck disable=SC2086
$PY -u scripts/05_pilot_report.py --model "$MODEL" --datasets $DATASETS | tee "$LOGDIR/pilot_report.txt"
[ "${PIPESTATUS[0]}" = 0 ] || finish 12
cp "outputs/$MODEL/pilot_report.md" "$LOGDIR/" 2>/dev/null
upload "pilot activations, labels, report" --folder "outputs/$MODEL" --path-in-repo "pilot/$MODEL"
finish 0
