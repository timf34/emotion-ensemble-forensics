# Handoff: knowing-transgression probe (for a fresh cloud session)

Read `docs/transgression_probe_plan.md` first (design, the four results we want, cost). Background on the
project: `docs/WRITEUP_GDOC.md` (Findings 0–8; Finding 7 = palisades "no scripting" rule).

**Status (22 Oct, laptop session):** dataset and replay transcripts built and committed under
`data/transgression/`; pod code tested on a tiny random GPT-OSS model; **no pod job has run** and
**no analysis code exists yet**. Tim wants to review the plan before the pod launches (≈ $15, ask him if
it would go over ~$20).

## What's built

| File | What it does | Command |
|---|---|---|
| `transgression/stories.py` | Story families via OpenRouter (writer = `openai/gpt-oss-120b`), leak/length checks, and `build`: 5 conditions per family with char spans of setup / norm / act / suffix | `python -m transgression.stories generate --n-per-cell 2` (resumable; already run) · `python -m transgression.stories build` |
| `transgression/norm_variants.py` | Allow/forbid(/orig/none) replay variants of the real cheat runs, truncated 2 turns after the cheat; writes `meta.json` (onset, palisades rule-restating turns) | `python -m transgression.norm_variants` (already run; **needs the raw rollouts, laptop only**, see below) |
| `transgression/readout.py` | Pod readers. `read_texts`: per story, mean residual over act / norm span, mean from token 50, last token. `build_sequence` + `SpanReader`: per turn, `pre` (generation prompt), `reason` (analysis channel mean), `action` (tool-call argument mean) | used by run_pod.py |
| `transgression/run_pod.py` | Loads a GPT-OSS model (MXFP4 → bf16, eager, as `context_rot/run_readout.py`), reads stories then transcripts; resumable; skips `lru/orig` by default | `python transgression/run_pod.py --model openai/gpt-oss-120b --out /workspace/tg/oss120b` |
| `transgression/pod_job.sh` | Whole pod job: 120B then 20B, uploads to HF `context_rot/transgression_gptoss{120b,20b}/`, stops the pod (`SHUTDOWN=stop`, `MAX_HOURS`) | see "Launching the pod" |

Data (all in git, ~45 MB, nothing secret; transcripts are GPT-OSS rollouts on the public agent-interp-envs tasks):

- `data/transgression/families.jsonl`: 587 families (60 settings × 5 norm types × 2, 13 rejected)
- `data/transgression/texts.jsonl`: 2,935 texts; `split` = train/test by setting (every 3rd setting is test)
- `data/transgression/norm_runs/{lru,palisades}/{cond}/run-N/final/messages.json` + `meta.json`: 279 sequences
  (lru: orig/forbid/allow × 44; palisades: forbid/allow/none × 49). Verified: all prefix-consistent, cheat-turn
  action tokens identical across conditions.
- `data/transgression/lru120b_labels_code.json`, `lru20b_labels_code.json`: code-based cheat labels
  (`onset` = turn that writes the cheat) for the 50 LRU runs of each model (from `context_rot/labels.py`).

## Launching the pod (from a machine without Tim's laptop setup)

Env vars on the cloud machine: `RUNPOD_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY` (no `.env`).

```bash
git clone https://github.com/timf34/runpod-runner && pip install -e runpod-runner   # `rp` CLI (public); editable,
                                                             # because it reads profiles/bootstrap.sh from the checkout
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519             # rp injects ~/.ssh/id_ed25519.pub into the pod
# a .env for the pod (rp bootstrap copies it into the repo root):
printf 'HF_TOKEN=%s\nRUNPOD_API_KEY=%s\nHF_DATASET_REPO=timf34/emotion-ensemble-forensics\nHF_DATASET_PUBLIC=1\n' \
  "$HF_TOKEN" "$RUNPOD_API_KEY" > /tmp/pod.env && chmod 600 /tmp/pod.env
rp up --name tg --gpu h200 --gpus 2 --volume none --volume-size 300 --disk 80     # prints $/h: expect ~$10.6
rp bootstrap tg --repo https://github.com/timf34/emotion-ensemble-forensics --branch <branch with this code> \
  --env /tmp/pod.env --req requirements/pod.txt
rp run tg --job tg --env SHUTDOWN=stop --env MAX_HOURS=1.8 -- bash transgression/pod_job.sh
rp logs tg --job tg -n 30          # don't poll in a tight loop; job ~1.3 h
rp down tg --terminate -y          # once results are on HF
```

- If the cloud sandbox can't open outbound SSH to RunPod, `rp up/run` won't work; tell Tim rather than routing
  around it. (`--volume-size` is the pod's own disk; previous jobs used `--volume none`.)
- Memory on the pod: 120B in bf16 ≈ 234 GB across 2× H200 (282 GB); stories batch 16, transcripts chunked at a
  4 GB attention budget (the old LRU readout ran with the same settings, 20–50 s/run).
- Expected outputs on HF: `stories.npz` (keys act / norm / story / last, `[2935, n_blocks, H]` fp16, plus
  `ids`, `blocks`), `stories_meta.jsonl` (token counts, act-token hash per text), `transcripts/<env>/<cond>/run-N.npz`
  (pre / reason / action `[n_turns, n_blocks, H]`, `steps` JSON with spans), `transcripts_log.jsonl`.

## Other inputs and where to get them

| Need | Where |
|---|---|
| Emotion vectors (guilty, ashamed, regretful, sorry baselines) | HF `timf34/emotion-vectors` (public): `gptoss_120b/vectors_emotions_dn.pt`, `gptoss_20b/...` → `{emotion: {layer: Tensor[H]}}`, layer b = output of block b. Also GitHub `timf34/emotion-vectors-collection` (20B only in git) |
| Existing LRU per-turn readouts (step 3) | HF `timf34/emotion-ensemble-forensics`, `context_rot/gptoss120b_lru_main3/run-N.npz` (pre / reason, blocks 6–35), and `context_rot/gptoss20b_lru_main/` |
| Goodfire reward-hacking direction | HF same repo, `context_rot/goodfire_gptoss120b/*.npz`; direction = `dom()` in `context_rot/goodfire_eval.py` (`load_gf`) |
| Cheat-turn detection reference | `context_rot/detect_cheat_turn.py`, helpers in `context_rot/analyze.py` (`load_runs`) and `context_rot/cheat_probe.py` (`auroc`, `boot_ci`). These expect HF downloads under `outputs/hf/context_rot/...` |
| Raw rollouts (only to regenerate norm_runs) | laptop only: `outputs/context_rot/main3/...` and `outputs/context_rot/full_envs/palisades_bounty/...`. Not needed: `norm_runs` is already built |
| `../agent-interp-envs`, `../emotion-concepts` | not needed. The palisades tool schema (`end_session`) is copied into `readout.py`; the story design borrows from emotion-concepts' masking prompt but nothing imports it |

Download pattern: `huggingface_hub.snapshot_download("timf34/emotion-ensemble-forensics", repo_type="dataset",
allow_patterns=["context_rot/gptoss120b_lru_main3/*", ...], local_dir="outputs/hf")`.

## What's next

1. Tim reviews `docs/transgression_probe_plan.md`; launch the pod (above).
2. Write `transgression/analyze.py` (local, CPU, < 2 GB):
   - **stories**: per layer, fit on train settings: DoM forbid_do − allow_do on `act`; same with
     (forbid_decline − allow_decline) projected out; logistic. Test settings: AUROC forbid_do vs allow_do,
     matched-pair accuracy, 2×2 interaction, unaware_do position, leave-one-norm-type-out. Pick the layer by CV
     within train. Baselines: guilt-family emotion vectors, 100 random unit directions, Goodfire, a "rule text"
     probe fitted on the `norm` span.
   - **LRU cheat turn**: apply the probe (no refit) to `gptoss120b_lru_main3` (44 cheat runs): event-locked
     trajectory (tau −8..+4), detection of tau = 0 vs earlier turns (pooled AUROC + within-run rank), vs emotions,
     random directions, Goodfire.
   - **norm control**: Δ(turn) = probe(forbid) − probe(allow), in units of the probe's SD; difference in
     differences Δ(cheat turn) − mean Δ(tau ≤ −2) per run, bootstrap CI over runs, compared with the same
     statistic for random directions (null), emotions and Goodfire; for pre / reason / action. Palisades: also
     forbid vs none, and runs that restated the rule (38/49, `meta.json` `states_rule_turns`) vs not.
3. Results doc `docs/transgression_probe_results.md` (methods, the four results, figures in `docs/figures/`),
   plus a 5-line Slack summary that opens with one sentence of context (Sen asked for that).

Known rough edges: the leak regex rejected 13 families (re-running `generate` fills them; the regex is now less
strict); palisades `states_rule_turns` comes from a loose regex (spot-check a few); 7 palisades runs cheat at
turn 0, so they have no pre-cheat turns for the difference in differences.
