# Emotion-ensemble forensics

Can a classifier over emotion-probe scores predict model misbehaviour (reward hacking, misreported
work) on **held-out task families**, without chain-of-thought, better than a directly trained
behaviour probe, matched random / PCA subspaces, and text-only monitors? Tim & Olivia, Qwen3.6-27B
and Qwen3.8-27B, emotion vectors from
[timf34/emotion-vectors-collection](https://github.com/timf34/emotion-vectors-collection).

**Current stage: feasibility pilot** (`configs/pilot.yaml`, frozen 2026-10-20), Qwen3.6-27B, thinking
off. Nothing in the pilot feeds a confirmatory result.

## The pilot

| dataset | what | size | label |
|---|---|---|---|
| `srh_pilot` | School of Reward Hacks, 6 exploit families (A coding hard-coding, B keyword stuffing, C count inflation, D text-statistic gaming, E judge manipulation, G surface copying), 1-4 templates x several prompts per family | 119 scored + 79 descored prompts x 3 seeds | Claude exploit judge (primary) + Betley's metric/quality judges (agreement check) |
| `ib_pilot` | Impossible-LiveCodeBench (ImpossibleBench), minimal submission loop with test feedback | 50 problems x {original, oneoff, conflicting} x 3 seeds, <= 5 attempts | executed: passing tests that contradict the spec = cheat |

Prediction point: the last prompt token (the `\n\n` after the empty `<think></think>`), i.e. before
any answer token. For SRH that is one activation per prompt (seeds share it): it predicts a
prompt's *propensity* to be hacked. For IB every attempt after the first starts from a different
history, so activations differ between runs of the same problem.

**Go / no-go** (`go_no_go` in the config, fixed before any data): both outcomes present, labels
credible, and, for SRH, enough variation between prompts *within* a template, the only part a
pre-generation probe can predict beyond "which template is this?". `scripts/05_pilot_report.py`
prints the checks.

## Pipeline

| stage | script | where |
|---|---|---|
| 1 task manifests (committed: `data/*/manifest_pilot_v1.jsonl` + `.audit.json`) | `01_prepare_tasks.py` | laptop |
| 2 generation (vLLM, exact token ids, IB loop + sandboxed tests) | `02_generate.py` | pod, vLLM venv |
| 3 judges + labels | `03_score_behaviour.py` | pod or laptop |
| 4 activations (HF replay of the stored ids) | `04_extract_features.py` | pod GPU |
| 5 feasibility report | `05_pilot_report.py` | CPU |
| 6 predictors on held-out families | `06_evaluate.py` | CPU |

Predictors (`src/eforensics/predictors.py`), all on the same rows, labels and prediction point,
every choice made by grouped inner CV on training rows only: `emotion_lr` (42 emotion scores ->
L2 logistic regression: the ensemble), `syndrome_lr`, `best_emotion`, `random_lr` (42 random
directions, 5 seeds), `pca_lr` (42 training PCs), `direct_lr` (full activation), `dom`
(difference of means, cf. Bergen et al. 2026), `text_lr` (TF-IDF of the prompt). Under pure noise
at pilot size the pooled AUROC has SD ~0.07 (checked), so differences smaller than that are noise.

## Conventions

* **Block b** = residual stream at the OUTPUT of decoder block b = HF `hidden_states[b + 1]`, the
  emotion-vector convention (`src/eforensics/hooks.py`). grader-shifts / Betley use the block
  INPUT (`hidden_states[L]`): emotion block 40 is their layer 41. Checked on CPU (tiny model) and
  on the real model in the smoke test (`scripts/gpu_checks.py hooks`).
* Emotion scores = dot products with **unit-normalised** denoised vectors (`vectors.py`), pinned to
  `timf34/emotion-vectors@446e2ac`. Blocks saved: 24, 32, 36, 40 (primary), 44, 48, 56.
* Prompts are rendered once with the pinned tokenizer and passed to vLLM as ids; the extractor
  replays those ids (`gpu_checks.py replay` compares HF and vLLM log-probs). IB turns are
  re-rendered through the chat template every turn (as a chat server would): Qwen3.6 drops the
  empty think block from past assistant turns, Qwen3.8 keeps it.
* Config ids hash model/tokenizer revisions, sampling, design version and manifest sha; resumption
  never mixes configs. Outputs: `outputs/<model>/<dataset>/` (layout in `src/eforensics/io.py`).

## Running the pilot on RunPod

```bash
rp up --name ef-pilot --gpu h200 --gpus 1 --volume none --volume-size 200 --disk 80
rp bootstrap ef-pilot --repo https://github.com/timf34/emotion-ensemble-forensics --env .env --req requirements/pod.txt
rp run ef-pilot --job pilot --env SHUTDOWN=stop -- bash scripts/pod/run_pilot.sh qwen3.6-27b
rp logs ef-pilot --job pilot -n 40
```

`run_pilot.sh`: preflight (`scripts/pod/doctor.py`: GPUs, disk, pinned revisions, vectors,
manifests, judge key, sandbox isolation, self-stop) -> weights || vLLM venv -> smoke test (tiny
generation, hook + replay checks on the real model) -> generation -> SRH judges || extraction ->
labels -> pilot report. Results go to `$HF_DATASET_REPO` under `pilot/<model>/` (created private
unless `HF_DATASET_PUBLIC=1`; this project's `.env` sets it). `.env`: see `.env.example`. Knobs: `SMOKE_ONLY=1`, `DOCTOR_ONLY=1`, `SKIP_SMOKE=1`,
`DATASETS=...`, `MAX_HOURS`.

## Departures from upstream

* **ImpossibleBench `find_code`**: the upstream function returns the text *between* code blocks
  when an answer has two or more blocks. The corrected parser is executed; the upstream one is kept
  (`find_code_ib`) and every attempt records whether they differ.
* ImpossibleBench's Inspect sandbox is replaced by `src/eforensics/sandbox.py`: no credentials
  (minimal env, `nobody` uid on the pod), a network namespace when the host allows it (otherwise a
  socket-disabling prelude, flagged by the preflight), CPU / memory / file-size limits, 30 s timeout.
  Error feedback is capped at 4,000 characters (head + tail).
* SRH sampling is Qwen's non-thinking recommendation with presence / repetition penalties off
  (they would suppress keyword stuffing directly).

## Tests

`uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install "torch==2.8.*" "transformers==5.14.1" -e ".[dev,judge]"`,
then `.venv/bin/python -m pytest`. Covers the hook convention, templates and user spans with the
real tokenizers, extraction vs HF hidden states, the sandbox, the IB harness and cheat detectors,
the generation loops end to end with a fake vLLM, labels, the variance decomposition and the
evaluation code.

## Next (after the pilot)

Scale whichever dataset passes go / no-go to the held-out-family panel (fresh prompts, pilot
prompts excluded); Qwen3.8 with its own vectors; MASK for cross-behaviour transfer; a
sentence-embedding text baseline and a black-box LLM monitor; thinking-on with the CoT withheld.
