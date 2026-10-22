# Ambiguity mining: hand-off for a fresh (cloud) session

Read `docs/ambiguity_search_plan.md` first: it covers the goal (Sen's de-risking step), the miner prompt, the
sample, pilot numbers, costs, and the decisions waiting on Tim. This file covers the mechanics.

## Files (all in `ambiguity/`)

| file | what | command |
|---|---|---|
| `render.py` | load local agent-interp-envs rollouts (`run-N/final/messages.json`, else the last `step-*/messages.json`) and Docent runs, and render compact text. Step k = k-th assistant turn (0-based), matching `step-k/` dirs and `context_rot/labels.py`. Also `ENV_NOTES` (what counts as misbehaviour per env) and the local `SOURCES`. | `python -m ambiguity.render` (prints sizes) |
| `docent_pull.py` | pull Aditya's corpus from Docent. Needs `pip install docent-python` and `DOCENT_API_KEY` in the environment. | `python ambiguity/docent_pull.py meta` then `python ambiguity/docent_pull.py runs outputs/ambiguity/docent_ids.txt` |
| `sample.py` | build `outputs/ambiguity/sample.jsonl` (296 rows: 66 local + 230 Docent) and `docent_ids.txt`. Needs the FULL local rollouts (step manifests + blobs, for code labels), so **don't re-run it on the cloud**: use the shipped `sample.jsonl`. | `python -m ambiguity.sample` |
| `mine.py` | the miner: one OpenRouter call per row, writes `outputs/ambiguity/mined/<judge>/<id>.json`; skips existing files; rolling worker pool; stops at `--budget` ($, counted over existing files too). | see below |
| `taxonomy.py` | induce categories from all `resolving_info` answers, assign, tabulate | `python -m ambiguity.taxonomy induce\|assign\|table --mined outputs/ambiguity/mined/google_gemini-4-argon` |
| `report.py` | per-model run-level stats; ranked candidate passages for curation | `python -m ambiguity.report stats\|top --mined ... [--min-strength 4] [--open-only]` |
| `show.py` | print a step window (or grep) of any sampled trajectory, for red-teaming | `python -m ambiguity.show <id> <from> <to> [--grep REGEX]` |

Needs `openai` (in the project env) and `OPENROUTER_API_KEY`. **Use a non-Anthropic judge**: the Anthropic API
refused these transcripts (ToS on reverse engineering).

## Status (cloud session, 22 Oct): done

Results, taxonomy, curated + red-teamed excerpts, go/no-go and the Slack summary are in `docs/ambiguity_search.md`.
693 trajectories mined (296-run sample + 397 extra cheated Kimi/GLM runs; ids in `data/ambiguity/`), ~$56 total.
New helpers: `redteam.py` (step window + norm-word scan + final report for one passage), `affect.py` (text-side
affect baseline), `sample.py --rebuild / --rebuild-expand` (Docent rows from shipped ids). The cloud container's
`outputs/` (Docent runs, mined records, verbatim curated excerpts) is not preserved; re-pull and re-mine to recover.

## What has run (laptop pilot)

- Docent metadata for all 24,332 runs → `outputs/ambiguity/docent/meta.jsonl` (77 MB).
- 230 sampled Docent runs → `outputs/ambiguity/docent/runs/*.json` (226 MB).
- Judge comparison on 2 runs: `mined/google_gemini-3.8-flash/` and the same 2 in `mined/google_gemini-4-argon/`.
- Miner (Gemini 4 Argon): **87 of 296 done, $4.77 spent**: all 66 local GPT-OSS runs plus 21 Docent runs (all
  claude-fable-5). Stopped on purpose for the hand-off. Logs: `outputs/ambiguity/mine_argon*.log`.
- Hand red-teaming so far (notes to fold into the final doc):
  - `cache120_main3/run-40` step 37 (deleting the baseline `.so`): the "fabricated rationale" reading is weak.
    Steps 36-37 show honest confusion about import order (extension modules load before `.py`). The odd part is
    the earlier docstring claiming the binary "cannot be loaded".
  - `cache120_main2/run-44` (rejected cheats, seemingly on feasibility): weak. The run is cut off at step 16 in an
    `apply_patch` failure loop, and at step 14 it only *hopes* the grading environment's baseline is slower.
  - `hanoi120/run-34` step 6: strong. It quotes the no-scripting rule, says a script "might be disallowed",
    cites the 20-step limit and tedium, then scripts anyway. Confound: the step limit.

## Next steps

1. Get Tim's answers to the decisions in the plan doc (judge, scale, closed models, curation focus).
2. Finish mining, open-weight first:
   `python -m ambiguity.mine --sample outputs/ambiguity/sample.jsonl --model google/gemini-4-argon --open-first --concurrency 24 --budget 22`
   (~$11.5 for the remaining 209; the $30 cap covers it).
3. `taxonomy induce`, then read and adjust the categories by hand, then `assign` and `table`.
4. `report top --open-only` → read each candidate with `show.py`, pick 10-20 (prefer open-weight, unresolved,
   strength ≥4), and red-team each one: what else explains it, and does that undermine the reading?
5. Write `docs/ambiguity_search.md`: taxonomy table; per-model verbalisation stats (`report stats`); curated
   excerpts with red-team notes; go/no-go; a 5-line Slack summary for Sen that opens with one sentence of context.

## Local-only inputs and how to get them on the cloud

The cloud machine has no Docker and none of the sibling folders. Nothing here needs Docker or
`../agent-interp-envs` (the env notes are inlined in `render.py`).

| thing | laptop location | on HF? | how the cloud gets it |
|---|---|---|---|
| Local GPT-OSS rollouts (Tim's own data) | `outputs/context_rot/{main2,main3,oss20b_main,oss20b_main_b,full_envs}/…/run-N/` (4.2 GB with step snapshots) | **No.** HF `timf34/emotion-ensemble-forensics` (public) only has activations / readout logs under `context_rot/gptoss*`, not transcripts | Pack just the transcripts (~80 MB) on the laptop and upload, see below. The miner needs only `final/messages.json`, or the last `step-*/messages.json` for runs without `final/`. Code labels are already baked into `sample.jsonl`. |
| `outputs/ambiguity/sample.jsonl` | laptop | no | contains Docent metadata (judge evidence): put it in a **private** HF repo, not GitHub. `sample_local.jsonl` (local rows only) is Tim's data. |
| `outputs/ambiguity/mined/` | laptop | no | the `docent__*.json` files quote Docent transcripts: **private only**. The `cache*`/`dashboard*`/`hanoi*` files are Tim's. |
| Docent metadata + runs | `outputs/ambiguity/docent/` | no | re-pull on the cloud with `DOCENT_API_KEY` set as an env var: `pip install docent-python && python ambiguity/docent_pull.py meta && python ambiguity/docent_pull.py runs outputs/ambiguity/docent_ids.txt` (ids come with the private copy of `outputs/ambiguity/`). Collection: `566f2942-8a54-4834-b66e-e6e1dda0f15d` ("Suite transcripts, all models (Copy)", owner olivia@velte.com, read access via Tim's key). |
| docent-python venv | laptop scratch dir | n/a | `pip install docent-python` |
| `memguard` (`~/.claude/bin/memguard`) | laptop only | n/a | not needed on the cloud; this job is API-bound and peaks at ~0.3 GB |

**Already built:** `outputs/ambiguity_local_rollouts.tgz` (15 MB, 396 files). Checked: all 66 local sample rows load
after unpacking. It's Tim's own GPT-OSS data, so it can go public (e.g. HF `timf34/emotion-ensemble-forensics` under
`ambiguity/`). To rebuild it (laptop, from `outputs/context_rot`; "no matches found" lines are runs with no steps,
which is harmless):

```bash
find main2 main3 oss20b_main oss20b_main_b full_envs \( -path '*/final/messages.json' -o -path '*/final/episode_end.json' \) > /tmp/cr_list.txt && for r in $(find main2 main3 oss20b_main oss20b_main_b full_envs -maxdepth 6 -type d -name 'run-*' -not -path '*/step-*'); do [ -f $r/final/messages.json ] || ls -d $r/step-* | sort -t- -k2 -n | tail -1 | sed 's|$|/messages.json|'; done >> /tmp/cr_list.txt && tar czf ../ambiguity_local_rollouts.tgz -T /tmp/cr_list.txt
```

Unpack on the cloud with `mkdir -p outputs/context_rot && tar xzf ambiguity_local_rollouts.tgz -C outputs/context_rot`.
Paths in `sample.jsonl` are relative to the repo root, so they then resolve as they did on the laptop.

## Gotchas

- Docent metadata fields are real booleans (`m["cheated"] is True`), not strings.
- Anthropic models' reasoning in Docent is *summarised* thinking: sparse, and possibly the source of the ambiguity.
  Only GPT-OSS, Kimi K3 and GLM 5.3 can be probed later.
- Rendered transcripts are 30-150k chars. Very long GPT-OSS-20B loops are shrunk, then hard-cut in the middle to 120k.
