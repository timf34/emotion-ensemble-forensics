# Brief for the cloud session (ambiguity search)

**Context.** Tim Farrelly & Olivia's Neel Nanda sprint project asks whether emotion probes help model forensics. Mentor Sen (GDM, 21 Oct) set the first de-risking step: find chain-of-thought passages in forensics-relevant agent trajectories that are ambiguous in a way that matters, and where knowing the model's internal state (emotion, awareness of transgressing, grader beliefs, ...) would plausibly resolve the ambiguity. Build an open-ended taxonomy of what hidden information would resolve each one. Key ambiguity: the model reasons as if a hack is fine; is it amoral, or does it know and not say? If nothing convincing turns up, skip the probe validation.

The laptop pilot mined 87/296 runs with google/gemini-4-argon via OpenRouter ($4.77): all 66 local GPT-OSS runs plus 21 Fable 5 runs. The corpus is Aditya's Docent collection (24,332 runs, 8 models × 10 envs). Kimi K3 and GLM 5.3 are the only open-weight models in it, the only ones that can get probes later, and they are not mined yet.

**IMPORTANT:** this GitHub repo is public and Aditya's corpus is unpublished. Never commit Docent transcripts, Docent-derived outputs or verbatim Docent quotes. Keep them under `outputs/` (gitignored); in committed docs, cite run ids and paraphrase. The Anthropic API refused to judge these transcripts before, so use non-Anthropic judges via OpenRouter. Env vars: `DOCENT_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`.

**Task.**
1. Work on this branch (`cloud/ambiguity-search`). Read `ambiguity/HANDOFF.md`, `docs/ambiguity_search_plan.md` and `data/ambiguity/README.txt`.
2. Restore the inputs:
   - copy `data/ambiguity/*` into `outputs/ambiguity/` as the README says;
   - re-pull the 229 sampled Docent runs via `ambiguity/docent_pull.py` and `data/ambiguity/docent_ids.txt`;
   - rebuild the Docent rows of `sample.jsonl`, using `sample_local.jsonl` for the local rows (adapt `sample.py` if needed).
3. Post a plan summary to Tim (under 15 lines), including the decisions at the end of the plan doc, each with the default you'll pick, then proceed without waiting. Defaults:
   - mine Kimi K3 / GLM 5.3 first (`--open-first`), then finish the 296-run sample;
   - re-mine the 21 Fable 5 Docent runs (~$1);
   - mining every cheated Kimi/GLM run (~$35 more) is fine if the open-weight runs look promising.
   Check in with Tim before total API spend passes ~$60.
4. Run the taxonomy (induce / assign / table). Curate 10–20 of the strongest excerpts, preferring open-weight models, and red-team each one.
5. Done = `docs/ambiguity_search.md`: the taxonomy table (counts per category and model), the curated excerpts with red-team notes, an honest go/no-go on whether emotion or transgression probes look worth validating, and a 5-line Slack-ready summary for Sen that opens with one sentence of context. Push code and docs (no Docent content) to this branch. No push to `main`, no PR unless Tim asks.
