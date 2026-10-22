# Brief for the cloud session (transgression probe)

**Context.** Tim Farrelly & Olivia's Neel Nanda sprint project asks whether emotion probes help model forensics. Mentor Sen (GDM, 21 Oct): prediction AUROC isn't the interesting part. The open question is whether a model doing forward search toward a hack *knows* it is transgressing, or is acting amorally. NLAs can't settle this, because they read the text. A probe trained out-of-distribution (emotion-pipeline style, on stories) that still fires on real cheats is a "fairly good target", if we can argue it tracks the model's knowledge rather than textual patterns of cheating.

Prior results (the full write-up `docs/WRITEUP_GDOC.md` is not in the repo):
- GPT-OSS-120B cheats on the impossible LRU-cache task in 44/50 runs, usually openly and without refusing first.
- Emotion probes show defeat before the cheat and calm/relief plus some guilt after, but tie random directions for prediction (AUROC ~0.75–0.83 vs 0.69 for turn number alone).
- A Goodfire reward-hacking direction transfers poorly; single emotions (a drop in "nervous") pick out the cheat turn better.
- Finding 7: on palisades/tower-of-hanoi with an explicit "no scripting" rule, GPT-OSS states the rule and then breaks it a few turns later.
- Activations: HF `timf34/emotion-ensemble-forensics` under `context_rot/`. Emotion vectors: GitHub `timf34/emotion-vectors-collection`. Push HF results public.

**Task.**
1. Work on this branch (`cloud/transgression-probe`). Read `transgression/HANDOFF.md`, then `docs/transgression_probe_plan.md`. Don't rerun `norm_variants.py`: it needs laptop-only rollouts, and its outputs are committed.
2. Post a plan summary to Tim (under 15 lines), including the decisions at the end of the plan doc, each with the default you'll pick, then proceed without waiting. Tim: "if they need to do pod jobs, like through runpod, i trust them to go ahead with the notes and context that they already have!" Launch the RunPod job as HANDOFF.md describes (≈2×H200, ~1.5 h, ~$15). Check in before going past ~$40 total. Always terminate the pod once results are on HF. If the sandbox can't reach RunPod over SSH, say so and stop.
3. While the pod runs, write `transgression/analyze.py` as HANDOFF.md specifies. It should cover:
   (a) held-out story AUROC and matched-pair accuracy, against the guilt/regret/ashamed emotion vectors;
   (b) cheat-turn detection on LRU vs. the emotion probes, random directions and the Goodfire direction;
   (c) the allow-vs-forbid control (same action tokens, different norm context);
   (d) the palisades rule-restating runs.
4. Done = `docs/transgression_probe_results.md` plus figures in `docs/figures/`, pushed to this branch, with a 5-line Slack-ready summary for Sen that opens with one sentence of context. No push to `main`, no PR unless Tim asks.

**Tim's decision (22 Oct) on plan decision 2, story writer:** keep GPT-OSS-120B writing its own stories (on-policy), via the OpenRouter API (`openai/gpt-oss-120b`, as `transgression/stories.py` already does), never through our own GPU-hosted copy. The 2,935 texts are already generated and committed. If any stories need adding or regenerating, use OpenRouter. The pod is only for reading activations.
