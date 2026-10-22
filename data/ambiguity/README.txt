local copies of Tim's own (non-Docent) ambiguity outputs; outputs/ is gitignored. Copy into outputs/ambiguity/ (rollouts tgz -> outputs/ambiguity_local_rollouts.tgz) before running.

Docent-derived files (Aditya's unpublished corpus: sample.jsonl Docent rows, docent/ runs + meta, mined docent__*.json) are NOT in this public repo.
data/ambiguity/docent_ids.txt holds only the run ids of the 230 sampled Docent runs. On the cloud: re-pull them with
`python ambiguity/docent_pull.py meta` and `python ambiguity/docent_pull.py runs data/ambiguity/docent_ids.txt`, then rebuild the
Docent rows of sample.jsonl (sample.py's Docent branch; its local branch needs laptop-only rollouts, so use sample_local.jsonl for those).
Never commit Docent content to this repo.
