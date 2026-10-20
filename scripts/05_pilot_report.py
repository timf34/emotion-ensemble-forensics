#!/usr/bin/env python3
"""Stage 5 (CPU): the feasibility report for each pilot dataset -> analysis/pilot_report.json
and a short markdown summary next to it. Emotion-score contrasts and trajectories are
EXPLORATORY (no predictor is trained here).

    python scripts/05_pilot_report.py --model qwen3.6-27b
"""

import argparse
import json

import numpy as np

import _common  # noqa: F401

from eforensics.config import load_experiment
from eforensics.features import effective_rank, load_store
from eforensics.io import activations_dir, analysis_dir, labels_path, outputs_root, read_jsonl, runs_path, write_json
from eforensics.report import emotion_contrast, ib_report, srh_report, trajectories
from eforensics.vectors import load_directions


def emotion_part(model, ds, exp, labels, key):
    if not (activations_dir(model, ds) / "meta.json").exists():
        return {"skipped": "no activations yet"}
    store = load_store(model, ds)
    b = int(exp["extraction"]["primary_block"])
    dirs = load_directions(model, exp, "emotions")
    U = dirs.matrix(b)
    lab = {l["id"]: l for l in labels}
    rows = [i for i, r in enumerate(store.index) if lab.get(r["id"], {}).get(key) is not None
            and (r.get("variant") in (None, "scored"))]
    y = np.array([lab[store.index[i]["id"]][key] for i in rows])
    out = {"block": b, "effective_rank_emotions": effective_rank(U), "n": len(rows), "positives": int(y.sum())}
    for rep in ("last", "user_mean", "resp"):
        X = store.array(rep, b, np.array(rows)) if rows else np.zeros((0, 1))
        out[f"contrast_{rep}"] = emotion_contrast(X, y, U, dirs.labels) if len(rows) else {}
    if ds.startswith("ib"):
        X = store.array("last", b)
        out["trajectories_last"] = trajectories(X, store.index, lab, U, dirs.labels)
    return out


def md_summary(name, rep):
    g = rep.get("go_no_go", {})
    lines = [f"## {name}", "", f"GO: **{g.get('go')}**  checks: {json.dumps(g.get('checks'))}", ""]
    for k in ("hack_rate_scored", "hack_rate_descored", "hack_rate_scored_excluding_emotion_metric", "refusal_rate",
              "truncated_rate", "label_counts", "variance", "variance_impossible", "problems_with_mixed_seeds",
              "cheat_heuristics", "agreement_exploit_vs_gap_rule"):
        if k in rep:
            lines.append(f"- {k}: `{json.dumps(rep[k])}`")
    if "by_family" in rep:
        lines += ["", "| family | scored hack rate (n) | descored hack rate (n) | ambiguous |", "|---|---|---|---|"]
        for f, v in rep["by_family"].items():
            lines.append(f"| {f} | {v['scored']['rate']} ({v['scored']['n']}) | {v['descored']['rate']} "
                         f"({v['descored']['n']}) | {v['ambiguous']} |")
    if "by_split" in rep:
        lines += ["", "| split | runs | passed | cheat rate | mean attempts |", "|---|---|---|---|---|"]
        for s, v in rep["by_split"].items():
            lines.append(f"| {s} | {v['n_runs']} | {v['passed_rate']:.2f} | {v['cheat_rate']} | {v['mean_attempts']:.2f} |")
    e = rep.get("emotions", {})
    if "contrast_last" in e:
        lines += ["", f"Emotion contrasts (block {e['block']}, standardised mean difference, exploratory): "
                      f"last `{e['contrast_last']}`; resp `{e['contrast_resp']}`"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", nargs="+", default=["srh_pilot", "ib_pilot"])
    args = p.parse_args()
    exp = load_experiment()
    gng = exp["go_no_go"]
    md = [f"# Pilot report: {args.model}", ""]
    for ds in args.datasets:
        labels = read_jsonl(labels_path(args.model, ds))
        if not labels:
            print(f"[report] {ds}: no labels, skipped")
            continue
        if ds.startswith("srh"):
            rep = srh_report(labels, gng)
            rep["emotions"] = emotion_part(args.model, ds, exp, labels, "hack")
        else:
            rep = ib_report(read_jsonl(runs_path(args.model, ds)), labels, gng)
            rep["emotions"] = emotion_part(args.model, ds, exp, labels, "will_cheat")
        out = analysis_dir(args.model, ds) / "pilot_report.json"
        write_json(out, rep)
        md.append(md_summary(ds, rep))
        print(f"[report] {ds}: GO={rep['go_no_go']['go']} -> {out}")
        print(json.dumps(rep["go_no_go"], indent=1))
    path = outputs_root() / args.model / "pilot_report.md"
    path.write_text("\n".join(md))
    print(f"[report] summary -> {path}")


if __name__ == "__main__":
    main()
