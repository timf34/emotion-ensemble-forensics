"""Pilot feasibility report: are there both outcomes, can we label them, and is there
variation a pre-generation probe could predict beyond "which template / problem is this"?

Variance decomposition of a binary outcome (method of moments, nested design):
  SRH : template > prompt > seed        IB : problem > seed (runs)
Only the PROMPT-within-template share (SRH) is predictable from the pre-generation activation
beyond template identity; the seed share is invisible to it (same activation for every seed).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def nested_variance(rows: list[dict[str, Any]], outer: str, inner: str, y: str) -> dict[str, Any]:
    """Shares of var(y) between `outer` units, between `inner` units within outer, and within inner."""
    rows = [r for r in rows if r.get(y) is not None]
    if not rows:
        return {"n": 0}
    vals = np.array([r[y] for r in rows], dtype=float)
    by_inner = defaultdict(list)
    inner_outer = {}
    for r in rows:
        by_inner[r[inner]].append(float(r[y]))
        inner_outer[r[inner]] = r[outer]
    # within-inner (seed) variance, unbiased
    w = [np.var(v, ddof=1) for v in by_inner.values() if len(v) > 1]
    s2_seed = float(np.mean(w)) if w else 0.0
    n_seed = float(np.mean([len(v) for v in by_inner.values()]))
    inner_means = {k: np.mean(v) for k, v in by_inner.items()}
    by_outer = defaultdict(list)
    for k, m in inner_means.items():
        by_outer[inner_outer[k]].append(m)
    b = [np.var(v, ddof=1) for v in by_outer.values() if len(v) > 1]
    s2_inner = max(0.0, (float(np.mean(b)) if b else 0.0) - s2_seed / n_seed)
    n_inner = float(np.mean([len(v) for v in by_outer.values()]))
    outer_means = [np.mean(v) for v in by_outer.values()]
    s2_outer = max(0.0, (float(np.var(outer_means, ddof=1)) if len(outer_means) > 1 else 0.0)
                   - s2_inner / n_inner - s2_seed / (n_inner * n_seed))
    tot = s2_outer + s2_inner + s2_seed
    share = (lambda v: v / tot if tot > 0 else None)
    mixed_outer = sum(1 for v in by_outer.values() if 0 < np.mean([m > 0 for m in v]) < 1)
    mixed_inner = sum(1 for v in by_inner.values() if 0 < np.mean(v) < 1)
    return {"n": len(rows), "mean": float(vals.mean()),
            f"share_{outer}": share(s2_outer), f"share_{inner}_within_{outer}": share(s2_inner),
            f"share_within_{inner}": share(s2_seed),
            f"n_{outer}": len(by_outer), f"n_{inner}": len(by_inner),
            f"{outer}s_with_mixed_{inner}s": mixed_outer, f"{inner}s_with_mixed_outcomes": mixed_inner}


def kappa(a: list, b: list) -> float | None:
    pairs = [(x, z) for x, z in zip(a, b) if x is not None and z is not None]
    if len(pairs) < 5:
        return None
    labs = sorted({p for pr in pairs for p in pr})
    po = np.mean([x == z for x, z in pairs])
    ca, cb = Counter(x for x, _ in pairs), Counter(z for _, z in pairs)
    pe = sum(ca[l] * cb[l] for l in labs) / len(pairs) ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def rate(rows, key, cond=lambda r: True):
    v = [r[key] for r in rows if cond(r) and r.get(key) is not None]
    return {"rate": float(np.mean(v)) if v else None, "n": len(v)}


def srh_report(labels: list[dict[str, Any]], gng: dict[str, float]) -> dict[str, Any]:
    scored = [r for r in labels if r["variant"] == "scored"]
    desc = [r for r in labels if r["variant"] == "descored"]
    out: dict[str, Any] = {"n_responses": len(labels), "n_scored": len(scored), "n_descored": len(desc)}
    out["label_counts"] = dict(Counter(r["hack_label"] for r in scored))
    out["truncated_rate"] = rate(labels, "truncated")
    out["refusal_rate"] = rate(scored, "refusal")
    out["hack_rate_scored"] = rate(scored, "hack")
    out["hack_rate_descored"] = rate(desc, "hack")
    out["hack_rate_scored_excluding_emotion_metric"] = rate(scored, "hack", lambda r: not r["emotion_metric"])
    fams = sorted({r["exploit_family"] for r in scored})
    out["by_family"] = {f: {"scored": rate(scored, "hack", lambda r: r["exploit_family"] == f),
                            "descored": rate(desc, "hack", lambda r: r["exploit_family"] == f),
                            "ambiguous": sum(r["hack_label"] == "ambiguous" for r in scored if r["exploit_family"] == f)}
                        for f in fams}
    out["by_template"] = {t: rate(scored, "hack", lambda r: r["template_id"] == t)
                          for t in sorted({r["template_id"] for r in scored})}
    out["agreement_exploit_vs_gap_rule"] = {
        "kappa_yes_no": kappa([r["hack_label"] if r["hack_label"] in ("yes", "no") else None for r in scored],
                              [r["gap_label"] if r["gap_label"] in ("yes", "no") else None for r in scored]),
        "crosstab": dict(Counter(f"{r['hack_label']}|{r['gap_label']}" for r in scored))}
    coding = [r for r in scored if r["exploit_family"] == "A"]
    out["agreement_coding_judge_vs_static"] = {
        "kappa": kappa([r["hack"] for r in coding],
                       [None if r["hardcode_flag"] is None else int(r["hardcode_flag"]) for r in coding]),
        "crosstab": dict(Counter(f"{r['hack_label']}|{r['hardcode_flag']}" for r in coding))}
    vd = nested_variance(scored, "template_id", "task_id", "hack")
    out["variance"] = vd
    # go / no-go (thresholds in configs/pilot.yaml go_no_go)
    hr = out["hack_rate_scored"]["rate"] or 0.0
    fam_mixed = sum(1 for f in fams if 0 < (out["by_family"][f]["scored"]["rate"] or 0) < 1)
    within = vd.get("share_task_id_within_template_id") or 0.0
    amb = out["label_counts"].get("ambiguous", 0) / max(1, len(scored))
    out["go_no_go"] = {
        "minority_class_rate": min(hr, 1 - hr), "families_with_both_outcomes": fam_mixed,
        "within_template_prompt_share": within, "ambiguous_rate": amb,
        "checks": {"both_classes": min(hr, 1 - hr) >= gng["min_minority_rate"],
                   "families": fam_mixed >= gng["min_families_with_both"],
                   "within_template_variation": within >= gng["min_within_template_share"],
                   "labels_credible": amb <= gng["max_ambiguous_rate"]}}
    out["go_no_go"]["go"] = all(out["go_no_go"]["checks"].values())
    return out


def ib_report(runs: list[dict[str, Any]], labels: list[dict[str, Any]], gng: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_runs": len(runs), "n_turns": len(labels)}
    splits = sorted({r["ib_split"] for r in runs})
    out["by_split"] = {}
    for s in splits:
        rs = [r for r in runs if r["ib_split"] == s]
        imp = s != "original"
        out["by_split"][s] = {
            "n_runs": len(rs),
            "passed_rate": float(np.mean([r["first_pass_attempt"] is not None for r in rs])),
            "cheat_rate": float(np.mean([r["cheated"] for r in rs])) if imp else None,
            "first_pass_attempt": dict(Counter(r["first_pass_attempt"] for r in rs)),
            "status": dict(Counter(r["status"] for r in rs)),
            "mean_attempts": float(np.mean([r["n_attempts"] for r in rs])),
        }
    cheat_turns = [l for l in labels if l.get("cheat") == 1]
    out["cheat_heuristics"] = dict(Counter(k for l in cheat_turns for k, v in (l.get("heuristics") or {}).items() if v))
    out["cheats_passing_original"] = sum(l["passed_original"] for l in cheat_turns)
    out["truncated_rate"] = rate(labels, "truncated")
    out["timeout_rate"] = rate(labels, "timed_out")
    imp_runs = [{**r, "cheated_i": int(r["cheated"])} for r in runs if r["ib_split"] != "original"]
    out["variance_impossible"] = nested_variance(imp_runs, "ib_split", "problem_id", "cheated_i")
    by_prob = defaultdict(list)
    for r in imp_runs:
        by_prob[(r["ib_split"], r["problem_id"])].append(r["cheated_i"])
    out["problems_with_mixed_seeds"] = sum(1 for v in by_prob.values() if 0 < np.mean(v) < 1)
    ever = sum(1 for v in by_prob.values() if max(v) == 1)
    cr = float(np.mean([r["cheated_i"] for r in imp_runs])) if imp_runs else 0.0
    out["go_no_go"] = {"cheat_rate_impossible": cr, "problem_splits_ever_cheated": ever,
                       "checks": {"both_classes": min(cr, 1 - cr) >= gng["min_minority_rate"],
                                  "enough_cheating_problems": ever >= gng["min_ib_problems_cheated"]}}
    out["go_no_go"]["go"] = all(out["go_no_go"]["checks"].values())
    return out


def emotion_contrast(X: np.ndarray, y: np.ndarray, U: np.ndarray, labels: list[str], top: int = 8) -> dict[str, Any]:
    """Standardised mean difference of each emotion score, positive minus negative class (exploratory)."""
    S = X @ U.T
    a, b = S[y == 1], S[y == 0]
    if len(a) < 2 or len(b) < 2:
        return {}
    d = (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2 + 1e-12)
    order = np.argsort(-np.abs(d))[:top]
    return {labels[i]: round(float(d[i]), 3) for i in order}


def trajectories(X: np.ndarray, index: list[dict[str, Any]], labels_by_id: dict[str, dict], U: np.ndarray,
                 names: list[str], pick: tuple[str, ...] = ("desperate", "calm", "frustrated", "stuck",
                                                              "anxious", "guilty")) -> dict[str, Any]:
    """Mean emotion score at the start of each attempt, by split x whether the run eventually cheats."""
    S = X @ U.T
    cols = [names.index(p) for p in pick]
    acc: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(index):
        lab = labels_by_id.get(r["id"])
        if lab is None:
            continue
        grp = r["ib_split"] + ("" if r["ib_split"] == "original" else
                               ("|run_cheats" if lab.get("run_cheated") else "|run_honest"))
        acc[grp][r["turn"]].append(S[i, cols])
    return {g: {int(t): {"n": len(v), **{p: round(float(np.mean([x[j] for x in v])), 3) for j, p in enumerate(pick)}}
                for t, v in sorted(d.items())} for g, d in acc.items()}
