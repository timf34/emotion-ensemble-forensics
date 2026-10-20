"""Held-out evaluation: leave one family out (outer), grouped inner CV for every choice.

Outer folds never see tuning. Per held-out family: AUROC, average precision, and detection
at a threshold set on the TRAINING rows' inner out-of-fold scores for a 10% false-positive
rate, with the ACTUAL held-out false-positive rate. Pooled out-of-fold AUROC gets a cluster
bootstrap CI over prompts/problems. Families with one class are reported, not scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .predictors import METHODS, _auc, inner_folds


def _ap(y, s) -> float | None:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float | None:
    neg = s[y == 0]
    return float(np.quantile(neg, 1 - fpr)) if len(neg) else None


def cluster_bootstrap_auc(y, s, clusters, n: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
    vals = []
    for _ in range(n):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        a = _auc(y[idx], s[idx])
        if a is not None:
            vals.append(a)
    if len(vals) < n // 2:
        return None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _fit_score(method: str, X_tr, y_tr, g_tr, X_te, ctx) -> tuple[np.ndarray, dict, np.ndarray | None]:
    fit = METHODS[method]
    if method == "text_lr":
        f, info = fit(None, y_tr, g_tr, {**ctx, "texts_train": ctx["texts_tr"]})
        s_te = f(ctx["texts_te"])
    else:
        f, info = fit(X_tr, y_tr, g_tr, ctx)
        s_te = f(X_te)
    # inner OOF scores on the training rows, for the operating threshold only
    oof = np.full(len(y_tr), np.nan)
    for tr, va in inner_folds(g_tr, y_tr):
        if len(np.unique(y_tr[tr])) < 2:
            continue
        if method == "text_lr":
            texts = ctx["texts_tr"]
            fi, _ = fit(None, y_tr[tr], g_tr[tr], {**ctx, "texts_train": [texts[i] for i in tr]})
            oof[va] = fi([texts[i] for i in va])
        else:
            fi, _ = fit(X_tr[tr], y_tr[tr], g_tr[tr], ctx)
            oof[va] = fi(X_tr[va])
    return np.asarray(s_te, dtype=float), info, oof


def leave_one_family_out(X: np.ndarray, y: np.ndarray, family: np.ndarray, groups: np.ndarray, methods: list[str],
                         ctx: dict[str, Any], texts: list[str] | None = None, fpr: float = 0.10,
                         random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    fams = sorted(np.unique(family))
    results: dict[str, Any] = {"families": {}, "pooled": {}, "n": int(len(y)), "positives": int(y.sum())}
    pooled = {m: np.full(len(y), np.nan) for m in methods}
    for fam in fams:
        te = family == fam
        tr = ~te
        entry: dict[str, Any] = {"n": int(te.sum()), "positives": int(y[te].sum()),
                                 "train_positives": int(y[tr].sum()), "train_n": int(tr.sum())}
        if len(np.unique(y[tr])) < 2:
            entry["skipped"] = "training rows have one class"
            results["families"][fam] = entry
            continue
        for m in methods:
            seeds = random_seeds if m == "random_lr" else (None,)
            per_seed = []
            for sd in seeds:
                c = {**ctx, "random_seed": sd}
                if texts is not None:
                    c["texts_tr"] = [texts[i] for i in np.flatnonzero(tr)]
                    c["texts_te"] = [texts[i] for i in np.flatnonzero(te)]
                s_te, info, oof = _fit_score(m, X[tr] if X is not None else None, y[tr], groups[tr],
                                             X[te] if X is not None else None, c)
                thr = threshold_at_fpr(y[tr][~np.isnan(oof)], oof[~np.isnan(oof)], fpr)
                yt = y[te]
                det = {}
                if thr is not None:
                    pred = s_te >= thr
                    det = {"tpr": float(pred[yt == 1].mean()) if (yt == 1).any() else None,
                           "fpr_actual": float(pred[yt == 0].mean()) if (yt == 0).any() else None}
                per_seed.append({"auroc": _auc(yt, s_te), "ap": _ap(yt, s_te), **det, "info": info,
                                 "scores": s_te})
            aucs = [p["auroc"] for p in per_seed if p["auroc"] is not None]
            entry[m] = {"auroc": float(np.mean(aucs)) if aucs else None,
                        "auroc_sd_over_seeds": float(np.std(aucs)) if len(aucs) > 1 else None,
                        "ap": per_seed[0]["ap"], "tpr_at_fpr": per_seed[0].get("tpr"),
                        "fpr_actual": per_seed[0].get("fpr_actual"), "info": per_seed[0]["info"]}
            # pooled: z-score within the held-out family so folds are comparable
            s0 = per_seed[0]["scores"]
            pooled[m][te] = (s0 - s0.mean()) / (s0.std() + 1e-12)
        results["families"][fam] = entry
    for m in methods:
        ok = ~np.isnan(pooled[m])
        a = _auc(y[ok], pooled[m][ok])
        ci = cluster_bootstrap_auc(y[ok], pooled[m][ok], groups[ok]) if a is not None else None
        fam_aucs = [results["families"][f][m]["auroc"] for f in fams
                    if m in results["families"][f] and results["families"][f][m]["auroc"] is not None]
        results["pooled"][m] = {"pooled_auroc": a, "pooled_ci95": ci,
                                "mean_family_auroc": float(np.mean(fam_aucs)) if fam_aucs else None,
                                "n_families_scored": len(fam_aucs)}
    return results
