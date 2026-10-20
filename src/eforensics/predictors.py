"""Predictors compared on the same training rows, labels and prediction point.

Every method is `fit(X, y, groups, ctx) -> score(X_new)`; anything data-dependent
(standardisation, PCA, regularisation strength, which single emotion, its sign) is chosen
on the training rows only, with GROUPED inner cross-validation (groups = prompts / problems,
so seeds of one prompt never sit on both sides).

  emotion_lr     42 emotion scores -> standardise -> L2 logistic regression (the ensemble)
  syndrome_lr    the 6 syndrome scores, same classifier (ablation)
  best_emotion   the single emotion score (and sign) with the best inner-CV AUROC
  random_lr      42 random unit directions, same classifier (one seed per call)
  pca_lr         42 training-fitted principal components, same classifier
  direct_lr      the full activation -> standardise -> L2 logistic regression (behaviour probe)
  dom            difference of class means on the raw activation (behaviour probe, no fit)
  text_lr        TF-IDF of the prompt text -> L2 logistic regression (wording-only baseline)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _auc(y, s) -> float | None:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def inner_folds(groups: np.ndarray, y: np.ndarray, k: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    ug = np.unique(groups)
    k = min(k, len(ug))
    if k < 2:
        return []
    return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def _lr(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def fit_lr(F: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[Callable, dict[str, Any]]:
    """L2 logistic regression with C chosen by grouped inner-CV AUROC."""
    folds = inner_folds(groups, y)
    best, best_auc = C_GRID[len(C_GRID) // 2], -1.0
    if folds:
        for C in C_GRID:
            aucs = []
            for tr, va in folds:
                if len(np.unique(y[tr])) < 2:
                    continue
                m = _lr(C).fit(F[tr], y[tr])
                a = _auc(y[va], m.decision_function(F[va]))
                if a is not None:
                    aucs.append(a)
            if aucs and np.mean(aucs) > best_auc:
                best, best_auc = C, float(np.mean(aucs))
    m = _lr(best).fit(F, y)
    return m.decision_function, {"C": best, "inner_auc": best_auc if best_auc >= 0 else None}


def method_emotion_lr(X, y, g, ctx):
    U = ctx["U_emotion"]
    f, info = fit_lr(X @ U.T, y, g)
    return (lambda Z: f(Z @ U.T)), info


def method_syndrome_lr(X, y, g, ctx):
    U = ctx["U_syndrome"]
    f, info = fit_lr(X @ U.T, y, g)
    return (lambda Z: f(Z @ U.T)), info


def method_best_emotion(X, y, g, ctx):
    U, labels = ctx["U_emotion"], ctx["emotion_labels"]
    S = X @ U.T
    folds = inner_folds(g, y)
    score = np.zeros(S.shape[1])
    for k in range(S.shape[1]):
        vals = [a for tr, va in folds
                if (a := _auc(y[va], S[va, k])) is not None] if folds else [_auc(y, S[:, k]) or 0.5]
        # sign is part of the selection: |AUROC - 0.5| on inner folds, sign from the training set
        score[k] = abs(np.mean(vals) - 0.5) if vals else 0.0
    k = int(np.argmax(score))
    sign = 1.0 if (_auc(y, S[:, k]) or 0.5) >= 0.5 else -1.0
    u = U[k]
    return (lambda Z: sign * (Z @ u)), {"emotion": labels[k], "sign": sign, "inner_abs_auc_dev": float(score[k])}


def method_random_lr(X, y, g, ctx):
    from .features import random_directions

    R = random_directions(X.shape[1], ctx.get("k", 42), ctx.get("random_seed", 0))
    f, info = fit_lr(X @ R.T, y, g)
    return (lambda Z: f(Z @ R.T)), {**info, "random_seed": ctx.get("random_seed", 0)}


def method_pca_lr(X, y, g, ctx):
    from sklearn.decomposition import PCA

    k = min(ctx.get("k", 42), X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=k, random_state=0).fit(X)
    f, info = fit_lr(p.transform(X), y, g)
    return (lambda Z: f(p.transform(Z))), {**info, "k": k}


def method_direct_lr(X, y, g, ctx):
    return fit_lr(X, y, g)


def method_dom(X, y, g, ctx):
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    return (lambda Z: Z @ d), {"norm_before_unit": float(np.linalg.norm(X[y == 1].mean(0) - X[y == 0].mean(0)))}


def method_text_lr(X, y, g, ctx):
    """X is ignored: uses ctx['texts'] aligned with the training rows; Z must be the test texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    F = vec.fit_transform(ctx["texts_train"]).toarray()
    f, info = fit_lr(F, y, g)
    return (lambda texts: f(vec.transform(texts).toarray())), info


METHODS: dict[str, Callable] = {
    "emotion_lr": method_emotion_lr, "syndrome_lr": method_syndrome_lr, "best_emotion": method_best_emotion,
    "random_lr": method_random_lr, "pca_lr": method_pca_lr, "direct_lr": method_direct_lr, "dom": method_dom,
    "text_lr": method_text_lr,
}
