#!/usr/bin/env python3
"""Knowing-transgression probe: local analysis of the pod readouts (CPU, < 2 GB).

    python -m transgression.analyze --model 120b      # GPT-OSS-120B: stories, LRU cheat turn, norm control
    python -m transgression.analyze --model 20b       # GPT-OSS-20B: stories + norm control (off-policy replay)

Inputs (HF dataset timf34/emotion-ensemble-forensics, downloaded under outputs/hf/ with --download):
  context_rot/transgression_gptoss{120b,20b}/   stories.npz, transcripts/<env>/<cond>/run-N.npz  (pod job)
  context_rot/gptoss120b_lru_main3/             per-turn pre / reason readouts of the 50 original LRU runs
  context_rot/goodfire_gptoss{120b,20b}/        Goodfire reward-hacking readouts (direction = hack - honest DoM)
  HF timf34/emotion-vectors gptoss_{120b,20b}/vectors_emotions_dn.pt   (outputs/hf_vectors/)

1 stories   per block, fitted on train settings, read on the act span (identical action tokens within a family):
            dom       mean(forbid_do) - mean(allow_do)
            dom_perp  dom with the "rule in context" direction mean(forbid_decline) - mean(allow_decline) projected out
            logreg    logistic regression forbid_do vs allow_do (standardised, L2)
            rule_text DoM forbid_* - allow_* on the norm span (reads the rule sentence), applied to the act span
            baselines guilty / ashamed / regretful / sorry emotion vectors, Goodfire direction, 100 random directions
            Layer per probe by 5-fold grouped CV (by setting) inside train. Test settings: AUROC forbid_do vs allow_do,
            matched-pair accuracy, 2x2 interaction, unaware_do, leave-one-norm-type-out.
2 lru       the probe (no refit) on the 44 cheating LRU runs: event-locked trajectory (tau -8..+4); detection of the
            cheat-writing turn (tau = 0) vs earlier turns (pooled AUROC, within-run rank, tau 0 vs -1).
3 control   replayed runs with one sentence of the task prompt flipped (allow vs forbid; palisades also none).
            Delta(turn) = s(forbid) - s(allow) in probe-SD units; per run DiD = Delta(tau 0) - mean Delta(tau <= -2);
            bootstrap over runs; null = the same statistic for random directions. pre / reason / action.
Writes outputs/transgression/<model>/results.json and docs/figures/tg_<model>_*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "timf34/emotion-ensemble-forensics"
GUILT = ("guilty", "ashamed", "regretful", "sorry")
EXTRA_EMO = ("nervous", "calm", "desperate")
PROBES = ("dom", "dom_perp", "logreg", "rule_text")
N_RAND = 100
REPS = ("pre", "reason", "action")


# ------------------------------------------------------------------------------------------------ utils
def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def boot_auroc(y, s, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    by = {u: np.flatnonzero(g == u) for u in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[u] for u in rng.choice(ug, len(ug))])
        if len(np.unique(y[idx])) == 2:
            vals.append(auroc(y[idx], s[idx]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def r3(x):
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 3)
    if isinstance(x, np.integer):
        return int(x)
    return x


def load_emotions(vec_dir: Path):
    import torch

    emo = torch.load(vec_dir / "vectors_emotions_dn.pt", map_location="cpu", weights_only=True)
    return {name: {b: unit(t.float().numpy()) for b, t in layers.items()} for name, layers in emo.items()}


def load_goodfire(gf_dir: Path, blocks):
    """Goodfire reward-hacking direction per block: mean over hack steps - mean over honest steps (step readout)."""
    files = sorted(gf_dir.glob("*.npz"))
    if not files:
        return {}
    sums = {b: [0.0, 0.0] for b in blocks}
    cnt = [0, 0]
    for f in files:
        z = np.load(f, allow_pickle=False)
        jb = {int(b): j for j, b in enumerate(z["blocks"])}
        steps = json.loads(str(z["steps"]))
        A = z["step"].astype(np.float64)
        for k, s in enumerate(steps):
            if s["label"] not in ("hack", "honest"):
                continue
            c = int(s["label"] == "hack")
            cnt[c] += 1
            for b in blocks:
                if b in jb:
                    sums[b][c] = sums[b][c] + A[k, jb[b]]
    return {b: unit(sums[b][1] / cnt[1] - sums[b][0] / cnt[0]) for b in blocks if not np.isscalar(sums[b][1])}


# ---------------------------------------------------------------------------------------------- stories
class Stories:
    def __init__(self, tg: Path):
        z = np.load(tg / "stories.npz", allow_pickle=False)
        self.blocks = [int(b) for b in z["blocks"]]
        ids = [str(i) for i in z["ids"]]
        rows = {r["id"]: r for r in map(json.loads, open(tg / "texts.jsonl" if (tg / "texts.jsonl").exists()
                                                        else ROOT / "data/transgression/texts.jsonl"))}
        self.rows = [rows[i] for i in ids]
        self.act, self.norm = z["act"], z["norm"]          # [N, B, H] fp16
        self.cond = np.array([r["condition"] for r in self.rows])
        self.fam = np.array([r["family"] for r in self.rows])
        self.kind = np.array([r["kind"] for r in self.rows])
        self.setting = np.array([r["setting"] for r in self.rows])
        self.test = np.array([r["split"] == "test" for r in self.rows])
        meta = tg / "stories_meta.jsonl"
        self.act_hash_ok = None
        if meta.exists():                    # act tokens identical within a family (do and decline separately)
            h = {m["id"]: m["act_hash"] for m in map(json.loads, open(meta))}
            groups = {}
            for i, f, c in zip(ids, self.fam, self.cond):
                groups.setdefault((f, c.endswith("_do")), set()).add(h[i])
            self.act_hash_ok = float(np.mean([len(v) == 1 for v in groups.values()]))

    def X(self, b, span="act"):
        return getattr(self, span)[:, self.blocks.index(b)].astype(np.float64)


def fit_probes(S: Stories, b: int, m: np.ndarray) -> dict[str, np.ndarray]:
    """All story probes at block b, fitted on rows in mask m. Returned as unit directions in residual space."""
    from sklearn.linear_model import LogisticRegression

    X, N = S.X(b), S.X(b, "norm")
    c = S.cond
    mu = lambda k: X[m & (c == k)].mean(0)
    d = mu("forbid_do") - mu("allow_do")
    r = unit(mu("forbid_decline") - mu("allow_decline"))
    out = {"dom": unit(d), "dom_perp": unit(d - (d @ r) * r)}
    sel = m & np.isin(c, ["forbid_do", "allow_do"])
    sd = X[sel].std(0) + 1e-6
    lr = LogisticRegression(C=0.01, max_iter=3000).fit((X[sel] - X[sel].mean(0)) / sd, c[sel] == "forbid_do")
    out["logreg"] = unit(lr.coef_[0] / sd)                  # back to residual space (a linear readout)
    fb = m & np.char.startswith(c.astype(str), "forbid")
    al = m & np.char.startswith(c.astype(str), "allow")
    out["rule_text"] = unit(N[fb].mean(0) - N[al].mean(0))
    return out


def pair_scores(S: Stories, s: np.ndarray, m: np.ndarray, a="forbid_do", b="allow_do"):
    """Per family in m: (score of a, score of b)."""
    idx = {(f, c): i for i, (f, c) in enumerate(zip(S.fam, S.cond)) if m[i]}
    fams = sorted({f for f, _ in idx if (f, a) in idx and (f, b) in idx})
    return np.array([[s[idx[f, a]], s[idx[f, b]]] for f in fams]), fams


def story_metrics(S: Stories, s: np.ndarray, m: np.ndarray) -> dict:
    c = S.cond
    sel = m & np.isin(c, ["forbid_do", "allow_do"])
    out = {"auroc": auroc(c[sel] == "forbid_do", s[sel])}
    P, _ = pair_scores(S, s, m)
    out["pair_acc"] = float((P[:, 0] > P[:, 1]).mean())
    sd = s[m & np.isin(c, ["forbid_do", "allow_do", "forbid_decline", "allow_decline"])].std() + 1e-12
    mu0 = s[m & (c == "allow_do")].mean()
    out["cond_z"] = {k: float((s[m & (c == k)].mean() - mu0) / sd) for k in
                     ("forbid_do", "allow_do", "unaware_do", "forbid_decline", "allow_decline")}
    # 2x2 interaction per family, in SD units
    idx = {(f, k): i for i, (f, k) in enumerate(zip(S.fam, c)) if m[i]}
    keys = ("forbid_do", "allow_do", "forbid_decline", "allow_decline")
    fams = [f for f in np.unique(S.fam[m]) if all((f, k) in idx for k in keys)]
    inter = np.array([(s[idx[f, keys[0]]] - s[idx[f, keys[1]]]) - (s[idx[f, keys[2]]] - s[idx[f, keys[3]]])
                      for f in fams]) / sd
    out["interaction"] = float(inter.mean())
    out["interaction_ci"] = boot_mean(inter)
    Pu, _ = pair_scores(S, s, m, "unaware_do", "allow_do")
    Pf, _ = pair_scores(S, s, m, "forbid_do", "unaware_do")
    out["unaware_vs_allow_pair"] = float((Pu[:, 0] > Pu[:, 1]).mean())
    out["forbid_vs_unaware_pair"] = float((Pf[:, 0] > Pf[:, 1]).mean())
    selu = m & np.isin(c, ["unaware_do", "allow_do"])
    out["auroc_unaware_vs_allow"] = auroc(c[selu] == "unaware_do", s[selu])
    return out


def cv_auroc(S: Stories, b: int, n_folds=5) -> dict[str, float]:
    from sklearn.model_selection import GroupKFold

    tr = np.flatnonzero(~S.test)
    res = {p: [] for p in PROBES}
    for f_tr, f_te in GroupKFold(n_folds).split(tr, groups=S.setting[tr]):
        mtr = np.zeros(len(S.cond), bool); mtr[tr[f_tr]] = True
        mte = np.zeros(len(S.cond), bool); mte[tr[f_te]] = True
        D = fit_probes(S, b, mtr)
        X = S.X(b)
        sel = mte & np.isin(S.cond, ["forbid_do", "allow_do"])
        for p, d in D.items():
            res[p].append(auroc(S.cond[sel] == "forbid_do", X[sel] @ d))
    return {p: float(np.mean(v)) for p, v in res.items()}


def run_stories(S: Stories, emo, gf, rng) -> tuple[dict, dict]:
    res = {"blocks": S.blocks, "act_hash_consistent": S.act_hash_ok, "per_block": {}, "cv": {}}
    test, train = S.test, ~S.test
    H = S.act.shape[-1]
    R = unit(rng.standard_normal((N_RAND, H)))
    dirs = {}
    for b in S.blocks:
        X = S.X(b)
        D = fit_probes(S, b, train)
        dirs[b] = D
        res["cv"][b] = cv_auroc(S, b)
        pb = {p: story_metrics(S, X @ d, test) for p, d in D.items()}
        for e in GUILT + EXTRA_EMO:
            if e in emo:
                pb[f"emo:{e}"] = story_metrics(S, X @ emo[e][b], test)
        if b in gf:
            pb["goodfire"] = story_metrics(S, X @ gf[b], test)
        sel = test & np.isin(S.cond, ["forbid_do", "allow_do"])
        ra = np.array([auroc(S.cond[sel] == "forbid_do", X[sel] @ r) for r in R])
        pb["random"] = {"auroc_mean": float(ra.mean()), "auroc_abs_p95": float(np.quantile(np.maximum(ra, 1 - ra), .95)),
                        "auroc_abs_max": float(np.maximum(ra, 1 - ra).max())}
        res["per_block"][b] = pb
    # layer per probe, chosen on train CV only
    res["chosen_block"] = {p: max(S.blocks, key=lambda b: res["cv"][b][p]) for p in PROBES}
    bstar = res["chosen_block"]["dom"]
    res["main"] = {p: {"block": res["chosen_block"][p], **res["per_block"][res["chosen_block"][p]][p]} for p in PROBES}
    res["main_baselines_at_dom_block"] = {k: v for k, v in res["per_block"][bstar].items() if k not in PROBES}
    # leave-one-norm-type-out: fit on train settings without kind k, test on test-setting families of kind k
    lono = {}
    for k in np.unique(S.kind):
        D = fit_probes(S, bstar, train & (S.kind != k))
        X = S.X(bstar)
        m = test & (S.kind == k)
        lono[k] = {p: {kk: story_metrics(S, X @ D[p], m)[kk] for kk in ("auroc", "pair_acc", "interaction")}
                   for p in ("dom", "dom_perp")}
    res["leave_one_kind_out"] = lono
    # per kind on the main probe (no refit)
    X = S.X(bstar)
    res["per_kind_dom"] = {k: story_metrics(S, X @ dirs[bstar]["dom"], test & (S.kind == k)) for k in np.unique(S.kind)}
    cosines = {p: float(dirs[bstar][p] @ dirs[bstar]["dom"]) for p in PROBES}
    for e in GUILT:
        if e in emo:
            cosines[f"emo:{e}"] = float(emo[e][bstar] @ dirs[bstar]["dom"])
    if bstar in gf:
        cosines["goodfire"] = float(gf[bstar] @ dirs[bstar]["dom"])
    res["cosine_with_dom"] = cosines
    return res, dirs


# ----------------------------------------------------------------------------------- directions to test
def direction_set(dirs, emo, gf, b, H, rng):
    """name -> unit direction at block b. Random directions fixed by rng."""
    out = {p: dirs[b][p] for p in PROBES}
    for e in GUILT + EXTRA_EMO:
        if e in emo:
            out[f"emo:{e}"] = emo[e][b]
    if b in gf:
        out["goodfire"] = gf[b]
    R = unit(rng.standard_normal((N_RAND, H)))
    return out, R


# ----------------------------------------------------------------------------------------- LRU cheat turn
def load_lru_runs(acts: Path, labels: list[dict], b: int, rep: str):
    lab = {l["run"]: l for l in labels}
    out = []
    for f in sorted(acts.glob("run-*.npz"), key=lambda p: int(p.stem.split("-")[1])):
        if f.stem not in lab or not lab[f.stem].get("cheated"):
            continue
        z = np.load(f, allow_pickle=False)
        j = [int(x) for x in z["blocks"]].index(b)
        X = z[rep][:, j, :].astype(np.float64)
        if f"{rep}_tokens" in z.files:
            X[z[f"{rep}_tokens"] == 0] = np.nan
        out.append({"run": f.stem, "X": X, "onset": lab[f.stem]["onset"]})
    return out


def detection(runs, d):
    y, s, g, within, adj = [], [], [], [], []
    for r in runs:
        sc = r["X"] @ d
        k0 = r["onset"]
        if k0 >= r["X"].shape[0]:
            continue
        for k in range(k0 + 1):
            if np.isfinite(sc[k]):
                y.append(int(k == k0)); s.append(sc[k]); g.append(r["run"])
        pre = sc[:k0][np.isfinite(sc[:k0])]
        if len(pre) and np.isfinite(sc[k0]):
            within.append(float((sc[k0] > pre).mean()))
            if np.isfinite(sc[k0 - 1]):
                adj.append(float(sc[k0] > sc[k0 - 1]))
    y, s, g = np.array(y), np.array(s), np.array(g)
    return {"auroc": auroc(y, s), "auroc_ci": boot_auroc(y, s, g), "within_rank": float(np.mean(within)),
            "tau0_gt_tau-1": float(np.mean(adj)), "n_runs": len(set(g))}


def event_locked(runs, d, taus=range(-8, 5)):
    allsc = np.concatenate([r["X"] @ d for r in runs])
    mu, sd = np.nanmean(allsc), np.nanstd(allsc) + 1e-12
    out = {}
    for t in taus:
        v = [((r["X"] @ d)[r["onset"] + t] - mu) / sd for r in runs if 0 <= r["onset"] + t < r["X"].shape[0]]
        v = np.array([x for x in v if np.isfinite(x)])
        out[t] = [float(v.mean()) if len(v) else float("nan"), *boot_mean(v)] if len(v) > 1 else [float("nan")] * 3
    return out


def run_lru(acts: Path, labels, dirs, emo, gf, blocks_main: dict, rng) -> dict:
    res = {}
    for rep in ("reason", "pre"):
        res[rep] = {}
        b = blocks_main["dom"]
        runs = load_lru_runs(acts, labels, b, rep)
        D, R = direction_set(dirs, emo, gf, b, runs[0]["X"].shape[1], rng)
        det = {n: detection(runs, d) for n, d in D.items()}
        rd = [detection(runs, r) for r in R]
        ra = np.array([x["auroc"] for x in rd])
        det["random"] = {"auroc_mean": float(ra.mean()), "auroc_p95": float(np.quantile(ra, .95)),
                         "auroc_abs_p95": float(np.quantile(np.maximum(ra, 1 - ra), .95)),
                         "within_rank_mean": float(np.mean([x["within_rank"] for x in rd]))}
        # turn index alone (drift baseline)
        turn = [{"run": r["run"], "X": np.arange(r["X"].shape[0], dtype=float)[:, None], "onset": r["onset"]} for r in runs]
        det["turn_index"] = detection(turn, np.ones(1))
        # all 42 emotions at this block (best single emotion, for context)
        emo_auc = {e: detection(runs, emo[e][b])["auroc"] for e in emo}
        det["best_emotions"] = dict(sorted(emo_auc.items(), key=lambda kv: -kv[1])[:5])
        det["worst_emotions"] = dict(sorted(emo_auc.items(), key=lambda kv: kv[1])[:5])
        res[rep] = {"block": b, "n_cheat_runs": len(runs), "detection": det,
                    "event_locked": {n: event_locked(runs, D[n]) for n in ("dom", "dom_perp", "logreg", "rule_text",
                                                                            "emo:guilty", "goodfire") if n in D},
                    "event_locked_random": [event_locked(runs, r) for r in R[:20]]}
        # per block detection for the main probe (layer sweep, reported not selected)
        sweep = {}
        for bb in sorted(dirs):
            rr = load_lru_runs(acts, labels, bb, rep)
            sweep[bb] = {p: detection(rr, dirs[bb][p])["auroc"] for p in ("dom", "dom_perp")}
        res[rep]["sweep"] = sweep
    return res


# ---------------------------------------------------------------------------------------- norm control
def load_cond(tg: Path, env: str, cond: str, run: str, b: int, rep: str):
    f = tg / "transcripts" / env / cond / f"{run}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    j = [int(x) for x in z["blocks"]].index(b)
    X = z[rep][:, j, :].astype(np.float64)
    if f"{rep}_tokens" in z.files:
        X[z[f"{rep}_tokens"] == 0] = np.nan
    return X


def did_stats(pairs, d, sd):
    """pairs: list of (run meta, X_a, X_b). Delta = (X_a - X_b) @ d / sd. DiD per run (onset >= 2)."""
    did, base, at0, traj = [], [], [], {}
    for m, A, B in pairs:
        n = min(len(A), len(B))
        delta = (A[:n] - B[:n]) @ d / sd
        k0 = m["onset"]
        for k in range(n):
            traj.setdefault(k - k0, []).append(delta[k])
        if k0 < 2 or k0 >= n or not np.isfinite(delta[k0]):
            continue
        pre = delta[:k0 - 1]
        pre = pre[np.isfinite(pre)]
        if not len(pre):
            continue
        did.append(delta[k0] - pre.mean()); base.append(pre.mean()); at0.append(delta[k0])
    did = np.array(did)
    return {"did": float(did.mean()) if len(did) else float("nan"), "did_ci": boot_mean(did),
            "delta_pre": float(np.mean(base)) if base else float("nan"), "delta_tau0": float(np.mean(at0)) if at0 else float("nan"),
            "n_runs": len(did),
            "traj": {t: float(np.nanmean(v)) for t, v in sorted(traj.items()) if -8 <= t <= 2 and len(v) >= 3}}, did


def run_control(tg: Path, meta, dirs, emo, gf, b: int, rng, contrasts) -> dict:
    res = {}
    for env, (ca, cb) in contrasts:
        key = f"{env}:{ca}-{cb}"
        res[key] = {}
        runs = [m for m in meta if m["env"] == env]
        for rep in REPS:
            pairs = []
            for m in runs:
                A, B = load_cond(tg, env, ca, m["run"], b, rep), load_cond(tg, env, cb, m["run"], b, rep)
                if A is not None and B is not None:
                    pairs.append((m, A, B))
            if not pairs:
                continue
            H = pairs[0][1].shape[1]
            D, R = direction_set(dirs, emo, gf, b, H, rng)
            allX = np.concatenate([np.concatenate([A, B]) for _, A, B in pairs])
            allX = allX[np.isfinite(allX).all(1)]
            out = {}
            for n, d in D.items():
                out[n], _ = did_stats(pairs, d, (allX @ d).std() + 1e-12)
            rnd = [did_stats(pairs, r, (allX @ r).std() + 1e-12)[0] for r in R]
            rd = np.array([x["did"] for x in rnd])
            rp = np.array([x["delta_pre"] for x in rnd])
            out["random"] = {"did_mean": float(np.nanmean(rd)), "did_abs_p95": float(np.nanquantile(np.abs(rd), .95)),
                             "delta_pre_abs_p95": float(np.nanquantile(np.abs(rp), .95)),
                             "traj_p5": {t: float(np.nanquantile([x["traj"].get(t, np.nan) for x in rnd], .05)) for t in out["dom"]["traj"]},
                             "traj_p95": {t: float(np.nanquantile([x["traj"].get(t, np.nan) for x in rnd], .95)) for t in out["dom"]["traj"]}}
            for n in D:
                out[n]["p_vs_random"] = float((np.abs(rd) >= abs(out[n]["did"])).mean())
            # palisades: runs that restated the rule (in their reasoning, before the cheat) vs not
            if env == "palisades":
                said = [p for p in pairs if p[0].get("states_rule_turns")]
                not_said = [p for p in pairs if not p[0].get("states_rule_turns")]
                sd = (allX @ D["dom"]).std() + 1e-12
                out["dom_by_restated"] = {"restated": did_stats(said, D["dom"], sd)[0],
                                          "not_restated": did_stats(not_said, D["dom"], sd)[0]}
                for g in out["dom_by_restated"].values():
                    g.pop("traj", None)
            res[key][rep] = out
    return res


# ---------------------------------------------------------------------------------------------- figures
def figures(res: dict, model: str, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    nn = lambda v: np.nan if v is None else v
    st = res["stories"]
    blocks = st["blocks"]
    st["per_block"] = {int(k): v for k, v in st["per_block"].items()}
    # 1. held-out AUROC per block
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pb = st["per_block"]
    for p, c in zip(PROBES, ("C0", "C1", "C2", "C7")):
        ax[0].plot(blocks, [nn(pb[b][p]["auroc"]) for b in blocks], "-o", ms=3, c=c, label=p)
    for e, c in zip(GUILT, ("C3", "C4", "C5", "C6")):
        if f"emo:{e}" in pb[blocks[0]]:
            ax[0].plot(blocks, [nn(pb[b][f"emo:{e}"]["auroc"]) for b in blocks], "--", c=c, lw=1, label=e)
    if "goodfire" in pb[blocks[0]]:
        ax[0].plot(blocks, [nn(pb[b]["goodfire"]["auroc"]) for b in blocks], ":", c="k", label="goodfire")
    ax[0].fill_between(blocks, [1 - pb[b]["random"]["auroc_abs_p95"] for b in blocks],
                       [pb[b]["random"]["auroc_abs_p95"] for b in blocks], color="0.85", label="random 5-95%")
    ax[0].axhline(.5, c="0.5", lw=.5)
    ax[0].set(xlabel="block", ylabel="AUROC forbid_do vs allow_do (test settings)", title=f"GPT-OSS-{model}: held-out stories")
    ax[0].legend(fontsize=7, ncol=2)
    b = st["main"]["dom"]["block"]
    conds = ("allow_do", "forbid_do", "unaware_do", "allow_decline", "forbid_decline")
    for i, (n, c) in enumerate((("dom", "C0"), ("dom_perp", "C1"), ("emo:guilty", "C3"))):
        if n in pb[b]:
            ax[1].bar(np.arange(5) + (i - 1) * .27, [pb[b][n]["cond_z"][k] for k in conds], .27, color=c, label=n)
    ax[1].set_xticks(range(5), conds, rotation=20, fontsize=8)
    ax[1].set(ylabel="mean score - allow_do (SD units)", title=f"conditions at block {b} (test)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / f"tg_{model}_stories.png", dpi=130); plt.close(fig)

    # 2. LRU event-locked
    if "lru" in res:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, rep in zip(ax, ("reason", "pre")):
            L = res["lru"][rep]
            for rr in L["event_locked_random"]:
                t = sorted(int(k) for k in rr)
                a.plot(t, [nn(rr[str(k)][0]) for k in t], c="0.8", lw=.6)
            for n, c in zip(L["event_locked"], ("C0", "C1", "C2", "C7", "C3", "k")):
                el = L["event_locked"][n]
                t = sorted(int(k) for k in el)
                a.plot(t, [nn(el[str(k)][0]) for k in t], "-o", ms=3, c=c, label=n)
            a.axvline(0, c="r", lw=.5)
            a.set(xlabel="turns from the cheat-writing turn (tau)", ylabel="z", title=f"LRU cheat runs, {rep}, block {L['block']}")
            a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / f"tg_{model}_lru.png", dpi=130); plt.close(fig)

    # 3. norm control: Delta by tau
    ctl = res.get("control", {})
    keys = [k for k in ctl if ctl[k]]
    if keys:
        fig, ax = plt.subplots(len(keys), 3, figsize=(13, 3.2 * len(keys)), squeeze=False)
        for i, key in enumerate(keys):
            for j, rep in enumerate(REPS):
                a = ax[i, j]
                if rep not in ctl[key]:
                    a.axis("off"); continue
                o = ctl[key][rep]
                t = sorted(int(x) for x in o["dom"]["traj"])
                a.fill_between(t, [nn(o["random"]["traj_p5"][str(x)]) for x in t], [nn(o["random"]["traj_p95"][str(x)]) for x in t],
                               color="0.85", label="random 5-95%")
                for n, c in (("dom", "C0"), ("dom_perp", "C1"), ("rule_text", "C7"), ("emo:guilty", "C3"), ("goodfire", "k")):
                    if n in o:
                        a.plot(t, [nn(o[n]["traj"].get(str(x))) for x in t], "-o", ms=3, c=c, label=n)
                a.axvline(0, c="r", lw=.5); a.axhline(0, c="0.5", lw=.5)
                a.set(title=f"{key} {rep}", xlabel="tau", ylabel="Delta (probe SD)")
                if i == 0 and j == 0:
                    a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / f"tg_{model}_control.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------------- main
def download(model: str):
    from huggingface_hub import snapshot_download

    pats = [f"context_rot/transgression_gptoss{model}/*", f"context_rot/transgression_gptoss{model}/**",
            f"context_rot/goodfire_gptoss{model}/*"]
    if model == "120b":
        pats.append("context_rot/gptoss120b_lru_main3/*")
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=pats, local_dir=ROOT / "outputs/hf")
    snapshot_download("timf34/emotion-vectors", repo_type="dataset",
                      allow_patterns=[f"gptoss_{model}/vectors_emotions_dn.pt"], local_dir=ROOT / "outputs/hf_vectors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["120b", "20b"], default="120b")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--hf", default=str(ROOT / "outputs/hf/context_rot"))
    ap.add_argument("--tg", default=None, help="pod outputs (default <hf>/transgression_gptoss<model>)")
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default=str(ROOT / "docs/figures"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["lru", "control"])
    a = ap.parse_args()
    if a.download:
        download(a.model)
    hf = Path(a.hf)
    tg = Path(a.tg or hf / f"transgression_gptoss{a.model}")
    vec = Path(a.vectors or ROOT / f"outputs/hf_vectors/gptoss_{a.model}")
    out = Path(a.out or ROOT / f"outputs/transgression/{a.model}")
    out.mkdir(parents=True, exist_ok=True)

    S = Stories(tg)
    emo = load_emotions(vec)
    gf = load_goodfire(hf / f"goodfire_gptoss{a.model}", S.blocks)
    print(f"[tg] {len(S.cond)} stories, blocks {S.blocks}, {len(emo)} emotions, goodfire blocks {len(gf)}", flush=True)
    res = {"model": a.model}
    res["stories"], dirs = run_stories(S, emo, gf, np.random.default_rng(0))
    main_b = res["stories"]["chosen_block"]
    print("[tg] stories main:", json.dumps(r3(res["stories"]["main"]["dom"]))[:400], flush=True)
    np.savez(out / "probe_dirs.npz", blocks=np.array(S.blocks),
             **{f"{p}_{b}": dirs[b][p] for b in S.blocks for p in PROBES})

    if a.model == "120b" and "lru" not in a.skip:
        labels = json.load(open(ROOT / "data/transgression/lru120b_labels_code.json"))
        res["lru"] = run_lru(hf / "gptoss120b_lru_main3", labels, dirs, emo, gf, main_b, np.random.default_rng(1))
        print("[tg] lru reason dom:", r3(res["lru"]["reason"]["detection"]["dom"]), flush=True)
    if "control" not in a.skip and (tg / "transcripts").exists():
        meta_f = tg / "meta.json" if (tg / "meta.json").exists() else ROOT / "data/transgression/norm_runs/meta.json"
        meta = json.load(open(meta_f))
        contrasts = [("lru", ("forbid", "allow")), ("palisades", ("forbid", "allow")), ("palisades", ("forbid", "none")),
                     ("palisades", ("none", "allow"))]
        res["control"] = run_control(tg, meta, dirs, emo, gf, main_b["dom"], np.random.default_rng(2), contrasts)
        for k, v in res["control"].items():
            for rep, o in v.items():
                print(f"[tg] control {k} {rep}: dom DiD {o['dom']['did']:.2f} {r3(o['dom']['did_ci'])} "
                      f"p_rand {o['dom']['p_vs_random']:.2f} | pre-offset {o['dom']['delta_pre']:.2f}", flush=True)
    res = r3(res)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    figures(json.loads(json.dumps(res)), a.model, Path(a.figdir))
    print(f"[tg] wrote {out / 'results.json'} and figures in {a.figdir}")


if __name__ == "__main__":
    main()
