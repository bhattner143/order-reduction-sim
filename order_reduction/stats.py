# -*- coding: utf-8 -*-
"""Summary statistics for S1-S5.

Tests are two-sided. The pre-registered prediction was that S2 would need
fewer trials than the others, but a one-sided test would hide the outcome that
actually occurred, which is that some comparisons run the other way.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu

from .experiment import CONDITIONS


def holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[idx]))
        adj[idx] = running
    return adj


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def summarise(logs: dict) -> dict:
    ttc = {c: np.array([lg.ttc for lg in logs[c]], dtype=float) for c in logs}
    n_trials = logs[next(iter(logs))][0].err.size

    def curve(attr, fn=np.mean):
        return {c: fn(np.stack([getattr(lg, attr) for lg in logs[c]]), axis=0) for c in logs}

    mean_curve = curve("err")
    se_curve = {
        c: np.std(np.stack([lg.err for lg in logs[c]]), axis=0, ddof=1) / np.sqrt(len(logs[c]))
        for c in logs
    }
    fail_curve = {
        c: np.cumsum(np.mean(np.stack([lg.failed for lg in logs[c]]), axis=0)) for c in logs
    }
    with np.errstate(invalid="ignore"):
        kappa_curve = {
            c: np.nanmedian(np.stack([lg.jac_cond for lg in logs[c]]), axis=0) for c in logs
        }

    tests = []
    p_raw, names, deltas = [], [], []
    for other in [c for c in CONDITIONS if c != "S2" and c in logs]:
        res = mannwhitneyu(ttc["S2"], ttc[other], alternative="two-sided")
        p_raw.append(float(res.pvalue))
        names.append(f"S2 vs {other}")
        deltas.append(float(np.median(ttc["S2"]) - np.median(ttc[other])))
    for name, p, padj, d in zip(names, p_raw, holm(np.array(p_raw)), deltas):
        tests.append(
            {
                "contrast": name,
                "median_diff": d,
                "direction": "S2 faster" if d < 0 else ("tie" if d == 0 else "S2 slower"),
                "p": p,
                "p_holm": float(padj),
                "stars": stars(float(padj)),
            }
        )

    return {
        "n_trials": int(n_trials),
        "n_seeds": {c: len(logs[c]) for c in logs},
        "ttc_mean": {c: float(np.mean(ttc[c])) for c in logs},
        "ttc_median": {c: float(np.median(ttc[c])) for c in logs},
        "ttc_se": {c: float(np.std(ttc[c], ddof=1) / np.sqrt(len(ttc[c]))) for c in logs},
        "reached_frac": {c: float(np.mean(ttc[c] <= n_trials)) for c in logs},
        "asymp_err": {c: float(np.mean([lg.err[-10:].mean() for lg in logs[c]])) for c in logs},
        "n_failed": {c: float(np.mean([lg.n_failed for lg in logs[c]])) for c in logs},
        "tests": tests,
        "mean_curve": {c: mean_curve[c].tolist() for c in logs},
        "se_curve": {c: se_curve[c].tolist() for c in logs},
        "fail_curve": {c: fail_curve[c].tolist() for c in logs},
        "kappa_curve": {c: kappa_curve[c].tolist() for c in logs},
    }
