# -*- coding: utf-8 -*-
"""Pairwise Mann-Whitney tests with Holm correction on trials-to-criterion."""
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
    n_trials = logs[next(iter(logs))][0].rmse_true.size
    mean_curve = {
        c: np.mean(np.stack([lg.rmse_true for lg in logs[c]]), axis=0) for c in logs
    }
    se_curve = {
        c: np.std(np.stack([lg.rmse_true for lg in logs[c]]), axis=0, ddof=1)
        / np.sqrt(len(logs[c]))
        for c in logs
    }
    cond_curve = {
        c: np.median(np.stack([lg.cond_phi for lg in logs[c]]), axis=0) for c in logs
    }
    param_curve = {
        c: np.mean(np.stack([lg.param_err for lg in logs[c]]), axis=0) for c in logs
    }

    pairs = [("S2", o) for o in CONDITIONS if o != "S2"]
    p_raw = []
    pair_names = []
    for a, b in pairs:
        res = mannwhitneyu(ttc[a], ttc[b], alternative="less")
        p_raw.append(float(res.pvalue))
        pair_names.append(f"{a}<{b}")
    p_adj = holm(np.array(p_raw))

    tests = []
    for name, p, padj in zip(pair_names, p_raw, p_adj):
        tests.append(
            {
                "contrast": name,
                "p": float(p),
                "p_holm": float(padj),
                "stars": stars(float(padj)),
            }
        )

    return {
        "n_trials": int(n_trials),
        "n_seeds": {c: len(logs[c]) for c in logs},
        "ttc_mean": {c: float(np.mean(ttc[c])) for c in logs},
        "ttc_median": {c: float(np.median(ttc[c])) for c in logs},
        "ttc_se": {
            c: float(np.std(ttc[c], ddof=1) / np.sqrt(len(ttc[c]))) for c in logs
        },
        "reached_frac": {c: float(np.mean(ttc[c] <= n_trials)) for c in logs},
        "asymp_rmse": {
            c: float(np.mean([lg.rmse_true[-10:].mean() for lg in logs[c]])) for c in logs
        },
        "tests": tests,
        "mean_curve": {c: mean_curve[c].tolist() for c in logs},
        "se_curve": {c: se_curve[c].tolist() for c in logs},
        "cond_curve": {c: cond_curve[c].tolist() for c in logs},
        "param_curve": {c: param_curve[c].tolist() for c in logs},
    }
