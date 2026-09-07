# -*- coding: utf-8 -*-
"""Figures for Learner 4: three strategies, three scoreboards."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bias import predicted_bias
from .experiment4 import (
    CONDITION_META,
    CONDITIONS,
    FAILURE_COSTS,
    SLOW_CRITERION,
    TEST_CRITERION,
    XI_STIFF,
    effective_ttc,
)
from .plant import XI_SOFT

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _mean(logs, attr, fn=None):
    stack = np.stack([getattr(lg, attr) for lg in logs])
    if fn is not None:
        stack = fn(stack)
    return np.mean(stack, axis=0)


def _order(logs):
    return [c for c in CONDITIONS if c in logs]


def plot_learner4(logs, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    n = logs[next(iter(logs))][0].bias_d.size
    t = np.arange(1, n + 1)
    order = _order(logs)

    # -- row 1: the three scoreboards ------------------------------------
    for c in order:
        meta = CONDITION_META[c]
        b = _mean(logs[c], "bias_d", np.abs)
        ax1.semilogy(t, np.maximum(b, 1e-4), color=meta["color"], lw=1.3, label=meta["label"])
    ax1.axhline(predicted_bias(XI_SOFT), color="#C5761A", ls=":", lw=0.9)
    ax1.axhline(predicted_bias(XI_STIFF), color="#1B365D", ls=":", lw=0.9)
    ax1.axhline(SLOW_CRITERION, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("trial")
    ax1.set_ylabel(r"$|\hat\tau_d - \tau_d|$  (s)")
    ax1.legend(frameon=False, fontsize=5.6, ncol=2)
    ax1.set_title("slow scoreboard: the lag a person keeps", fontsize=7.5)

    for c in order:
        meta = CONDITION_META[c]
        ax2.semilogy(t, np.maximum(_mean(logs[c], "test_err"), 1e-3),
                     color=meta["color"], lw=1.3)
    ax2.axhline(TEST_CRITERION, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("trial")
    ax2.set_ylabel("relative error on soft fast probe")
    ax2.set_title("test scoreboard: everyone tested soft,\nwhatever they trained at", fontsize=7.5)

    for c in order:
        meta = CONDITION_META[c]
        ax3.semilogy(t, np.maximum(_mean(logs[c], "err"), 1e-3), color=meta["color"], lw=1.3)
    ax3.set_xlabel("trial")
    ax3.set_ylabel("full-object impulse error")
    ax3.set_title("full scoreboard: the one S3/K3\nwere failed on", fontsize=7.5)

    # -- row 2: what each strategy actually did --------------------------
    for c in order:
        meta = CONDITION_META[c]
        ax4.plot(t, _mean(logs[c], "xi"), color=meta["color"], lw=1.3)
    ax4.set_xlabel("trial")
    ax4.set_ylabel(r"co-contraction $\xi$")
    ax4.set_title("only stiffness is staged;\nmovement speed is fixed", fontsize=7.5)

    for c in order:
        meta = CONDITION_META[c]
        ax5.plot(t, _mean(logs[c], "n_eff"), color=meta["color"], lw=1.3)
    ax5.axhline(1.0, color="k", ls=":", lw=0.8)
    ax5.set_xlabel("trial")
    ax5.set_ylabel(r"effective order $n_{\mathrm{eff}}$")
    ax5.set_title("order recruitment follows the release\nof co-contraction", fontsize=7.5)

    x = np.arange(len(order))
    ttc_slow = [np.median([lg.ttc_slow for lg in logs[c]]) for c in order]
    ttc_test = [np.median([lg.ttc_test for lg in logs[c]]) for c in order]
    fails = [np.mean([lg.n_failed for lg in logs[c]]) for c in order]
    ax6.bar(x - 0.26, np.minimum(ttc_slow, n), width=0.25,
            color=[CONDITION_META[c]["color"] for c in order], alpha=0.95)
    ax6.bar(x, np.minimum(ttc_test, n), width=0.25,
            color=[CONDITION_META[c]["color"] for c in order], alpha=0.55)
    ax6b = ax6.twinx()
    ax6b.bar(x + 0.26, fails, width=0.25, color="grey", alpha=0.5)
    ax6b.spines["top"].set_visible(False)
    ax6.set_xticks(x)
    ax6.set_xticklabels(order)
    ax6.set_ylabel("median trials (solid: slow, faded: test)")
    ax6b.set_ylabel("failed trials (grey)")
    ax6.set_title("trials to each criterion, and what\nwas paid in lost attempts", fontsize=7.5)

    fig.tight_layout()
    _save(fig, path)


def plot_failure_cost(logs, path, key: str = "ttc_test") -> None:
    """When a failed attempt costs something, who actually gets there first?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    costs = np.array(FAILURE_COSTS)
    order = [c for c in ("L1", "L5", "L6", "L2") if c in logs]

    for c in order:
        meta = CONDITION_META[c]
        med = [np.median([effective_ttc(lg, key, k) for lg in logs[c]]) for k in costs]
        ax1.plot(costs, med, "o-", color=meta["color"], lw=1.4, ms=3.4, label=meta["label"])
    ax1.set_xlabel("recovery cost of one failed attempt (trials)")
    ax1.set_ylabel("effective trials to soft-probe criterion")
    ax1.legend(frameon=False, fontsize=6)
    ax1.set_title("the deep end wins only while\nfailure is free", fontsize=7.5)

    for c in order:
        meta = CONDITION_META[c]
        fails = [lg.n_failed for lg in logs[c]]
        ax2.bar(order.index(c), np.mean(fails), width=0.6, color=meta["color"], alpha=0.9)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order)
    ax2.set_ylabel("failed attempts in 60 trials")
    ax2.set_title("what the deep end is spending", fontsize=7.5)

    fig.tight_layout()
    _save(fig, path)


# Proposal mapping (URF v8): only the four strategies named in the text.
# L1→S1, L2→S2, L6→S3 (warm-start hand-over), L3→S4 (soft dominant-only control).
PROPOSAL_STRATEGIES = (
    ("S1", "L1", "#8C1428", "S1 low co-contraction, full model"),
    ("S2", "L2", "#1B365D", "S2 high co-contraction, dominant only"),
    ("S3", "L6", "#6B2D8C", "S3 co-contract, then hand over"),
    ("S4", "L3", "#C5761A", "S4 low co-contraction, dominant only"),
)


def plot_learner4_proposal(logs, path) -> None:
    """Slim 1×3 figure for the URF proposal: S1–S4 only."""
    present = [(s, L, col, lab) for s, L, col, lab in PROPOSAL_STRATEGIES if L in logs]
    if not present:
        raise ValueError("no proposal strategies found in logs")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.2, 2.55))
    n = logs[present[0][1]][0].bias_d.size
    t = np.arange(1, n + 1)

    for s, L, col, lab in present:
        b = _mean(logs[L], "bias_d", np.abs)
        ax1.semilogy(t, np.maximum(b, 1e-4), color=col, lw=1.4, label=s)
    ax1.axhline(SLOW_CRITERION, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("trial")
    ax1.set_ylabel(r"$|\hat\tau_d - \tau_d|$  (s)")
    ax1.set_title("Dominant-mode error", fontsize=8)
    ax1.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    for s, L, col, lab in present:
        ax2.semilogy(
            t,
            np.maximum(_mean(logs[L], "test_err"), 1e-3),
            color=col,
            lw=1.4,
            label=s,
        )
    ax2.axhline(TEST_CRITERION, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("trial")
    ax2.set_ylabel("relative error on soft probe")
    ax2.set_title("Soft-probe skill", fontsize=8)

    x = np.arange(len(present))
    ttc_slow = [np.median([lg.ttc_slow for lg in logs[L]]) for _, L, _, _ in present]
    fails = [np.mean([lg.n_failed for lg in logs[L]]) for _, L, _, _ in present]
    colors = [col for _, _, col, _ in present]
    ax3.bar(x - 0.18, np.minimum(ttc_slow, n), width=0.35, color=colors, alpha=0.95, label="trials to dominant")
    ax3b = ax3.twinx()
    ax3b.bar(x + 0.18, fails, width=0.35, color="grey", alpha=0.45, label="failed attempts")
    ax3b.spines["top"].set_visible(False)
    ax3.set_xticks(x)
    ax3.set_xticklabels([s for s, _, _, _ in present])
    ax3.set_ylabel("median trials to dominant mode")
    ax3b.set_ylabel("mean failed attempts")
    ax3.set_title("Speed and safety", fontsize=8)
    # one combined legend
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper right")

    fig.tight_layout(w_pad=1.2)
    _save(fig, path)


def write_snippet(logs, summary, path) -> None:
    s = summary
    n = s["n_trials"]

    def ttc(key, c):
        v = s[key][c]
        return "never" if v > n else f"{v:.0f}"

    lines = [
        "% auto-generated by scripts/run_learner4.py -- do not edit by hand",
        f"Three strategies were compared at a fixed movement speed, so that the "
        f"only thing varying is co-contraction: learn the full third-order plant "
        f"without co-contracting (L1); co-contract and learn only the slow lag, "
        f"treating the transients as noise (L2); and co-contract, then relax "
        f"progressively and learn the whole model (L5, and L6 which warm-starts a "
        f"full-order learner from the slow lag). On the slow-mode criterion "
        f"$|\\hat\\tau_d-\\tau_d|<{s['slow_criterion']:.2f}$\\,s the co-contracting "
        f"learners arrive in a median of {ttc('ttc_slow_median', 'L2')} (L2), "
        f"{ttc('ttc_slow_median', 'L5')} (L5) and {ttc('ttc_slow_median', 'L6')} "
        f"(L6) trials against {ttc('ttc_slow_median', 'L1')} for the soft learner, "
        f"and they lose {s['n_failed']['L2']:.1f}, {s['n_failed']['L5']:.1f} and "
        f"{s['n_failed']['L6']:.1f} trials to failure against "
        f"{s['n_failed']['L1']:.1f}. Tested on a common soft, fast probe --- every "
        f"condition asked to predict the same unbraced reach --- L5 reaches "
        f"relative error {s['test_err_final']['L5']:.3f} and L1 "
        f"{s['test_err_final']['L1']:.3f}, while the stiff slow-only learner "
        f"plateaus at {s['test_err_final']['L2']:.3f}: co-contraction buys a "
        f"correct slow model quickly, but only relaxation buys the transients. "
        f"That the filter rather than the reduced parameterisation is responsible "
        f"is shown by L4, a full-order learner held stiff, which ties with L2 on "
        f"the slow criterion ({ttc('ttc_slow_median', 'L4')} trials), and by L3, a "
        f"first-order learner held soft, which never reaches it and plateaus at "
        f"{s['bias_final']['L3']:+.3f}\\,s of bias.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")
