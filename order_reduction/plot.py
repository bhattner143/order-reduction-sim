"""Proposal two-panel figure plus mechanism plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiment import CONDITION_META, CONDITIONS
from .plant import TRUE_POLES, experienced_poles as _poles

INK = "#1B365D"
WARM = "#8C1428"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_proposal_figure(logs: dict, summary: dict, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    n = summary["n_trials"]
    t = np.arange(1, n + 1)
    for c in CONDITIONS:
        mu = np.array(summary["mean_curve"][c])
        se = np.array(summary["se_curve"][c])
        col = CONDITION_META[c]["color"]
        ax.plot(t, mu, color=col, lw=1.8, label=CONDITION_META[c]["label"])
        ax.fill_between(t, mu - se, mu + se, color=col, alpha=0.18, lw=0)
    ax.axhline(0.12, color="0.5", ls="--", lw=0.8, label="criterion")
    ax.set_xlabel("trial $n$")
    ax.set_ylabel("relative impulse error vs true $G$")
    ax.set_xlim(1, n)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("A  Learning curves", loc="left", color=INK, fontsize=10)

    ax = axes[1]
    means = [summary["ttc_mean"][c] for c in CONDITIONS]
    ses = [summary["ttc_se"][c] for c in CONDITIONS]
    colors = [CONDITION_META[c]["color"] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    ax.bar(x, means, yerr=ses, color=colors, ecolor="black", capsize=3, width=0.72)
    ax.set_xticks(x, CONDITIONS)
    ax.set_ylabel("trials to criterion")
    ax.set_title("B  Trials to criterion", loc="left", color=INK, fontsize=10)
    # Holm stars vs S2, drawn above each non-S2 bar.
    star_map = {t["contrast"].split("<")[1]: t["stars"] for t in summary["tests"]}
    ymax = max(m + s for m, s in zip(means, ses))
    for i, c in enumerate(CONDITIONS):
        if c == "S2":
            continue
        ax.text(i, means[i] + ses[i] + 0.04 * ymax, star_map.get(c, ""), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_mechanism(logs: dict, summary: dict, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.05))
    n = summary["n_trials"]
    t = np.arange(1, n + 1)

    ax = axes[0]
    for c in ("S1", "S2", "S4"):
        ax.semilogy(t, summary["cond_curve"][c], color=CONDITION_META[c]["color"], lw=1.8, label=c)
    ax.set_xlabel("trial $n$")
    ax.set_ylabel(r"median $\kappa(\Phi)$")
    ax.set_title("Hankel singular-value ratio", loc="left", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for c in CONDITIONS:
        ax.plot(t, summary["param_curve"][c], color=CONDITION_META[c]["color"], lw=1.6, label=c)
    ax.set_xlabel("trial $n$")
    ax.set_ylabel(r"impulse mismatch $\|\hat G - G\|$")
    ax.set_title("Model error vs true 3rd-order $G$", loc="left", color=INK, fontsize=9)

    ax = axes[2]
    # S2 pole migration: mean experienced poles by stage, plus true poles.
    s2 = logs["S2"]
    xi = np.stack([lg.xi for lg in s2], axis=0)
    mean_xi_by_trial = xi.mean(axis=0)
    poles = np.stack([_poles(x) for x in mean_xi_by_trial])
    ax.plot(t, poles[:, 0], color=INK, lw=1.8, label="dominant")
    ax.plot(t, poles[:, 1], color=WARM, lw=1.8, label="transient 1")
    ax.plot(t, poles[:, 2], color="#1E6B3A", lw=1.8, label="transient 2")
    for p, ls in zip(TRUE_POLES, (":", ":", ":")):
        ax.axhline(p, color="0.6", ls=ls, lw=0.7)
    ax.set_xlabel("trial $n$")
    ax.set_ylabel("experienced poles (rad/s)")
    ax.set_title("S2 pole locations", loc="left", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)



def write_snippet(summary: dict, out: Path) -> None:
    s1 = summary["ttc_mean"]["S1"]
    s2 = summary["ttc_mean"]["S2"]
    s4 = summary["ttc_mean"]["S4"]
    r1 = 100 * summary["reached_frac"]["S1"]
    r2 = 100 * summary["reached_frac"]["S2"]
    r3 = 100 * summary["reached_frac"]["S3"]
    r5 = 100 * summary["reached_frac"]["S5"]
    a3 = summary["asymp_rmse"]["S3"]
    text = (
        "% Auto-generated from order-reduction-sim.\n"
        "A third-order linear simulation (poles at $-1,-10,-20$ rad/s, "
        f'{summary["n_seeds"]["S2"]} seeds) shows that co-contraction cannot '
        "be left on as a stability crutch. A learner that stays stiff and "
        f"first-order (S3) never recovered the true plant ({r3:.0f} percent to "
        f"criterion, asymptotic relative impulse error {a3:.2f}). Random "
        f"trial-to-trial co-contraction (S5) likewise failed ({r5:.0f} percent). "
        "Learners that recruited the transient modes did recover the plant: "
        f"deep-end third-order identification from trial one (S1) in {s1:.1f} "
        f"trials ({r1:.0f} percent), staged order-recruitment (S2) in {s2:.1f} "
        f"trials ({r2:.0f} percent), and a bandwidth-only curriculum (S4) in "
        f"{s4:.1f} trials. On this linear plant the deep end is not slower "
        "than the curriculum; the result that matters for H0 is that refusing "
        "to relax leaves the transients unlearned.\n"
    )
    out.write_text(text)
