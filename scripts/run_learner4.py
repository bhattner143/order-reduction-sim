#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learner 4: co-contract to learn, and does it learn faster?

    ./scripts/run_learner4.sh
    python scripts/run_learner4.py --trials 60 --seeds 20
    python scripts/run_learner4.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from order_reduction.experiment4 import (
    CONDITION_META,
    CONDITIONS,
    FAILURE_COSTS,
    SLOW_CRITERION,
    TEST_CRITERION,
    break_even_cost,
    effective_ttc,
    run_experiment,
)
from order_reduction.plot4 import plot_failure_cost, plot_learner4, plot_learner4_proposal, write_snippet
from order_reduction.stats import holm, stars

# Every contrast is stated as a question, so a null result reads as an
# answer rather than as a missing effect.
CONTRASTS = (
    ("L2", "L1", "ttc_slow", "does co-contracting reach the slow lag sooner?"),
    ("L5", "L1", "ttc_slow", "does staged co-contraction reach it sooner?"),
    ("L6", "L1", "ttc_slow", "does a warm-started full learner reach it sooner?"),
    ("L2", "L4", "ttc_slow", "is the speed-up the filter or the restriction?"),
    ("L2", "L3", "ttc_slow", "is the filter necessary for a slow-only learner?"),
    ("L5", "L1", "ttc_test", "on the common soft probe, is staging faster?"),
    ("L6", "L5", "ttc_test", "does warm-starting beat plain staging?"),
    ("L5", "L1", "ttc_full", "on the full model, is staging faster?"),
)


def summarise(logs) -> dict:
    def med(fn):
        return {c: float(np.median([fn(lg) for lg in logs[c]])) for c in logs}

    n_trials = int(logs[next(iter(logs))][0].bias_d.size)

    def ttc_array(key, c):
        return np.array([getattr(lg, key) for lg in logs[c]], dtype=float)

    tests, p_raw, rows = [], [], []
    for a, b, key, question in CONTRASTS:
        if a not in logs or b not in logs:
            continue
        xa, xb = ttc_array(key, a), ttc_array(key, b)
        res = mannwhitneyu(xa, xb, alternative="two-sided")
        p_raw.append(float(res.pvalue))
        rows.append((a, b, key, question, float(np.median(xa) - np.median(xb))))
    if p_raw:
        for (a, b, key, question, d), p, padj in zip(rows, p_raw, holm(np.array(p_raw))):
            tests.append(
                {
                    "contrast": f"{a} vs {b}",
                    "metric": key,
                    "question": question,
                    "median_diff": d,
                    "direction": f"{a} faster" if d < 0 else ("tie" if d == 0 else f"{a} slower"),
                    "p": p,
                    "p_holm": float(padj),
                    "stars": stars(float(padj)),
                }
            )

    failure_sweep = {
        c: {
            f"{cost:g}": float(np.median([effective_ttc(lg, "ttc_test", cost) for lg in logs[c]]))
            for cost in FAILURE_COSTS
        }
        for c in logs
    }
    break_even = {
        f"{a} overtakes {b}": break_even_cost(logs, a, b, "ttc_test")
        for a, b in (("L5", "L1"), ("L6", "L1"))
        if a in logs and b in logs
    }

    return {
        "n_trials": n_trials,
        "n_seeds": {c: len(logs[c]) for c in logs},
        "slow_criterion": SLOW_CRITERION,
        "test_criterion": TEST_CRITERION,
        "failure_cost_sweep": failure_sweep,
        "break_even_failure_cost": break_even,
        "labels": {c: CONDITION_META[c]["short"] for c in logs},
        "ttc_slow_median": med(lambda lg: lg.ttc_slow),
        "ttc_full_median": med(lambda lg: lg.ttc_full),
        "ttc_test_median": med(lambda lg: lg.ttc_test),
        "reached_slow": {c: float(np.mean(ttc_array("ttc_slow", c) <= n_trials)) for c in logs},
        "reached_test": {c: float(np.mean(ttc_array("ttc_test", c) <= n_trials)) for c in logs},
        "err_final": med(lambda lg: float(np.mean(lg.err[-5:]))),
        "slow_err_final": med(lambda lg: float(np.mean(lg.slow_err[-5:]))),
        "test_err_final": med(lambda lg: float(np.mean(lg.test_err[-5:]))),
        "bias_final": med(lambda lg: float(lg.bias_d[-1])),
        "peak_bias": med(lambda lg: float(np.max(np.abs(lg.bias_d[1:])))),
        "neff_final": med(lambda lg: float(np.mean(lg.n_eff[-5:]))),
        "xi_final": med(lambda lg: float(lg.xi[-1])),
        "n_failed": {c: float(np.mean([lg.n_failed for lg in logs[c]])) for c in logs},
        "tests": tests,
    }


def _fmt(value: float, n_trials: int) -> str:
    return "never" if value > n_trials else f"{value:.0f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "learner4")
    p.add_argument("--smoke", action="store_true", help="2 seeds, 16 trials")
    args = p.parse_args()
    if args.smoke:
        args.trials, args.seeds = 16, 2
        args.out = ROOT / "outputs" / "_smoke_learner4"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"Running L1-L6 x {args.seeds} seeds x {args.trials} trials "
        "(no co-contraction / co-contract+slow / co-contract+relax) ...",
        flush=True,
    )
    logs = run_experiment(n_trials=args.trials, n_seeds=args.seeds)
    summary = summarise(logs)
    n = summary["n_trials"]

    (out / "results.json").write_text(
        json.dumps({"summary": summary}, indent=2, default=float)
    )
    plot_learner4(logs, out / "learner4.png")
    plot_learner4_proposal(logs, out / "learner4_proposal.png")
    plot_failure_cost(logs, out / "failure_cost.png")
    write_snippet(logs, summary, out / "learner4_snippet.tex")

    print(f"\n{'':4} {'ttc slow':>8} {'ttc test':>8} {'ttc full':>8} "
          f"{'|bias|':>8} {'peak':>7} {'test err':>8} {'obj err':>8} "
          f"{'n_eff':>6} {'failed':>7}  strategy")
    for c in CONDITIONS:
        if c not in logs:
            continue
        print(
            f"{c:4} {_fmt(summary['ttc_slow_median'][c], n):>8} "
            f"{_fmt(summary['ttc_test_median'][c], n):>8} "
            f"{_fmt(summary['ttc_full_median'][c], n):>8} "
            f"{abs(summary['bias_final'][c]):8.4f} "
            f"{summary['peak_bias'][c]:7.3f} "
            f"{summary['test_err_final'][c]:8.3f} "
            f"{summary['err_final'][c]:8.3f} "
            f"{summary['neff_final'][c]:6.2f} "
            f"{summary['n_failed'][c]:7.1f}  {CONDITION_META[c]['short']}"
        )

    if summary["tests"]:
        print("\nHolm-corrected two-sided Mann-Whitney:")
        for t in summary["tests"]:
            print(
                f"  {t['contrast']:9} on {t['metric']:9} "
                f"{t['median_diff']:+6.1f} trials ({t['direction']:9}) "
                f"p_holm={t['p_holm']:9.3g} {t['stars']:3}  {t['question']}"
            )

    print("\nEffective trials to the soft-probe criterion, charging a recovery")
    print("cost per failed attempt:")
    header = "     " + "".join(f"{c:>9}" for c in FAILURE_COSTS)
    print(f"{'cost':>5}" + header[5:])
    for c in ("L1", "L5", "L6"):
        if c not in logs:
            continue
        row = "".join(
            f"{summary['failure_cost_sweep'][c][f'{cost:g}']:9.1f}" for cost in FAILURE_COSTS
        )
        print(f"{c:>5}{row}")
    for name, cost in summary["break_even_failure_cost"].items():
        verdict = "never in the swept range" if cost is None else f"at a cost of {cost:g} trials"
        print(f"  {name}: {verdict}")

    print(f"\nWrote {out / 'learner4.pdf'}")
    print(f"Wrote {out / 'failure_cost.pdf'}")
    print(f"Wrote {out / 'learner4_snippet.tex'}")


if __name__ == "__main__":
    main()
