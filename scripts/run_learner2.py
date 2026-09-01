#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learner 2: the reduced-order bias law, and K1-K5 with the two-timescale EKF.

    ./scripts/run_learner2.sh
    python scripts/run_learner2.py --trials 60 --seeds 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from order_reduction.bias import report as bias_report
from order_reduction.bias import sweep as bias_sweep
from order_reduction.experiment2 import run_experiment
from order_reduction.plot2 import plot_bias_law, plot_learner2, write_snippet


def summarise(logs) -> dict:
    def med(fn):
        return {c: float(np.median([fn(lg) for lg in logs[c]])) for c in logs}

    return {
        "n_trials": int(logs[next(iter(logs))][0].err.size),
        "n_seeds": {c: len(logs[c]) for c in logs},
        "ttc_median": med(lambda lg: lg.ttc),
        "err_final": med(lambda lg: float(np.mean(lg.err[-5:]))),
        "bias_final": med(lambda lg: float(lg.bias_d[-1])),
        "peak_bias": med(lambda lg: float(np.max(np.abs(lg.bias_d[1:])))),
        "neff_final": med(lambda lg: float(np.mean(lg.n_eff[-5:]))),
        "n_failed": {c: float(np.mean([lg.n_failed for lg in logs[c]])) for c in logs},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "learner2")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("Reduced-order bias against co-contraction ...", flush=True)
    rows = bias_sweep()
    print(bias_report(rows))

    print(f"\nRunning K1-K5 x {args.seeds} seeds x {args.trials} trials ...", flush=True)
    logs = run_experiment(n_trials=args.trials, n_seeds=args.seeds)
    summary = summarise(logs)

    (out / "results.json").write_text(
        json.dumps({"bias_sweep": rows, "summary": summary}, indent=2, default=float)
    )
    plot_bias_law(rows, out / "bias_law.png")
    plot_learner2(logs, out / "learner2.png")
    write_snippet(rows, logs, summary, out / "learner2_snippet.tex")

    print(
        f"\n{'':4} {'ttc':>5} {'final err':>10} {'final bias':>11} "
        f"{'peak bias':>10} {'n_eff':>7} {'failed':>7}"
    )
    for c in logs:
        print(
            f"{c:4} {summary['ttc_median'][c]:5.0f} {summary['err_final'][c]:10.3f} "
            f"{summary['bias_final'][c]:+11.4f} {summary['peak_bias'][c]:10.3f} "
            f"{summary['neff_final'][c]:7.2f} {summary['n_failed'][c]:7.1f}"
        )
    print(f"\nWrote {out / 'bias_law.pdf'}")
    print(f"Wrote {out / 'learner2.pdf'}")
    print(f"Wrote {out / 'learner2_snippet.tex'}")


if __name__ == "__main__":
    main()
