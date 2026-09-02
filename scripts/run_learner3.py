#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learner 3: slosh object, bias law, open vs closed plant-inverse commands.

    ./scripts/run_learner3.sh
    python scripts/run_learner3.py --trials 60 --seeds 20
    python scripts/run_learner3.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from order_reduction.experiment3 import CONDITIONS, run_experiment
from order_reduction.plot3 import plot_bias_law, plot_learner3, write_snippet
from order_reduction.slosh import report as bias_report
from order_reduction.slosh import sweep as bias_sweep


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
        "wn_final": med(lambda lg: float(lg.wn_hat[-1])),
        "wn_err_final": med(lambda lg: float(abs(lg.wn_hat[-1] - 10.0))),
        "neff_final": med(lambda lg: float(np.mean(lg.n_eff[-5:]))),
        "n_failed": {c: float(np.mean([lg.n_failed for lg in logs[c]])) for c in logs},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "learner3")
    p.add_argument("--smoke", action="store_true", help="2 seeds, 12 trials, all conditions")
    args = p.parse_args()
    if args.smoke:
        args.trials, args.seeds = 12, 2
        args.out = ROOT / "outputs" / "_smoke_learner3"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("Slosh reduced-order bias against co-contraction ...", flush=True)
    rows = bias_sweep()
    print(bias_report(rows))

    print(f"\nRunning C1-C5 x {args.seeds} seeds x {args.trials} trials ...", flush=True)
    logs = run_experiment(n_trials=args.trials, n_seeds=args.seeds)
    summary = summarise(logs)

    (out / "results.json").write_text(
        json.dumps({"bias_sweep": rows, "summary": summary}, indent=2, default=float)
    )
    plot_bias_law(rows, out / "slosh_bias_law.png")
    plot_learner3(logs, out / "learner3.png")
    write_snippet(rows, logs, summary, out / "learner3_snippet.tex")

    print(
        f"\n{'':4} {'ttc':>5} {'final err':>10} {'peak bias':>10} "
        f"{'wn err':>8} {'n_eff':>7} {'failed':>7}"
    )
    for c in CONDITIONS:
        if c not in logs:
            continue
        print(
            f"{c:4} {summary['ttc_median'][c]:5.0f} {summary['err_final'][c]:10.3f} "
            f"{summary['peak_bias'][c]:10.3f} {summary['wn_err_final'][c]:8.2f} "
            f"{summary['neff_final'][c]:7.2f} {summary['n_failed'][c]:7.1f}"
        )
    print(f"\nWrote {out / 'slosh_bias_law.pdf'}")
    print(f"Wrote {out / 'learner3.pdf'}")
    print(f"Wrote {out / 'learner3_snippet.tex'}")


if __name__ == "__main__":
    main()
