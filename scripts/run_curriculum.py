#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Thrish's S1-S5 third-order curriculum simulation and write proposal figures.

    ./scripts/run_curriculum.sh
    python scripts/run_curriculum.py --trials 80 --seeds 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from order_reduction.diagnose import report, sweep
from order_reduction.experiment import run_experiment
from order_reduction.plot import plot_mechanism, plot_proposal_figure, write_snippet
from order_reduction.stats import summarise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "curriculum")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running S1-S5 x {args.seeds} seeds x {args.trials} trials ...", flush=True)
    logs = run_experiment(n_trials=args.trials, n_seeds=args.seeds, seed0=args.seed0)
    summary = summarise(logs)

    (out / "results.json").write_text(json.dumps(summary, indent=2))

    plot_proposal_figure(logs, summary, out / "proposal_figure.png")
    plot_mechanism(logs, summary, out / "mechanism.png")
    write_snippet(summary, out / "proposal_snippet.tex")

    print("\nTrials to criterion (median | mean +/- SE):")
    for c in logs:
        print(
            f"  {c}: {summary['ttc_median'][c]:5.1f} | {summary['ttc_mean'][c]:6.2f}"
            f" +/- {summary['ttc_se'][c]:4.2f}"
            f"   reached {100 * summary['reached_frac'][c]:5.1f}%"
            f"   asymptotic error {summary['asymp_err'][c]:.3f}"
            f"   failed trials {summary['n_failed'][c]:5.1f}"
        )
    print("\nHolm-corrected Mann-Whitney, two-sided:")
    for t in summary["tests"]:
        print(
            f"  {t['contrast']:10s}  median diff {t['median_diff']:+6.1f}"
            f"  ({t['direction']})  p_Holm={t['p_holm']:.3g}  {t['stars']}"
        )

    print("\nIdentifiability of the full third-order fit vs co-contraction:")
    print(report(sweep()))

    print(f"\nWrote {out / 'proposal_figure.pdf'}")
    print(f"Wrote {out / 'mechanism.pdf'}")
    print(f"Wrote {out / 'proposal_snippet.tex'}")


if __name__ == "__main__":
    main()
