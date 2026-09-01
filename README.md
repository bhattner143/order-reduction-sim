# order-reduction-sim

Standalone repo for the third-order simulation Thrish asked to put in the
URF four-pager (meeting 1 July 2026). It is intentionally *not* the PyDrake
cup-and-ball twin. The claim to be shown is about **order reduction**:
co-contraction stretches the fast poles of a coupled plant, a learner who
never relaxes never sees those transients, and the true third-order object
stays unlearned.

Plant (unit DC gain):

    G(s) = 1 / ((1 + s)(1 + 0.1 s)(1 + 0.05 s))

Poles at -1, -10, -20 rad/s. Co-contraction xi stretches the two fast time
constants by the overdamped separation ratio
rho(xi) = (xi + sqrt(xi^2 - 1))^2.
The learner is ERA (Hankel SVD) of a noisy impulse, keeping `order` modes.
Skill is always scored as relative impulse error against the *true*
(unstiffened) G.

| Condition | What the learner does |
|-----------|------------------------|
| S1 deep end | xi = 0.7, identify 3rd order from trial one |
| S2 order recruitment | xi = 3 -> 2 -> 0.7, model order 1 -> 2 -> 3, 10 trials per stage |
| S3 stiff throughout | xi = 3, first-order model forever |
| S4 bandwidth only | xi = 0.7, command bandwidth staged (curriculum without impedance) |
| S5 random tuning | xi ~ Uniform[0.7, 3] each trial, 3rd-order identifier |

## Run

```bash
cd /Volumes/Data/royal-sc-urf/order-reduction-sim
./scripts/run_curriculum.sh                  # 60 trials x 20 seeds
./scripts/run_curriculum.sh --trials 40 --seeds 5 --out outputs/_smoke
```

Uses `/opt/anaconda3/envs/pydrake_sfil/bin/python`.

Outputs in `outputs/curriculum/`:

- `proposal_figure.pdf` -- learning curves (mean +/- SE) and trials-to-criterion
- `mechanism.pdf` -- Hankel SV ratio, impulse mismatch, S2 pole migration
- `results.json` -- all numbers
- `proposal_snippet.tex` -- paragraph ready to paste into v8

## Result (20 seeds, 60 trials)

S3 never recovers the true plant (0% to criterion, asymptotic error 0.41).
S5 (random co-contraction) also fails (0%). S1, S2 and S4 all recover it
(100%). On this linear plant the deep end is *faster* than the curriculum
(7 vs 24 trials); the result that matters for H0 is that refusing to relax
leaves the transients unlearned. That is the figure that belongs in the
four-pager, not a claim that S2 beats S1.
