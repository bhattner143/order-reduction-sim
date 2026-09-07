# order-reduction-sim

Simulation suite for the order-reduction / co-contraction hypothesis used in
the Royal Society URF proposal *Tuning to Learn*.

## Regenerate the URF proposal figure (Learner 4)

The proposal cites **Learner 4** only (EKF, fixed fast reach, L1–L6). Full
walkthrough: [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

```bash
python -m pip install -r requirements.txt
./scripts/run_learner4.sh --trials 60 --seeds 20
# override interpreter if needed:
# PYTHON=python3 ./scripts/run_learner4.sh --trials 60 --seeds 20
```

Writes under `outputs/learner4/`:

| File | Role |
|------|------|
| `learner4_proposal.pdf` / `.png` | **URF panel** (copy to `urf-proposal/figures/learner4-sim.pdf`) |
| `learner4.pdf` / `.png` | Full diagnostic |
| `failure_cost.pdf` / `.png` | Failure-recovery cost sweep |
| `results.json` | Numbers + Holm tests |
| `learner4_snippet.tex` | Auto-generated LaTeX prose |

Smoke test: `./scripts/run_learner4.sh --smoke`.

Committed plots under `outputs/learner4/` match the canonical 20×60 run; re-run
the command above to regenerate them from scratch.

---

# Historical note: Learners 1–3

Earlier experiments tested the order-reduction hypothesis from the 2026-07-01
meeting with Thrishantha Nanayakkara
(`meetings/2026-07-01-meeting-thrish.tex`).

**For Learners 1–3: the hypothesis is not supported in the form the meeting
note states it, and the reason is structural rather than a matter of parameter
tuning.** Details below. That negative result motivated Learner 4 (different
scoreboard, same plant), which *is* the evidence used in the URF draft.

## What is being claimed

The note argues that co-contraction helps learning by reducing the effective
order of the system the learner faces. Raising the damping ratio past one
separates the time scales by `rho(xi) = (xi + sqrt(xi^2-1))^2`, the fast modes
settle within a fraction of a movement, and the learner is left facing the
dominant first-order lag alone. The advantage is then claimed through sample
complexity, Eq. (7)-(8) of the note:

    N(r, delta)  ~  sigma^2 r / (delta lambda_min(Phi_r))

so a staged curriculum that identifies one mode at a time should need fewer
trials than identifying all three at once.

## Set-up

The object never changes: `G(s) = 1/((1+s)(1+0.1s)(1+0.05s))`, poles at -1,
-10, -20. On each trial the learner makes one reach while holding a chosen
co-contraction, which compresses the two fast time constants by `rho(xi)`.

Five conditions, matching the note:

| | co-contraction | movement | modes fitted |
|---|---|---|---|
| S1 deep end | 0.7 fixed | fast | all three from trial one |
| S2 order recruitment | 3.0 -> 1.3 -> 0.7 | slow -> fast | 1 -> 2 -> 3 |
| S3 stiff throughout | 3.0 fixed | slow | 1 only |
| S4 bandwidth only | 0.7 fixed | slow -> fast | 1 -> 2 -> 3 |
| S5 random tuning | uniform per trial | fast | all three |

Three modelling choices matter more than any parameter value:

* **Exploration is band-limited to the movement.** A limb cannot emit an
  impulse or white noise. If it could, every mode would be excited at once and
  the sample-complexity argument would be void by construction.
* **Learning is fixed-step gradient descent, not a batch solve.** A batch
  least-squares fit with the correct structure is consistent, so it recovers
  all three time constants as soon as the data support them and a curriculum
  could only ever cost trials. Eq. (7)-(8) is a statement about convergence
  *rate*, and rate is what a gradient learner is limited by.
* **Trials can fail.** A reach is aborted when the mismatch between what the
  learner predicted and what it felt exceeds what the limb can absorb,
  tolerance scaling with stiffness. This is the note's "if stabilisation is not
  there the ball goes out, the task fails, there is no learning".

## Findings

Run `./scripts/run_curriculum.sh --trials 80 --seeds 20`.

**1. Co-contraction destroys identifiability of the transients rather than
improving it.** The condition number of the full third-order fit rises by
thirteen orders of magnitude across the co-contraction range, because
compressing the fast modes removes exactly the information needed to estimate
them:

| xi | rho | kappa(Phi) | curvature on tau_d |
|---|---|---|---|
| 0.7 | 1.0 | 5.5e3 | 2.54e-2 |
| 1.3 | 4.5 | 6.2e5 | 2.70e-2 |
| 3.0 | 34 | 1.8e12 | 2.71e-2 |
| 4.0 | 62 | 2.7e16 | 2.72e-2 |

Note the last column. What stiffening buys for the dominant mode is about
seven percent more curvature. What it costs for the transients is everything.
The reduced problem is well posed while stiff, but only because the modes that
make it hard have been deleted from the data, not solved.

**2. Badly identified directions are also behaviourally cheap.** This is the
finding that closes the sample-complexity route. Taking equal-sized steps
along the eigenvectors of the curvature matrix, the direction that is 5500
times slower to learn changes the object's impulse response 28 times less.
Poor conditioning and low behavioural relevance are the same thing here, so a
bound on parameter error does not become a bound on performance. Eq. (7)-(8)
is true and does not imply what the note needs it to imply.

**3. The curriculum therefore loses.** S2 reaches criterion in a median of 80
trials against 47 for the deep end (Holm-corrected two-sided Mann-Whitney,
p = 1.6e-7). It is paying roughly fifteen trials of staging overhead for an
identification advantage that does not exist. S3 never reaches criterion,
plateauing at a model error of 0.44, but that is close to tautological: it was
never allowed to unlock the second and third modes.

**4. Co-contraction does buy safety, and that effect is real.** S2 and S3
record zero failed trials; S1 and S4, which are soft while their model is
still wrong, record 10 to 12. This is the one part of the mechanism that
survives intact.

## What this means for the proposal

The order-reduction story should be reframed from a statistical claim to a
risk claim. Co-contraction is not a way to make identification cheaper. It is
what makes an aggressive movement survivable before the model that would
justify it exists. Learning is gated by the trials you do not lose, not by the
conditioning of the regressor.

That reframing also changes what the experiment has to measure. Trials to
criterion on a model-error metric will not separate the conditions, because
the deep end is not actually penalised for its ill-conditioning. The
discriminating measurement is failure rate and the recovery cost of a failed
attempt, under a task whose performance demand does not wait for the model.

The version of H1 that this simulation does support is narrower and cleaner
than the one in the note: *the dominant mode is estimable under co-contraction
and the transients are not, so any learner that stays stiff is provably stuck
at first order.* Panel 1 of `outputs/curriculum/mechanism.pdf` is that claim,
and it needs no curriculum to make it.

---

# Learner 2: the misspecification bias, and a learner that carries uncertainty

Run `./scripts/run_learner2.sh --trials 60 --seeds 20`.

Learner 1 is kept exactly as it was, because its failure is what identifies the
defect. That defect was **not** the plant or the schedule. It was that learner 1
was *correctly specified*: it always fitted the true three-lag structure, so its
only error was variance, and variance is not what co-contraction acts on.

## The bias law

A learner that fits **fewer** modes than the world has is biased, and the bias
has a closed form. Expanding the object at low frequency,
`G(s) = 1 - (tau_d + tau_1 + tau_2)s + O(s^2)`, and matching a first-order model
`1/(1 + tau_hat s)`:

    tau_hat(xi) = tau_d + (tau_1 + tau_2)/rho(xi)

Three lags in a row feel like one lag equal to their sum, so a reduced-order
learner is off by exactly the time constants it omits -- and co-contraction
divides that error by `rho`. Least squares confirms it:

| xi | rho | fitted bias | closed form |
|---|---|---|---|
| 0.7 | 1.0 | 0.180 s | 0.150 s |
| 1.2 | 3.5 | 0.040 s | 0.043 s |
| 2.0 | 13.9 | 0.005 s | 0.011 s |
| 4.0 | 62 | 0.0006 s | 0.0024 s |

This is the claim worth making. Co-contraction is not a statistical aid to
identification; it is what makes a **low-order model of a high-order world
correct**. Unlike a variance argument, it does not go away with more trials.

## The learner

State is `[log tau_d, log tau_1, log tau_2, w_1, w_2]`, updated by an iterated
EKF in information form, with anisotropic process noise -- small on the dominant
mode, larger on the transients. That is the two-timescale split: a slow,
well-retained estimate and a fast, forgetful one.

Each transient enters through a **participation weight**,
`blend(tau, w): x -> (1-w)x + w*lag(tau)x`, with an ARD prior shrinking the
weights to zero, so `n_eff = 1 + w_1 + w_2` comes off the posterior instead of
being imposed. Nothing tells the learner what order to use.

The participation weight is load-bearing. Encoding "this mode is off" as
`tau -> 0` also drives the mode's sensitivity to zero, so a pruned mode becomes
invisible to the data and **pruning is irreversible** -- a learner that switches
a mode off while stiff can never switch it back on when it relaxes, however
plainly the soft data show it. That was observed before it was diagnosed.

Relaxation is earned, not scheduled: K2 and K4 advance when the posterior
standard deviation on `log tau_d` drops below a threshold.

## Findings

| | trials | final error | peak &#124;tau_d error&#124; | n_eff | failed |
|---|---|---|---|---|---|
| K1 deep end | 5 | 0.031 | 0.691 s | 2.18 | 4.0 |
| K2 confidence-gated relaxation | 19 | 0.030 | 0.087 s | 2.18 | 0.0 |
| K3 stiff throughout | never | 0.404 | 0.013 s | 1.00 | 0.0 |
| K4 bandwidth only | 4 | 0.030 | 0.026 s | 2.19 | 1.1 |
| K5 random tuning | 35 | 0.076 | 0.040 s | 1.92 | 0.7 |

**K3 prunes itself to n_eff = 1.00** and holds an unbiased dominant estimate
(+0.0014 s) while plateauing at 0.404. Staying stiff leaves you provably at
first order, and here the learner *discovers* that rather than being told --
which is what makes it evidence rather than a definition.

**K2 holds n_eff at 1.0 through the stiff stage and recruits to 2.18 the moment
co-contraction falls.** Progressive recruitment emerges from evidence.

**K1 reaches the same asymptote but swings through a peak dominant-mode error of
0.691 s**, eight times K2's, and loses four trials to failure against K2's zero.

Still not confirmed, and not to be claimed: K2 is *not* faster than K1 in trials
to criterion, and K4 (bandwidth alone) is fastest of all. Trials to criterion
does not separate these schedules and has been dropped as the headline metric.
`n_eff` asymptotes near 2.2 rather than 3 because the third mode does not pay
for itself in this excitation -- the ARD prior behaving correctly.

## Next

- Close the loop: a controller designed from the posterior, with co-contraction
  setting the gain margin. Instability during learning is the effect, not an
  obstacle to it.
- Sweep the cost of a failed trial; if recovery is expensive the K1/K2 ordering
  may reverse on trial count too.
- Measure retention of each estimate across a change in co-contraction.
- Test the bias law against WP1 data: does a participant's residual dominant
  time constant track `1/rho` of their own measured co-contraction?

## Layout

    order_reduction/plant.py           object, rho(xi), band-limited exploration
    order_reduction/learner.py         learner 1, gray-box gradient, order unlocking
    order_reduction/experiment.py      S1-S5 trial loop, surprise-driven failure
    order_reduction/diagnose.py        curvature spectrum vs behavioural cost
    order_reduction/stats.py           Holm-corrected two-sided tests
    order_reduction/plot.py            proposal_figure.pdf, mechanism.pdf
    order_reduction/bias.py            the (tau_1+tau_2)/rho bias law
    order_reduction/kalman_learner.py  learner 2, two-timescale EKF with ARD
    order_reduction/experiment2.py     K1-K5, confidence-gated relaxation
    order_reduction/plot2.py           bias_law.pdf, learner2.pdf
    order_reduction/slosh.py           lag + complex-pole slosh, plant inverse
    order_reduction/slosh_learner.py   learner 3, EKF on (tau_d, wn, zeta, w)
    order_reduction/experiment3.py     C1-C5, open vs cautious plant-inverse
    order_reduction/plot3.py           slosh_bias_law.pdf, learner3.pdf
    order_reduction/slow_learner.py    learner 4, first-order EKF on tau_d only
    order_reduction/experiment4.py     L1-L6, three strategies, three scoreboards
    order_reduction/plot4.py           learner4.pdf, failure_cost.pdf

---

# Learner 3: slosh object, no task model

Run `./scripts/run_learner3.sh --trials 60 --seeds 20`.

Learner 1 and 2 keep the three real-lag cup. Learner 3 replaces the two fast
lags with a lightly damped resonance, because coffee sloshes:

    G(s) = 1/(1 + tau_d s) * wn^2 / (s^2 + 2 zeta wn s + wn^2)

with `(tau_d, wn, zeta) = (1.0 s, 10 rad/s, 0.12)`. Co-contraction still
compresses only the fast part: `wn_felt = wn * rho(xi)`. The learner is the
same two-timescale EKF, now over `[log tau_d, log wn, log zeta, w]`.

**Nothing here is a model of the task.** The min-jerk reach is the same goal
already used as an open-loop command. Scoring is object impulse error, bias
on `tau_d`, `n_eff`, and failed trials. Tracking error is logged and is never
the criterion.

Closed-loop conditions (C4, C5) build the command from the current *plant*
belief: a regularised inverse of the felt slosh, mixed cautiously with the
reach. Authority grows with the slosh weight and shrinks with uncertainty, so
a mode that has not been recruited is not inverted. A wrong `wn` puts the
lead at the wrong frequency and corrupts later data. That is the information
change; it is not a controller designed from a task cost.

| | command | stiffness / speed |
|---|---|---|
| C1 | open (reach + explore) | soft, fast |
| C2 | open | confidence-gated curriculum |
| C3 | open | stiff throughout |
| C4 | cautious plant inverse | soft, fast |
| C5 | cautious plant inverse | confidence-gated curriculum |

## Verification pass, and four defects it found

The first version of this learner was audited numerically rather than by
eye. Four things were wrong; all are fixed, and the fixes changed the
numbers. Recording them because three are the same class of mistake the
earlier learners were criticised for.

**1. The resonance was switched off by an `if`, not compressed.** At
`dt = 0.01` s a felt slosh at `xi = 3` sits near 340 rad/s, which the grid
cannot represent, so the original code passed the signal straight through
above a threshold. That threshold fell at a *different* `xi` for the plant
than for the learner, because their `wn` differ; it put a cliff in the loss
surface; and it made `dh/d log wn` **exactly zero** (measured 0.000e+00 at
`xi = 3`) rather than merely small. C3's "it never learns the frequency" was
therefore a branch, not a discovery, which is precisely the hand-coding that
Learner 1's mode-locking was rejected for. Fixed by integrating the
resonance on an adaptive sub-grid and decimating. The Jacobian now decays
smoothly (3.0e-1, 4.7e-2, 9.7e-3, 3.6e-3, 2.1e-3 at `xi` = 0.7, 1.2, 2.0,
3.0, 4.0) and is never exactly zero.

**2. The decimation offset was wrong.** Taking `y_fine[os-1::os]` reads one
sub-step early and biases the low-frequency gain; `y_fine[::os]` is correct.
Checked against the analytic `2 zeta / wn`, the corrected version now agrees
to five decimals at every `wn` tested.

**3. A 5 ms discretisation floor swamped the effect.** Holding the command
across a sample adds `dt/2` of pure delay, which a first-order fit absorbs
as extra time constant. At `dt = 0.01` s that floor is 5 ms while the
predicted slosh bias at `xi = 3` is 0.7 ms, so the 1/rho law was
unmeasurable. Same trap that `cascade` vs `cascade_exact` documents for the
three-lag plant, one order worse here. The bias sweep now runs at
`dt = 5e-4`.

**4. The bias law needs a slow probe, and a fast probe inverted its sign.**
`tau_hat = tau_d + 2 zeta / wn` is the *leading order* term of a
low-frequency expansion. Probe near `wn` and the least squares fit stops
being a low-frequency match: it chases the ringing, and the measured bias
goes **negative**, -0.030 s at `t_move = 0.25` against a predicted +0.024.
Slowing the probe recovers the law monotonically: +0.0005 at 0.8 s, +0.011
at 1.5 s, +0.017 at 2.5 s, +0.0195 at 4.0 s. The three-lag plant never
showed this because a cascade of real lags cannot ring. The sweep now probes
at `t_move = 2.5` s and the caveat is stated rather than hidden.

Checked and correct as written: `invert_lag` is exact to 1e-12,
`invert_slosh` has DC gain 1.000000 with the intended rolloff, and the
analytic bias formula matches the numeric low-frequency slope exactly at
`xi = 0.7`.

**The multi-basin premise is real, but the learner was never made to face
it.** Scanning the loss over `log wn` on a soft fast trial gives two minima,
the true one at 9.9 rad/s and a side lobe at 22.2 rad/s. The original
`WN_INIT = 4.0` sits downhill of the true minimum, so the learner walked
straight to it and the non-convexity that motivated a complex pole was never
encountered. `WN_INIT` is now 24.0, inside the wrong basin.

## Findings (20 seeds, 60 trials, after the fixes)

| | trials | final error | peak &#124;tau_d error&#124; | wn error | n_eff | failed |
|---|---|---|---|---|---|---|
| C1 deep end, open | 7 | 0.039 | 0.329 s | 0.05 | 2.00 | 3.6 |
| C2 curriculum, open | 22 | 0.041 | 0.024 s | 0.12 | 1.98 | 0.0 |
| C3 stiff throughout | never | 0.570 | 0.017 s | 14.00 | 1.00 | 0.0 |
| C4 deep end, closed | 8 | 0.031 | 0.700 s | 0.08 | 2.00 | 4.3 |
| C5 curriculum, closed | 22 | 0.030 | 0.024 s | 0.08 | 2.00 | 0.0 |

**The slosh bias law now holds across the range.** Fitted against predicted:
0.0166/0.0240 at `xi = 0.7`, then 0.0065/0.0069, 0.0035/0.0035,
0.0019/0.0017, 0.0012/0.0010, 0.0009/0.0007, 0.0006/0.0004. The soft end
sits at 69% of the asymptote only because a finite-speed probe is still not
a zero-frequency probe.

**C3 stays at first order for a real reason now.** `n_eff = 1.00` and `wn`
never leaves its initial value, but the Jacobian on `wn` at `xi = 3` is
3.6e-3, not zero, and the loss over `wn` has a total range of 2.5e-5. The
information is absent, not deleted. That is the claim K3 was supposed to
make, now made without a branch propping it up.

**Curriculum is still safer, not faster, and the result is now much harder
to dismiss.** C2 and C5 have zero failed trials and roughly 14 times smaller
peak dominant-mode bias than the deep end, while taking three times as many
trials. This survives a genuinely non-convex frequency search with the
learner initialised *inside* the wrong basin.

**Why the deep end escapes the spurious basin anyway.** The EKF inflates its
measurement noise while the residual is large, `R = max(R_std^2, mean(e^2))`.
That flattens the effective loss surface exactly when the model is worst,
which is graduated non-convexity arriving for free. The filter already does
internally what the curriculum was supposed to supply externally. This is
the sharpest single reason the speed claim keeps failing across all three
learners, and it belongs in the proposal.

**Closed-loop plant inversion still did not flip the ordering.** C4 matches
C1 on trials (8 vs 7) but is worse on both risk measures: peak bias 0.700 s
against 0.329 s, and 4.3 failures against 3.6. A wrong inverse does corrupt
the trajectory; it just does not slow identification. Authority is limited
by the slosh weight and by uncertainty, so the inverse stays nearly
open-loop until the slosh has already been learned. An unrestricted inverse
aborts every trial before usable data.

Do not claim: "Learner 3 learns the dominant pole faster under a curriculum
than C1/K1/S1." It does not, and it now fails to under conditions
deliberately built to favour it. The usable content is the slosh plant, the
bias law surviving a change of transient, emergent recruitment of a
resonance, and a closed-loop command that is explicitly *not* a task model.

---

# Learner 4: do we co-contract in order to learn, and does it learn faster?

Run `./scripts/run_learner4.sh --trials 60 --seeds 20`.

Learners 1--3 scored a model of the *whole object*. That is why S3/K3/C3
looked like failures: a stiff first-order learner's impulse error against the
true third-order cup plateaus near 0.40, even when its estimate of the
dominant lag is essentially perfect. Learner 4 changes the scoreboard, not
the plant, and compares the three strategies a person could actually adopt:

1. **Do not co-contract, and identify the whole third-order plant** (L1).
2. **Co-contract and learn only the slow lag**, treating the transients as
   noise (L2).
3. **Co-contract, then relax progressively and learn the whole model**
   (L5, and L6 which hands the slow lag to a full-order learner explicitly).

L3 and L4 are the controls that say which half of L2 is doing the work.
Movement time is held at the fast reach (`t_move = 0.25` s) in *every* cell,
including the staged ones, so the only thing that ever varies is `xi`.

| | learner | stiffness | strategy |
|---|---|---|---|
| L1 | third-order EKF | soft throughout | no co-contraction, learn the full model |
| L2 | first-order EKF | stiff throughout | co-contract, learn the slow model only |
| L3 | first-order EKF | soft throughout | control: capacity without the filter |
| L4 | third-order EKF | stiff throughout | control: filter without the restriction |
| L5 | third-order EKF | staged 3.0 -> 1.3 -> 0.7 | co-contract, then learn the whole model gradually |
| L6 | first -> third-order | staged | learn the slow lag stiff, then warm-start the full model |

Relaxation in L5/L6 is confidence-gated exactly as K2 was: advance when the
posterior sd on `log tau_d` falls below 0.04, minimum five trials per stage.
L6 promotes at the first relaxation, transferring `log tau_d` and its
posterior variance into a fresh `TwoTimescaleEKF` and nothing else.

## Three scoreboards, because the scoreboard is the whole argument

| name | measure | criterion |
|---|---|---|
| slow | `\|tau_hat_d - tau_d\|` | < 0.08 s |
| full | impulse error of the belief about the whole object | < 0.10 |
| test | relative error predicting a **common soft, fast probe** | < 0.10 |

The test scoreboard is the fair one and is new. Every condition is asked to
predict the same deterministic, noise-free soft reach, whatever stiffness it
happens to be holding, so "I can only do this while braced" shows up as a
cost rather than being hidden.

## Findings (20 seeds, 60 trials)

| | ttc slow | ttc test | ttc full | peak \|tau_d err\| | test err | obj err | n_eff | failed |
|---|---|---|---|---|---|---|---|---|
| L1 soft, 3rd | 6 | **5** | **5** | 0.691 s | 0.007 | 0.031 | 2.18 | 4.0 |
| L2 stiff, slow-only | **3** | never | never | 0.015 s | 0.216 | 0.405 | 1.00 | 0.0 |
| L3 soft, slow-only | never | never | never | 0.962 s | 0.175 | 0.338 | 1.00 | 16.3 |
| L4 stiff, 3rd | **3** | never | never | 0.015 s | 0.215 | 0.404 | 1.00 | 0.0 |
| L5 staged, 3rd | **3** | 17 | 19 | 0.092 s | 0.007 | 0.030 | 2.18 | 0.0 |
| L6 staged, slow then recruit | **3** | 8 | 13 | 0.025 s | 0.007 | 0.031 | 2.19 | 0.0 |

Holm-corrected two-sided Mann-Whitney, all on trials to criterion:

| contrast | metric | difference | p |
|---|---|---|---|
| L2 vs L1 | slow | -3 (L2 faster) | 3.9e-7 |
| L5 vs L1 | slow | -3 (L5 faster) | 3.9e-7 |
| L6 vs L1 | slow | -3 (L6 faster) | 3.9e-7 |
| L2 vs L4 | slow | 0 (tie) | ns |
| L2 vs L3 | slow | -58 (L2 faster) | 3.8e-9 |
| L5 vs L1 | test | +12 (L5 **slower**) | 3.7e-8 |
| L6 vs L5 | test | -9 (L6 faster) | 1.8e-8 |
| L5 vs L1 | full | +14 (L5 **slower**) | 6.1e-8 |

**Co-contraction does learn the slow dynamics faster: 3 trials against 6,
with zero failures against four.** Every co-contracting strategy (L2, L5,
L6) reaches the slow lag in half the trials the soft learner needs, and
without a single lost attempt. This is the claim, and it holds.

**The filter is doing the work, not the restriction.** L2 and L4 are an
exact tie. Once co-contraction has compressed the transients, a full
third-order EKF prunes itself to `n_eff = 1.00` and learns `tau_d` just as
fast as a learner that was never offered the extra modes. Order reduction
is a property of the coupled plant, not a modelling choice.

**Neglecting the transients only works if something has actually removed
them.** L3 -- the same first-order learner held soft -- never reaches
criterion, plateaus at +0.218 s of bias (the closed-form floor is
`(tau_1+tau_2)/rho = 0.150` s, the rest is the fast probe), and loses 16
trials. Calling the transients "noise" is legitimate *after* the filter and
a systematic error before it.

**But on the whole object, co-contraction is still not a shortcut.** This is
the part that does not go the way the hypothesis wants. Tested on the common
soft probe, L1 arrives in 5 trials, L6 in 8 and L5 in 17. Staging costs
12 trials, not saves them, and the reason is mechanical: the staged learner
spends its first ~15 trials in a regime where the transients are not in the
data at all, so it cannot learn them however good its slow model is. L2,
which never relaxes, never reaches the soft criterion at all -- it plateaus
at a test error of 0.216. Co-contraction buys a correct slow model quickly;
only relaxation buys the transients.

**Warm-starting recovers most of that loss.** L6 beats L5 by 9 trials
(p = 1.8e-8) purely by handing the converged slow lag to the full-order
learner rather than letting the ARD prior rediscover it. If a curriculum is
going to be defended, this is the version to defend: the value is in the
*transfer* of the slow estimate, not in the staging itself.

## The failure cost is what decides it

`./scripts/run_learner4.sh` also writes `failure_cost.pdf`. The simulation
charges nothing for a failed attempt beyond losing the rest of that trial's
data, which is the most generous possible assumption for the deep end. This
is the sweep `report/theory.tex` pre-registered. Charging a recovery
overhead of `k` extra trials per failure:

| recovery cost (trials) | 0 | 0.5 | 1 | 2 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|
| L1 soft | 5.0 | 6.5 | 8.0 | 11.0 | 14.0 | 20.0 | 29.0 | 41.0 |
| L5 staged | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 |
| L6 staged, warm-start | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 |

**L6 draws level with L1 as soon as one failed attempt costs a single extra
trial, and wins outright beyond that. L5 overtakes at a cost of 5.** The
staged learners are flat because they never fail. So the honest statement is
conditional, and the condition is cheap to satisfy: *if recovering from a
dropped attempt costs anything at all, co-contracting to learn is also the
faster route to the whole model.* For a cup of coffee, a bicycle, or a
trainee surgeon, a failed attempt costs a great deal more than one trial.

## What to claim and what not to

Claim:

- Co-contraction makes the felt plant first-order, and identifying one lag
  is twice as fast and far safer than identifying three (L2/L5/L6 vs L1 on
  the slow scoreboard, p = 3.9e-7).
- The speed-up comes from the coupling, not from the learner choosing a
  smaller model (L2 = L4), and the filter is necessary (L3 fails).
- With any non-zero cost for a failed attempt, the co-contracting route is
  also the faster route to a full model of the object.

Do not claim:

- "Restricting the learner to first order is what speeds learning." L4
  refutes that.
- "Staging is faster than the deep end, full stop." On free failures it is
  12--14 trials slower. The claim requires the failure cost.
- "A learner that stays stiff eventually learns the object." L2 never
  reaches the soft-probe criterion and plateaus at 0.216.

## Sensitivity worth knowing

The 12-trial staging penalty in L5 is set by `MIN_TRIALS_PER_STAGE = 5`
across three stages: 15 trials of forced dwell before the learner is fully
soft. That parameter, not the physics, sets the size of the penalty. The
break-even failure costs above move with it, so quote them as
order-of-magnitude, not to two significant figures.
