# order-reduction-sim

A test of the order-reduction hypothesis from the 2026-07-01 meeting with
Thrishantha Nanayakkara, as that hypothesis is stated in
`meetings/2026-07-01-meeting-thrish.tex`.

**Result: the hypothesis is not supported in the form the meeting note states
it, and the reason is structural rather than a matter of parameter tuning.**
Details below. The negative result is more useful to the proposal than the
demonstration would have been, because it identifies which of the two claims
in the note is load-bearing.

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

## Layout

    order_reduction/plant.py       object, rho(xi), band-limited exploration
    order_reduction/learner.py     gray-box gradient learner, order unlocking
    order_reduction/experiment.py  S1-S5 trial loop, surprise-driven failure
    order_reduction/diagnose.py    curvature spectrum vs behavioural cost
    order_reduction/stats.py       Holm-corrected two-sided tests
    order_reduction/plot.py        proposal_figure.pdf, mechanism.pdf
