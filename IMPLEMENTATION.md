# Learner 4 implementation (URF proposal simulation)

This document describes **Learner 4** only — the simulation cited in the Royal
Society URF proposal *Tuning to Learn* (figure `learner4-sim` /
`outputs/learner4/learner4_proposal.*`). Earlier learners (1–3) motivated the
design; they are not what the proposal figure regenerates.

**One-command regenerate** (canonical URF numbers: 20 seeds × 60 trials):

```bash
./scripts/run_learner4.sh --trials 60 --seeds 20
```

Outputs land in `outputs/learner4/`. The proposal panel is
`learner4_proposal.pdf` / `.png`.

---

## 1. Scientific claim being tested

People (and robots) co-contract when meeting a new dynamic object. The meeting
claim, restated for this experiment:

> If you do **not** co-contract, you must identify the full high-order plant.
> If you **do** co-contract, the limb–object coupling filters the fast wobbles,
> so you are effectively identifying a first-order plant. Is that second path
> faster and safer?

Learner 4 keeps the **same third-order object** in every condition, holds
**movement speed fixed**, and varies only co-contraction and model capacity.
That isolates the filter story from “just move more slowly.”

---

## 2. The physical system (plant)

File: `order_reduction/plant.py`

### 2.1 True object

A cascade of three real first-order lags (a cup-like object with one slow sway
and two fast transients):

\[
G(s)=\frac{1}{(1+\tau_d s)\,(1+\tau_1 s)\,(1+\tau_2 s)}
\]

with fixed truth

\[
(\tau_d,\tau_1,\tau_2)=(1.0,\;0.1,\;0.05)\,\mathrm{s}
\]

i.e. poles at \(-1\), \(-10\), \(-20\) rad/s. The object **never** changes
across conditions or trials.

### 2.2 What co-contraction does

Co-contraction enters as a damping ratio \(\xi\) of the coupled limb. It does
**not** rewrite the object; it changes the *felt* plant. Following the
singular-perturbation reduction used in the meeting note, overdamping separates
time scales by

\[
\rho(\xi)=
\begin{cases}
1 & \xi\le 1\\
\bigl(\xi+\sqrt{\xi^2-1}\bigr)^2 & \xi>1
\end{cases}
\]

Implemented as `rho(xi)` and `experienced_taus(taus, xi)`:

- dominant lag \(\tau_d\) is **unchanged**;
- fast lags become \(\tau_1/\rho\), \(\tau_2/\rho\).

Defaults in the experiment:

| regime | \(\xi\) | \(\rho\) (approx.) | felt plant |
|--------|---------|--------------------|------------|
| soft   | 0.7     | 1                  | all three modes visible |
| stiff  | 3.0     | ~34                | fast modes settle inside a fraction of the reach |

At \(\xi=3\), the felt plant collapses toward a single lag — that is the
filtering story in the proposal.

### 2.3 Discrete simulation

Trials use `dt = 0.01` s and `cascade_exact`, a zero-order-hold cascade
**without** the one-sample delay per stage that would floor reduced-order bias
measurements. Measurement noise on the felt force is Gaussian with
`NOISE_STD = 0.012`.

---

## 3. Hand trajectory generation

Every trial is the same class of voluntary movement: a short **out-and-back
minimum-jerk reach**, plus a little exploration.

### 3.1 Minimum-jerk reach

`min_jerk(t, t_move)` implements the classical Flash–Hogan quintic:

\[
s(\sigma)=10\sigma^3-15\sigma^4+6\sigma^5,\qquad \sigma=\mathrm{clip}(t/t_\mathrm{move},0,1)
\]

`reach_reference` builds one out-and-back: outward jerk, brief pause, reverse
jerk (negative amplitude). In Learner 4, **every** condition uses

```text
FAST_MOVE = t_move = 0.25 s
```

so bandwidth never stages. Any simplification must come from \(\xi\).

### 3.2 Band-limited exploration

`band_limited_noise` adds limb-plausible jitter: white noise filtered by a lag
with time constant \(t_\mathrm{move}/3\), then unit-scaled. Amplitude on the
command is `U_EXPLORE = 0.10`.

Why this matters: a white or impulsive probe would excite every mode on every
trial and erase the sample-complexity / filtering contrast. Voluntary movement
is smooth; energy near the 10 and 20 rad/s poles only appears if the reach is
fast **or** the limb is soft enough that those poles remain in the felt band.

Trial command:

```text
u = reach_reference(...) + U_EXPLORE * band_limited_noise(...)
y = cascade_exact(experienced_taus(TAU_TRUE, xi), u, dt) + noise
```

---

## 4. The learners (Kalman filters)

Learner 4 compares two estimators that share the **same update mathematics**;
they differ only in **what is estimated**.

Shared hyperparameters live in `order_reduction/kalman_learner.py`
(`PROCESS_NOISE`, `R_STD`, `IEKF_ITERS`, `THIN`, bounds, etc.).

### 4.1 Full-order learner: `TwoTimescaleEKF`

State (5-D):

\[
q=[\log\tau_d,\;\log\tau_1,\;\log\tau_2,\;w_1,\;w_2]
\]

Internal model of felt force given hand motion \(u\) and grip \(\xi\):

1. Pass \(u\) through the dominant lag \(\tau_d\).
2. Blend each transient with participation weight \(w_i\):

\[
\mathrm{blend}(\tau,w):\; x \mapsto (1-w)\,x + w\,\mathrm{lag}(\tau)\,x
\]

with felt transient time constant \(\tau_i/\rho(\xi)\).

**Why weights, not “set \(\tau\to 0\)”:** driving a time constant to zero also
kills its sensitivity, so pruning becomes irreversible. A weight keeps the
Jacobian alive at \(w=0\), so recruitment can happen when the limb softens.

**Automatic relevance (ARD):** unused weights are shrunk toward zero; effective
order is read off as

\[
n_\mathrm{eff}=1+w_1+w_2
\]

**Update:** iterated EKF in information form (`observe`):

- predict felt force \(h(q,u,\xi)\);
- finite-difference Jacobian, thinned by `THIN=5` after simulating at full `dt`;
- adaptive measurement variance \(R=\max(R_\mathrm{std}^2,\mathrm{mean}(e^2))\) so
  early model error does not make the filter overconfident;
- anisotropic process noise: tiny on \(\log\tau_d\), larger on fast modes/weights
  (slow well-retained dominant estimate, forgetful transients).

### 4.2 Slow-only learner: `FirstOrderEKF`

File: `order_reduction/slow_learner.py`

State is a **single** number, \(\log\tau_d\). The plant in the world is still
third-order; the model simply predicts

\[
\hat y = \mathrm{lag}(\hat\tau_d)\,u
\]

and ignores \(\xi\) inside \(h\) (the dominant lag is not compressed). When the
limb is stiff, true fast modes have already settled, so the residual is mostly
sensor noise and the one-parameter filter is nearly unbiased. When the limb is
soft, omitted modes leak into the residual: updates are cautious and
\(\hat\tau_d\) is biased high by roughly \((\tau_1+\tau_2)/\rho(\xi)\) at low
frequency (see `order_reduction/bias.py`).

Same IEKF step as the dominant channel of `TwoTimescaleEKF` — so L1 vs L2 is a
comparison of **hypothesis class**, not of optimiser cleverness.

### 4.3 Warm start (L6 promotion)

`_promote` copies \(\log\tau_d\) and its posterior variance into a fresh
`TwoTimescaleEKF`. Transient parameters and weights start from the usual prior.
L6 therefore gets a head start on the slow lag and nothing else.

---

## 5. Experiment design (L1–L6)

File: `order_reduction/experiment4.py`

| ID | Learner | Stiffness schedule | Role |
|----|---------|--------------------|------|
| **L1** | 3rd-order EKF | soft (0.7) throughout | no brace; learn full model |
| **L2** | 1st-order EKF | stiff (3.0) throughout | brace; learn slow lag only |
| **L3** | 1st-order EKF | soft throughout | control: capacity without filter |
| **L4** | 3rd-order EKF | stiff throughout | control: filter without restriction |
| **L5** | 3rd-order EKF | staged 3.0 → 1.3 → 0.7 | brace, then recruit full model |
| **L6** | 1st → 3rd (promote) | same staged schedule | slow lag first, then warm-start |

Human-facing strategies: **L1, L2, L5/L6**. L3/L4 are factorial controls.

### 5.1 Staging rule (L5/L6)

Stages advance only when **both** hold:

- at least `MIN_TRIALS_PER_STAGE = 5` trials in the current stage;
- posterior sd on \(\log\tau_d\) below `CONFIDENCE_SD = 0.04`.

L6 promotes from first- to third-order EKF at the first relaxation.

### 5.2 Failure / aborted trials

Predicted feel \(y_\mathrm{hat}\) vs measured \(y\). If
\(|y-y_\mathrm{hat}|\) exceeds `S_BASE * xi / XI_SOFT`, the trial aborts at the
first exceedance. Only the usable prefix (`≥ MIN_USABLE` samples) updates the
filter; otherwise the covariance drifts (`drift`). Soft wrong models fail more;
stiff schedules almost never fail.

### 5.3 Three scoreboards

| Name | Quantity | Criterion (3-trial streak) |
|------|----------|----------------------------|
| **slow** | \(\lvert\hat\tau_d-\tau_d\rvert\) | < 0.08 s |
| **full** | impulse error of belief about whole object | < 0.10 |
| **test** | relative error on a **common soft, fast probe** | < 0.10 |

The **test** probe (`make_test_probe`) is deterministic, noise-free, always at
\(\xi=0.7\), identical across seeds. “I can only do this while braced” shows up
as cost.

### 5.4 Failure-cost sweep

Raw simulation charges almost nothing for a fail beyond truncated data. The
script also reports `effective_ttc` under recovery costs
\(k\in\{0,0.5,1,2,3,5,8,12\}\) trials per failure (`failure_cost.pdf`). With any
nontrivial recovery cost, warm-started staging (L6) overtakes soft deep-end (L1)
on the soft-probe criterion.

---

## 6. What each trial does (loop sketch)

For each seed and condition, for trial \(n=1\ldots N\):

1. Choose \(\xi\) from the schedule (soft / stiff / staged).
2. Build hand command \(u\) (min-jerk + band-limited explore).
3. Simulate felt force through the **true** felt plant at that \(\xi\).
4. Predict \(\hat y\) from the current EKF belief.
5. Possibly abort early on surprise; observe on the usable prefix.
6. Log slow / full / test errors, bias, \(n_\mathrm{eff}\), failures.
7. If staged and confident enough, advance stage (and promote if L6).

Seeds are offset by condition so L1–L6 do not share RNG streams.

---

## 7. Regenerating URF results

### Dependencies

```bash
python -m pip install -r requirements.txt
# numpy, scipy, matplotlib
```

`scripts/run_learner4.sh` defaults to
`/opt/anaconda3/envs/pydrake_sfil/bin/python` if present; override with:

```bash
PYTHON=python3 ./scripts/run_learner4.sh --trials 60 --seeds 20
```

### Canonical URF run

```bash
./scripts/run_learner4.sh --trials 60 --seeds 20
# or:
python scripts/run_learner4.py --trials 60 --seeds 20
```

### Outputs written to `outputs/learner4/`

| File | Use |
|------|-----|
| `learner4_proposal.pdf` / `.png` | **URF figure** (two-panel summary) |
| `learner4.pdf` / `.png` | Full six-panel diagnostic |
| `failure_cost.pdf` / `.png` | Recovery-cost sweep |
| `results.json` | Medians, Holm tests, failure sweep |
| `learner4_snippet.tex` | Auto prose for pasting into LaTeX |

Quick smoke test (2 seeds, 16 trials):

```bash
./scripts/run_learner4.sh --smoke
# → outputs/_smoke_learner4/
```

### Expected headline numbers (20×60)

Medians: co-contracting L2/L5/L6 reach slow criterion in **3** trials vs **6**
for L1; failed attempts **0** vs **~4**; peak \(\lvert\tau_d\rvert\) error much
smaller under brace. Soft probe: L1 fastest if failures are free; L6 overtakes
once one failure costs ~1 recovery trial. See `README.md` (Learner 4 section)
and `results.json` for the full table.

### Copying into the URF proposal tree

After regeneration, the proposal figure historically lived at

```text
urf-proposal/figures/learner4-sim.pdf
```

Copy from this repo:

```bash
cp outputs/learner4/learner4_proposal.pdf /path/to/urf-proposal/figures/learner4-sim.pdf
cp outputs/learner4/learner4_proposal.png /path/to/urf-proposal/figures/learner4-sim.png
```

---

## 8. Code map (Learner 4 only)

```text
order_reduction/plant.py           object, rho(xi), min-jerk, band-limited noise
order_reduction/bias.py            closed-form (tau_1+tau_2)/rho bias law
order_reduction/kalman_learner.py  TwoTimescaleEKF (full-order)
order_reduction/slow_learner.py    FirstOrderEKF (slow lag only)
order_reduction/experiment4.py     L1–L6 trial loop, scoreboards, failure cost
order_reduction/plot4.py           learner4 / proposal / failure_cost figures
scripts/run_learner4.py            CLI, stats, write results + plots
scripts/run_learner4.sh            thin wrapper (PYTHON, Agg backend)
```

Optional block-diagram / slides (not required for the URF panel):

```bash
./scripts/build_learner4_block_diagram.sh
./scripts/build_learner4_slides.sh
```

---

## 9. What to claim from this implementation

**Supported**

- Bracing makes the felt plant effectively first-order; learning \(\tau_d\) is
  faster and safer than facing all three modes soft (L2/L5/L6 vs L1 on the slow
  scoreboard).
- The speed-up is the **coupling filter**, not “use a smaller model by fiat”
  (L2 ties L4; L3 fails).
- With any non-zero recovery cost for failed attempts, co-contract-then-recruit
  (especially L6) is competitive or better on the soft full-object probe.

**Not supported / do not claim**

- Staging is faster than soft deep-end **when failures are free**.
- Staying stiff forever eventually learns the soft object (L2 never passes the
  soft probe).
- Restricting capacity alone (without the physical filter) is enough (L3).
