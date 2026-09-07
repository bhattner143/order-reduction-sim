# Learner 4 — comprehensive implementation guide

> **This is the only learner cited in the Royal Society URF proposal**
> (*Tuning to Learn*, Method validation **MV1**).
> Code name: **Learner 4**. Proposal figure:
> [`outputs/learner4/learner4_proposal.png`](outputs/learner4/learner4_proposal.png).
>
> Learners 1–3: [`OTHER_LEARNERS.md`](OTHER_LEARNERS.md) (not for the proposal figure).

**Theory PDF (derivations, \(\rho(\xi)\), bias law, worked EKF notes):**
[`report/theory.pdf`](report/theory.pdf) · build notes in [`report/README.md`](report/README.md).

**Regenerate (canonical: 20 seeds × 60 trials):**

```bash
./scripts/run_learner4.sh --trials 60 --seeds 20
```

**Optional one-trial block diagram:**

```bash
./scripts/build_learner4_block_diagram.sh
# → outputs/learner4/learner4_block_diagram.pdf
```

Annotated home-page figure (also on the README):
[`outputs/learner4/learner4-complete-block.png`](outputs/learner4/learner4-complete-block.png)
· SVG: [`outputs/learner4/learner-4-block-diagram-aranged.svg`](outputs/learner4/learner-4-block-diagram-aranged.svg).

**Short presentation:** [`outputs/learner4/learner4_slides.pdf`](outputs/learner4/learner4_slides.pdf)
(`./scripts/build_learner4_slides.sh`).

---

## Contents

1. [Proposal / MV1 mapping](#1-proposal--mv1-mapping)
2. [Physical system](#2-physical-system)
3. [Hand motion generation](#3-hand-motion-generation)
4. [What the learner feels](#4-what-the-learner-feels)
5. [Misspecification bias law](#5-misspecification-bias-law)
6. [EKF implementation](#6-ekf-implementation)
7. [Experiment design (L1–L6 / S1–S4)](#7-experiment-design-l1l6--s1s4)
8. [Per-trial loop](#8-per-trial-loop)
9. [Scoreboards, criteria, failure cost](#9-scoreboards-criteria-failure-cost)
10. [Results and figures](#10-results-and-figures)
11. [Hyperparameters cheat-sheet](#11-hyperparameters-cheat-sheet)
12. [Code map](#12-code-map)
13. [What to claim / not claim](#13-what-to-claim--not-claim)

---

## 1. Proposal / MV1 mapping

Proposal strategies (MV1 wording) map to code conditions via
`PROPOSAL_STRATEGIES` in [`order_reduction/plot4.py`](order_reduction/plot4.py):

| Proposal | Code | Strategy |
|----------|------|----------|
| **S1** | **L1** | Soft from trial 1, **full** third-order EKF |
| **S2** | **L2** | Braced throughout, **dominant-only** first-order EKF |
| **S3** | **L6** | Braced first, then staged release + **warm-start hand-over** into full EKF |
| **S4** | **L3** | Soft like S1, but **forced** dominant-only (control) |

Extra cells (not drawn in the proposal panel):

| Code | Role |
|------|------|
| **L4** | Braced + full EKF — filter present, capacity unused (shows the *plant* simplifies) |
| **L5** | Staged full EKF without explicit warm-start (shows hand-over value of L6/S3) |

MV1 result line this implementation is built to produce: braced strategies reach
the **dominant dynamics** in half the trials with **zero** failures; only
**staged release (S3)** also passes a later **soft probe**.

---

## 2. Physical system

File: [`order_reduction/plant.py`](order_reduction/plant.py) · theory:
[`report/theory.pdf`](report/theory.pdf) (MSD → \(\rho(\xi)\) → third-order cup).

### 2.1 True object (never changes)

A cascade of three real first-order lags (cup-like: one slow sway, two fast
transients):

\[
G(s)=\frac{1}{(1+\tau_d s)\,(1+\tau_1 s)\,(1+\tau_2 s)},
\qquad (\tau_d,\tau_1,\tau_2)=(1.0,\,0.1,\,0.05)\,\mathrm{s}.
\]

Poles at \(-1\), \(-10\), \(-20\) rad/s. Constants: `TAU_TRUE`.

### 2.2 Discrete lag cascade

Each continuous lag \(1/(1+\tau s)\) is discretised at sample time `dt` as

\[
y[k] = a\,y[k-1] + (1-a)\,u[k],\qquad a=e^{-\mathrm{dt}/\tau}.
\]

`cascade_exact` chains three such stages **without** the one-sample delay per
stage that `cascade` had. That delay would floor any reduced-order bias
measurement (~`3·dt`); Learner 4 (and the bias law) therefore use
`cascade_exact`.

A stage with \(\tau\le 10^{-6}\) is a pass-through (already settled).

### 2.3 Co-contraction and the felt plant

Co-contraction enters as the limb damping ratio \(\xi\). It does **not** rewrite
the object; it changes the **coupled** plant the learner experiences.

Overdamped separation ratio (`rho`):

\[
\rho(\xi)=
\begin{cases}
1 & \xi \le 1,\\
\bigl(\xi+\sqrt{\xi^{2}-1}\bigr)^{2} & \xi > 1.
\end{cases}
\]

`experienced_taus(taus, xi)` returns \((\tau_d,\,\tau_1/\rho,\,\tau_2/\rho)\):
the dominant lag is untouched; fast lags are compressed.

| Regime | \(\xi\) | \(\rho\) (approx.) | Effect |
|--------|---------|--------------------|--------|
| Soft (`XI_SOFT`) | 0.7 | 1 | all three modes visible in the reach |
| Braced (`XI_STIFF`) | 3.0 | ~34 | fast modes settle inside a fraction of the reach |

At \(\xi=3\), the felt plant collapses toward a single lag — the filtering
story in the proposal.

---

## 3. Hand motion generation

Same file: [`order_reduction/plant.py`](order_reduction/plant.py).  
In Learner 4, **every** condition uses the **same** movement duration
(`FAST_MOVE = 0.25` s in [`experiment4.py`](order_reduction/experiment4.py)).
Only \(\xi\) (and model capacity) vary.

### 3.1 Minimum-jerk reach (voluntary path)

Flash–Hogan quintic (`min_jerk`):

\[
s(\sigma)=10\sigma^{3}-15\sigma^{4}+6\sigma^{5},\qquad
\sigma=\mathrm{clip}(t/t_{\mathrm{move}},0,1).
\]

`reach_reference(n, dt, t_move)` builds one **out-and-back**:

1. outward min-jerk of amplitude \(+1\) over \(t_{\mathrm{move}}\);
2. brief pause (~0.35 s);
3. reverse min-jerk of amplitude \(-1\).

Trial length is `T_TRIAL = 2.5` s (`n_t = T_TRIAL/dt` samples).

### 3.2 Band-limited exploration

`band_limited_noise(n, dt, t_move, rng)`:

1. draw white noise;
2. filter with a first-order lag of time constant \(t_{\mathrm{move}}/3\);
3. unit-normalise by standard deviation.

Trial command:

\[
u = u_{\mathrm{reach}} + U_{\mathrm{explore}}\,w_{\mathrm{band}},
\qquad U_{\mathrm{explore}}=0.10.
\]

**Why this matters.** A limb cannot emit an impulse or white noise. If it
could, every mode would be excited every trial and the filtering / sample-
complexity contrast would be void. Smooth voluntary motion only puts energy
near the 10 and 20 rad/s poles when the reach is fast **and** the limb is soft
enough that those poles remain in the felt band.

### 3.3 Soft-probe trajectory (evaluation only)

`make_test_probe` builds one **deterministic**, **noise-free** soft reach at
\(\xi=0.7\), identical for every seed and condition. It is never used as the
training command; it is the fair “can you do this unbraced?” scoreboard.

---

## 4. What the learner feels

On each training trial:

```text
u  = reach + explore
y  = cascade_exact(experienced_taus(TAU_TRUE, xi), u, dt) + NOISE_STD * N(0,1)
ŷ  = learner.predict(u, xi)
```

So:

- the **world** always has three lags, possibly compressed by \(\rho(\xi)\);
- the **sensor** adds Gaussian noise (`NOISE_STD = 0.012`);
- the **learner** predicts felt force from its current belief and the grip it
  knows it is holding.

Prediction residual \(y-\hat y\) drives both learning and the failure rule
(§8).

---

## 5. Misspecification bias law

File: [`order_reduction/bias.py`](order_reduction/bias.py) · theory PDF section
on \((\tau_1+\tau_2)/\rho\).

A learner that fits **fewer** modes than the world has is biased. Low-frequency
expansion:

\[
G(s)=1-(\tau_d+\tau_1+\tau_2)s+O(s^{2}),
\qquad
\frac{1}{1+\hat\tau s}=1-\hat\tau s+O(s^{2}),
\]

so matching gives \(\hat\tau=\tau_d+\tau_1+\tau_2\) when soft. Under
co-contraction the fast lags are compressed:

\[
\hat\tau(\xi)=\tau_d+\frac{\tau_1+\tau_2}{\rho(\xi)},
\qquad
\mathrm{bias}(\xi)=\frac{\tau_1+\tau_2}{\rho(\xi)}.
\]

At \(\xi=0.7\), \(\rho=1\), bias floor ≈ \(0.15\) s (plus finite-speed probe
effects → ~0.22 s on the soft fast reach). At \(\xi=3\), \(\rho\approx34\),
bias collapses. **That is why S4 (soft + dominant-only) fails and S2 (braced +
dominant-only) succeeds** — same estimator, different filter on the plant.

---

## 6. EKF implementation

Learner 4 compares two estimators that share the **same update mathematics**;
they differ only in **what is estimated**. Hyperparameters live in
[`order_reduction/kalman_learner.py`](order_reduction/kalman_learner.py).

### 6.1 Discrete lag used inside the filter

Same form as the plant:

```python
a = exp(-dt / tau)
y = lfilter([1 - a], [1, -a], u)   # scipy.signal
```

### 6.2 Full-order learner: `TwoTimescaleEKF`

Used by **S1/L1**, **L4**, **L5**, and **after promotion in S3/L6**.

**State (5-D):**

\[
q=\bigl[\log\tau_d,\;\log\tau_1,\;\log\tau_2,\;w_1,\;w_2\bigr].
\]

Working in log-\(\tau\) keeps positivity and stabilises relative updates.

**Internal model** `_h(q, u, xi)`:

1. pass \(u\) through the dominant lag \(\tau_d=\exp(q_0)\);
2. blend each transient with participation weight \(w_i\):

\[
\mathrm{blend}(\tau,w):\;
x \mapsto (1-w)\,x + w\,\mathrm{lag}(\tau)\,x,
\]

with felt time constant \(\tau_i/\rho(\xi)\).

**Why weights, not “set \(\tau\to0\)”.** Driving a time constant to zero also
kills its sensitivity, so pruning becomes irreversible. A weight keeps the
Jacobian alive at \(w=0\), so recruitment can happen when the limb softens.

**Effective order:** \(n_{\mathrm{eff}}=1+w_1+w_2\) (read off the posterior).

**Initialisation:**

| Channel | Init | \(P_0\) | Process noise \(Q\) |
|---------|------|---------|---------------------|
| \(\log\tau_d\) | \(\log 0.30\) | 0.50 | \(10^{-5}\) (slow, retained) |
| \(\log\tau_{1,2}\) | \(\log 0.05\), \(\log 0.02\) | 0.50 | \(2\cdot10^{-3}\) (forgetful) |
| \(w_{1,2}\) | 0.25 | 0.25 | \(2\cdot10^{-3}\) |

**ARD prior on weights.** Precision \(\alpha\) on \(w_i\) shrinks unused modes
toward \(q_{\mathrm{target}}=0\). After each update:

\[
\gamma_i=1-\alpha_i P_{ii},\qquad
\alpha_i \leftarrow \mathrm{clip}\bigl(\gamma_i / \max(w_i^{2},10^{-4}),\,10^{-3},\,10^{3}\bigr).
\]

**Iterated EKF (information form), `observe`:**

For `IEKF_ITERS = 3` iterations:

1. Predict \(h_0=h(q,u,\xi)\) at full sample rate.
2. Finite-difference Jacobian \(H=\partial h/\partial q\) (`FD_EPS=1e-3`),
   then **thin** residual and \(H\) by `THIN=5` (simulate full-rate, thin after —
   thinning the input first would break the discrete lag).
3. Adaptive measurement variance
   \(R=\max(R_{\mathrm{std}}^{2},\,\mathrm{mean}(e^{2}))\) with
   `R_STD=0.02`, so early model error does not make the filter overconfident.
4. Information update with prior and ARD:

\[
A = P_{\mathrm{pred}}^{-1} + H^{\mathsf T}H/R + \mathrm{diag}(\alpha),
\qquad
P_{\mathrm{post}}=A^{-1},
\]

\[
\Delta q = P_{\mathrm{post}}\Bigl(
  H^{\mathsf T}e/R
  -\alpha\odot(q-q_{\mathrm{target}})
  -P_{\mathrm{pred}}^{-1}(q-q_{\mathrm{pred}})
\Bigr),
\]

clipped per component by `MAX_STEP=0.4` and to log-\(\tau\) / weight bounds.

5. Set \(P \leftarrow P_{\mathrm{post}}+P_{\mathrm{FLOOR}}I\) (`P_FLOOR=1e-4`).

**`drift`:** on trials too short to learn, \(P\leftarrow P+Q\) (uncertainty grows,
estimate unchanged).

### 6.3 Dominant-only learner: `FirstOrderEKF`

File: [`order_reduction/slow_learner.py`](order_reduction/slow_learner.py).  
Used by **S2/L2**, **S4/L3**, and **early S3/L6**.

**State:** single scalar \(q=[\log\tau_d]\).

**Model:** \(\hat y=\mathrm{lag}(\hat\tau_d)\,u\). The argument \(\xi\) is
accepted for API compatibility but **unused** inside \(h\) — the dominant lag
is not compressed by co-contraction.

**Update:** identical IEKF step on that one channel (same `PROCESS_NOISE[0]`,
`P_INIT[0]`, `R_STD`, `THIN`, `IEKF_ITERS`, `MAX_STEP`). No ARD term.

So L1 vs L2 is a comparison of **hypothesis class**, not of optimiser
cleverness.

### 6.4 Warm-start hand-over (S3 = L6)

`_promote(first_order, dt)` in [`experiment4.py`](order_reduction/experiment4.py):

1. construct a fresh `TwoTimescaleEKF`;
2. copy \(\log\tau_d\) and \(P_{\tau_d}\) from the first-order filter;
3. leave transient \(\tau\)’s and weights at their default prior.

Proposal “brace then hand over” = confidence-gated stage advance **plus** this
promotion at the first relaxation.

---

## 7. Experiment design (L1–L6 / S1–S4)

File: [`order_reduction/experiment4.py`](order_reduction/experiment4.py).

| Code | Estimator | \(\xi\) schedule | Proposal |
|------|-----------|------------------|----------|
| L1 | `TwoTimescaleEKF` | soft 0.7 | **S1** |
| L2 | `FirstOrderEKF` | braced 3.0 | **S2** |
| L3 | `FirstOrderEKF` | soft 0.7 | **S4** |
| L4 | `TwoTimescaleEKF` | braced 3.0 | control |
| L5 | `TwoTimescaleEKF` | staged 3.0 → 1.3 → 0.7 | control |
| L6 | `FirstOrderEKF` → promote | same staged | **S3** |

**Staging rule (L5/L6):** advance when both hold:

- `trials_in_stage >= MIN_TRIALS_PER_STAGE` (5);
- posterior sd on \(\log\tau_d\) < `CONFIDENCE_SD` (0.04).

**Seeds:** `seed0 + 1000*(condition_index+1) + i` so L1–L6 do not share RNG
streams.

---

## 8. Per-trial loop

For each condition × seed × trial \(n=0\ldots N-1\):

1. **Schedule** \(\xi\) from soft / braced / staged stage.
2. **Command** \(u=\) min-jerk reach + band-limited explore.
3. **World** \(y=\) felt plant at that \(\xi\) + sensor noise.
4. **Predict** \(\hat y=\) `learner.predict(u, xi)`.
5. **Failure check.** Tolerance `S_BASE * xi / XI_SOFT` (`S_BASE=0.15`). If
   \(|y-\hat y|\) exceeds it at any sample, abort at first exceedance
   (`failed=True`). Soft wrong models fail more; braced schedules almost never.
6. **Learn or drift.** If usable length `≥ MIN_USABLE` (20 samples), call
   `learner.observe(u[:stop], y[:stop], xi)`; else `learner.drift()`.
7. **Log** slow / test / full errors, \(\hat\tau_d-\tau_d\), \(n_{\mathrm{eff}}\),
   \(\mathrm{sd}(\log\tau_d)\), \(\xi\), failure flag.
8. **Update TTC** with 3-trial streak criteria (§9).
9. **Maybe advance stage** (and promote if L6).

Block-diagram PDF of this loop:
[`outputs/learner4/learner4_block_diagram.pdf`](outputs/learner4/learner4_block_diagram.pdf)
(after building).

---

## 9. Scoreboards, criteria, failure cost

| Name | Quantity | Criterion (streak ≥ 3) |
|------|----------|------------------------|
| **slow / dominant** | \(\lvert\hat\tau_d-\tau_d\rvert\) | < `SLOW_CRITERION` 0.08 s |
| **test / soft probe** | relative RMSE predicting the common soft fast probe | < `TEST_CRITERION` 0.10 |
| **full** | relative impulse error of belief vs true object | < `FULL_CRITERION` 0.10 |

**Soft probe** (`test_error`): always evaluate `predict(probe_u, XI_SOFT)` against
the fixed soft truth — “I can only do this while braced” shows up as cost.

**Failure-cost sweep** (`effective_ttc`): raw simulation barely charges failures
beyond truncated data. Recompute trials-to-soft-probe as

\[
\mathrm{TTC}_{\mathrm{eff}}=\mathrm{TTC}+k\cdot(\#\text{failures before TTC}),
\]

for \(k\in\{0,0.5,1,2,3,5,8,12\}\). Break-even: L6/S3 overtakes L1/S1 at
\(k=1\); L5 at \(k=5\) (`break_even_cost` in results).

Stats: Holm-corrected two-sided Mann–Whitney on the contrasts in
[`scripts/run_learner4.py`](scripts/run_learner4.py).

---

## 10. Results and figures

Canonical run: 20 seeds × 60 trials → `outputs/learner4/`.

### Proposal panel (S1–S4)

![Proposal panel](outputs/learner4/learner4_proposal.png)

| Panel | Meaning |
|-------|---------|
| Left | Dominant-mode error — braced S2/S3 fast; S1 slower; S4 never |
| Middle | Soft-probe skill — S1 & S3 succeed; S2 stuck ~0.22; S4 high |
| Right | Trials to dominant (6/3/3/never) and failures (~4/0/0/~16) |

Relative to S1: S2/S3 cut dominant TTC by **50%**, failures by **100%**, peak
lag error by ~**97%**.

### Failure cost

![Failure cost](outputs/learner4/failure_cost.png)

Deep end wins the soft probe only while failure is free; S3/L6 is flat and wins
once recovery costs ≥ 1 trial.

### Full L1–L6 diagnostic

![Full diagnostic](outputs/learner4/learner4.png)

Controls: **L4≈L2** on slow scoreboard (plant does the simplifying);
**L3=S4** never reaches criterion; **L6>L5** on soft probe via warm-start.

---

## 11. Hyperparameters cheat-sheet

| Symbol / name | Value | Where |
|---------------|-------|--------|
| `dt` | 0.01 s | experiment / learners |
| `T_TRIAL` | 2.5 s | experiment4 |
| `FAST_MOVE` / `t_move` | 0.25 s | fixed for all L1–L6 |
| `U_EXPLORE` | 0.10 | command jitter amplitude |
| `NOISE_STD` | 0.012 | sensor on \(y\) |
| `XI_SOFT` / `XI_STIFF` | 0.7 / 3.0 | plant / schedules |
| `XI_STAGES` | (3.0, 1.3, 0.7) | staged L5/L6 |
| `MIN_TRIALS_PER_STAGE` | 5 | staging dwell |
| `CONFIDENCE_SD` | 0.04 | sd(\(\log\tau_d\)) to relax |
| `S_BASE` | 0.15 | failure tolerance scale |
| `MIN_USABLE` | 20 samples | min length to `observe` |
| `SLOW_CRITERION` | 0.08 s | dominant TTC |
| `TEST_CRITERION` | 0.10 | soft-probe TTC |
| `CRITERION_STREAK` | 3 | consecutive trials under threshold |
| `IEKF_ITERS` | 3 | kalman / slow learner |
| `THIN` | 5 | residual / Jacobian decimation |
| `R_STD` | 0.02 | floor for adaptive \(R\) |
| `MAX_STEP` | 0.4 | per-iter \(\Delta q\) clip |
| `P_FLOOR` | 1e-4 | covariance ridge |
| `FD_EPS` | 1e-3 | finite-difference Jacobian |

---

## 12. Code map

```text
order_reduction/plant.py           G(s), rho(xi), cascade_exact, min-jerk, band-limited noise
order_reduction/bias.py            (tau_1+tau_2)/rho closed form + LS check
order_reduction/kalman_learner.py  TwoTimescaleEKF (full-order IEKF + ARD)
order_reduction/slow_learner.py    FirstOrderEKF (dominant-only IEKF)
order_reduction/experiment4.py     L1–L6 loop, promote, scoreboards, failure cost
order_reduction/plot4.py           proposal / full / failure_cost figures + S1–S4 map
order_reduction/stats.py           Holm correction
scripts/run_learner4.py            CLI, Mann–Whitney, write results
scripts/run_learner4.sh            PYTHON + MPLBACKEND=Agg wrapper
report/theory.pdf                  full theory write-up (read this for derivations)
report/theory.tex                  source for the PDF
```

Copy proposal figure into the URF tree:

```bash
cp outputs/learner4/learner4_proposal.pdf /path/to/urf-proposal/figures/learner4-sim.pdf
```

Build theory PDF:

```bash
cd report && pdflatex theory.tex && pdflatex theory.tex
```

---

## 13. What to claim / not claim

**Claim (matches MV1):**

- Braced S2/S3 reach the dominant dynamics in half the trials with zero failures.
- Only staged-release S3 also passes a later soft probe.
- Soft dominant-only S4 fails: “ignore fast modes” needs the physical filter.
- Twin check only — not human evidence.

**Do not claim:**

- Humans already use this curriculum.
- Staying braced forever yields soft skill (S2 fails the soft probe).
- Staging beats soft deep-end when spills are free (raw soft-probe TTC).
- Restricting model order alone (without bracing) is enough (S4/L3).
