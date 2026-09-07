# Learner 4 — implementation guide (the URF / MV1 simulation)

> **This is the only learner cited in the Royal Society URF proposal**
> (*Tuning to Learn*, Method validation **MV1**).
> Code name: **Learner 4**. Proposal figure: `learner4-sim` /
> [`outputs/learner4/learner4_proposal.png`](outputs/learner4/learner4_proposal.png).
>
> Learners 1–3 are earlier development runs (not for the proposal figure).
> Short summaries and full write-ups: [`OTHER_LEARNERS.md`](OTHER_LEARNERS.md).

**Regenerate (canonical: 20 seeds × 60 trials):**

```bash
./scripts/run_learner4.sh --trials 60 --seeds 20
```

---

## 0. Consistency with the proposal (MV1)

Proposal wording (paraphrased from MV1):

> Before recruiting participants, the twin and filter pipeline was checked on
> the same third-order plant as §1. An **extended Kalman filter (EKF)** learned
> the plant’s time constants across trials. Four strategies were compared.
> **S1** soft from trial 1 with a full model. **S2** braced throughout,
> learning only the dominant dynamics. **S3** braced first, then staged release
> into a full soft model. **S4** soft like S1 but forced to learn only the
> dominant dynamics. Braced strategies reached the dominant dynamics in half
> the trials with zero failures; only the staged-release strategy also passed
> a later soft probe. *This confirms the twin can separate braced from soft
> curricula; it is not evidence that humans do the same.*

### Proposal label ↔ code condition

Defined in `order_reduction/plot4.py` as `PROPOSAL_STRATEGIES`:

| Proposal | Code ID | What it does |
|----------|---------|--------------|
| **S1** | **L1** | Soft (\(\xi=0.7\)), full third-order EKF — *deep end* |
| **S2** | **L2** | Braced (\(\xi=3\)), first-order EKF on \(\tau_d\) only — *brace and keep* |
| **S3** | **L6** | Braced then staged release; warm-start full EKF from converged \(\tau_d\) — *brace then hand over* |
| **S4** | **L3** | Soft, first-order EKF — *control: ignore fast modes without the filter* |

Code also runs **L4** (braced + full EKF) and **L5** (staged full EKF without
explicit warm-start). They are controls for the paper’s mechanism story; they
are **not** drawn in the MV1 / proposal panel.

---

## 1. Scientific claim being tested

Carry a cup-like object: one **slow** (dominant) mode and two **fast**
transients. Soft arm → all three arrive together. Co-contract → fast poles
settle inside a fraction of the reach, so the felt plant collapses toward a
single lag.

Question for Learner 4:

> If you do **not** co-contract, you must identify the full high-order plant.
> If you **do** co-contract, you are effectively identifying a first-order
> plant. Is that second path faster and safer? Does staged release recover soft
> skill afterward?

Movement time is fixed at the fast reach (`t_move = 0.25` s) in every cell, so
the answer cannot be “just move more slowly.”

---

## 2. The physical system (plant)

File: [`order_reduction/plant.py`](order_reduction/plant.py)

### 2.1 True object

\[
G(s)=\frac{1}{(1+\tau_d s)\,(1+\tau_1 s)\,(1+\tau_2 s)},
\qquad (\tau_d,\tau_1,\tau_2)=(1.0,\,0.1,\,0.05)\,\mathrm{s}
\]

Poles at \(-1\), \(-10\), \(-20\) rad/s. The object **never** changes.

### 2.2 Co-contraction → felt plant

Co-contraction \(\xi\) is the damping ratio of the coupled limb. It does not
rewrite the object; it changes what the learner **feels**:

\[
\rho(\xi)=
\begin{cases}
1 & \xi\le 1\\
\bigl(\xi+\sqrt{\xi^2-1}\bigr)^2 & \xi>1
\end{cases}
\]

`experienced_taus`: \(\tau_d\) unchanged; \(\tau_1/\rho\), \(\tau_2/\rho\).

| Regime | \(\xi\) | \(\rho\) | Felt plant |
|--------|---------|----------|------------|
| Soft | 0.7 | 1 | all three modes visible |
| Braced | 3.0 | ~34 | fast modes settle inside a fraction of the reach |

### 2.3 Discrete simulation

`dt = 0.01` s, `cascade_exact` (no one-sample delay floor), measurement noise
`NOISE_STD = 0.012`.

---

## 3. Hand trajectory generation

Every trial:

1. **Minimum-jerk** out-and-back reach (`min_jerk` / `reach_reference`) —
   Flash–Hogan quintic, fixed duration `FAST_MOVE = 0.25` s.
2. **Band-limited exploration** (`band_limited_noise`) — white noise filtered
   at \(t_\mathrm{move}/3\), scaled and added at amplitude `U_EXPLORE = 0.10`.

Why band-limited: a white/impulsive probe would excite every mode every trial
and erase the filtering contrast. Voluntary movement is smooth; energy near the
fast poles only appears if the reach is fast **and** the limb is soft enough
that those poles remain in the felt band.

```text
u = reach_reference(...) + U_EXPLORE * band_limited_noise(...)
y = cascade_exact(experienced_taus(TAU_TRUE, xi), u, dt) + noise
```

---

## 4. The learners (Kalman filters)

Shared IEKF hyperparameters: [`order_reduction/kalman_learner.py`](order_reduction/kalman_learner.py).

### 4.1 Full-order: `TwoTimescaleEKF` (used by S1 / L1, and after hand-over in S3)

State:

\[
q=[\log\tau_d,\;\log\tau_1,\;\log\tau_2,\;w_1,\;w_2]
\]

Felt prediction: dominant lag, then each transient via participation weight

\[
\mathrm{blend}(\tau,w):\; x \mapsto (1-w)\,x + w\,\mathrm{lag}(\tau)\,x
\]

with felt \(\tau_i/\rho(\xi)\). **Weights** (not \(\tau\to0\)) keep pruning
reversible. ARD shrinks unused weights; \(n_\mathrm{eff}=1+w_1+w_2\).

Update: iterated EKF in information form; adaptive
\(R=\max(R_\mathrm{std}^2,\mathrm{mean}(e^2))\); anisotropic process noise
(tiny on \(\log\tau_d\), larger on fast channels).

### 4.2 Dominant-only: `FirstOrderEKF` (S2 / L2, S4 / L3, early S3 / L6)

File: [`order_reduction/slow_learner.py`](order_reduction/slow_learner.py)

State is only \(\log\tau_d\). World is still third-order. When braced, omitted
modes have already settled → nearly unbiased. When soft, bias ≈
\((\tau_1+\tau_2)/\rho(\xi)\) (`bias.py`) — that is why S4 fails.

### 4.3 Warm start (S3 = L6)

`_promote` copies \(\log\tau_d\) and its posterior variance into a fresh
`TwoTimescaleEKF`. Transients start from the usual prior. Proposal “hand over”
= this promotion at the first confidence-gated relaxation.

---

## 5. Experiment design

File: [`order_reduction/experiment4.py`](order_reduction/experiment4.py)

| Code | Learner | \(\xi\) schedule | Proposal? |
|------|---------|------------------|-----------|
| L1 | 3rd-order EKF | soft | **S1** |
| L2 | 1st-order EKF | braced | **S2** |
| L3 | 1st-order EKF | soft | **S4** |
| L4 | 3rd-order EKF | braced | control |
| L5 | 3rd-order EKF | staged 3→1.3→0.7 | control |
| L6 | 1st→3rd promote | staged | **S3** |

**Staging (L5/L6):** advance when ≥5 trials in stage **and** posterior sd on
\(\log\tau_d\) < 0.04.

**Failure:** abort when \(|y-\hat y|\) exceeds `S_BASE * xi / XI_SOFT`; only
usable prefix updates the filter.

**Scoreboards:**

| Name | Measure | Criterion |
|------|---------|-----------|
| slow / dominant | \(\lvert\hat\tau_d-\tau_d\rvert\) | < 0.08 s (3-trial streak) |
| test / soft probe | relative error on common soft fast reach | < 0.10 |
| full | impulse error of whole-object belief | < 0.10 |

The soft probe is deterministic, noise-free, always at \(\xi=0.7\) — so “I can
only do this while braced” shows up as cost.

---

## 6. Results with figures

Canonical numbers: 20 seeds × 60 trials (`results.json`).

### 6.1 Proposal panel (S1–S4) — what MV1 shows

![Proposal panel](outputs/learner4/learner4_proposal.png)

| Panel | Readout |
|-------|---------|
| Left | Dominant-mode error. Braced S2/S3 under criterion by trial ~3; S1 later after a large transient; S4 never (bias floor). |
| Middle | Soft-probe skill. S1 and S3 → ~0.007; S2 stuck ~0.22; S4 stays high. |
| Right | Speed & safety. Trials to dominant: 6 / 3 / 3 / never. Failures: ~4 / 0 / 0 / ~16. |

**MV1 sentence this figure supports:** braced strategies reach the dominant
dynamics in half the trials with zero failures; only staged release (S3) also
passes a later soft probe.

Relative to S1 (medians): S2/S3 cut trials to \(|\hat\tau_d-\tau_d|<0.08\) by
**50%** (3 vs 6), failed attempts by **100%**, peak lag error by ~**97%**.

### 6.2 Failure-cost sweep — conditional claim on soft skill

![Failure cost](outputs/learner4/failure_cost.png)

Left: effective trials to the soft-probe criterion vs recovery cost of one
failure. Soft deep-end (L1/S1) wins only while failure is free; L6/S3 is flat
at 8 and overtakes once cost ≥ 1. Right: only L1 spends failed attempts (~4);
braced / staged spend zero.

### 6.3 Full L1–L6 diagnostic — mechanism controls

![Full diagnostic](outputs/learner4/learner4.png)

| Panel | Why it matters |
|-------|----------------|
| Slow scoreboard | Same story as proposal left, with L4/L5 included |
| Soft-probe scoreboard | S2/L2 and L4 never pass while stiff |
| Full-object impulse | Scoreboard Learners 1–3 used (stiff first-order plateaus ~0.4) |
| \(\xi(t)\) | Only stiffness is staged; speed fixed |
| \(n_\mathrm{eff}\) | Order recruitment follows release of co-contraction |
| Bars | Trials to slow vs test + failures |

**L4 ≈ L2 on the slow scoreboard:** full-capacity learner held braced prunes
itself to \(n_\mathrm{eff}=1\) — the **coupled plant** simplifies learning, not
a hand-coded model restriction. **L3 = S4** never reaches criterion: omitting
fast modes without bracing is misspecification. **L6 beats L5** on the soft
probe by warm-starting \(\tau_d\).

---

## 7. Trial loop (sketch)

For each seed × condition × trial:

1. Choose \(\xi\) (soft / braced / staged).
2. Build \(u\) (min-jerk + band-limited explore).
3. Simulate felt \(y\) through true felt plant at that \(\xi\).
4. Predict \(\hat y\) from current EKF belief.
5. Maybe abort on surprise; `observe` on usable prefix (else `drift`).
6. Log slow / test / full errors, bias, \(n_\mathrm{eff}\), failures.
7. If staged and confident, advance stage (promote if L6/S3).

---

## 8. Code map

```text
order_reduction/plant.py           object, rho(xi), min-jerk, band-limited noise
order_reduction/bias.py            (tau_1+tau_2)/rho bias law
order_reduction/kalman_learner.py  TwoTimescaleEKF (full-order)
order_reduction/slow_learner.py    FirstOrderEKF (dominant only)
order_reduction/experiment4.py     L1–L6 loop, scoreboards, failure cost
order_reduction/plot4.py           proposal / full / failure_cost figures
scripts/run_learner4.py            CLI + stats
scripts/run_learner4.sh            wrapper
```

Copy into the proposal tree:

```bash
cp outputs/learner4/learner4_proposal.pdf /path/to/urf-proposal/figures/learner4-sim.pdf
```

---

## 9. What to claim / not claim (proposal-safe)

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
