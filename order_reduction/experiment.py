# -*- coding: utf-8 -*-
"""S1-S5: does reducing the effective order first pay for itself?

Every trial the learner makes one reach while holding a chosen co-contraction
level. Two things follow from that choice, and they pull in opposite
directions, which is what makes the hypothesis non-trivial:

* stiffening compresses the fast modes, so the trial is *safe* but carries
  almost no information about the transients;
* moving fast while soft exposes the transients, but if the learner's model is
  still wrong the resulting mismatch is large and the trial *fails* -- this is
  Thrish's "if stabilisation is not there the ball goes out, the task fails,
  there is no learning".

Failure is modelled as surprise exceeding what the current limb can absorb:

    surprise  = max_t | y(t) - y_hat(t) |
    tolerance = S_BASE * xi / XI_SOFT

so a stiff limb tolerates a large modelling error and a soft limb does not.
A failed trial is aborted at the moment the tolerance is crossed and yields
only the short segment felt before that, so it is not worthless -- it is just
a poor experiment. The learner therefore has to *earn* the right to relax:
only once the dominant mode is known are soft, fast trials survivable, and
only soft, fast trials reveal the transients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .learner import GrayBoxLearner
from .plant import (
    TAU_TRUE,
    XI_SOFT,
    band_limited_noise,
    cascade,
    experienced_taus,
    model_error,
    reach_reference,
)

DT = 0.01
T_TRIAL = 2.0
NOISE_STD = 0.012
U_EXPLORE = 0.10
S_BASE = 0.15
MIN_USABLE = 20
CRITERION = 0.10
CRITERION_STREAK = 3
MIN_TRIALS_PER_STAGE = 6
PLATEAU_EPS = 5e-4
PLATEAU_STREAK = 2

# (xi, movement time, number of unlocked time constants)
STAGES = ((3.0, 0.80, 1), (1.3, 0.45, 2), (XI_SOFT, 0.25, 3))
FAST_MOVE = 0.25

CONDITIONS = ("S1", "S2", "S3", "S4", "S5")

CONDITION_META = {
    "S1": {"label": "S1 deep end", "color": "#8C1428"},
    "S2": {"label": "S2 order recruitment", "color": "#1B365D"},
    "S3": {"label": "S3 stiff throughout", "color": "#5C5C5C"},
    "S4": {"label": "S4 bandwidth only", "color": "#1E6B3A"},
    "S5": {"label": "S5 random tuning", "color": "#C5761A"},
}


@dataclass
class TrialLog:
    err: np.ndarray          # relative impulse error of the object model
    pred_rmse: np.ndarray    # the learner's own fit residual (its Delta_n signal)
    jac_cond: np.ndarray     # conditioning of the identification at this stage
    xi: np.ndarray
    t_move: np.ndarray
    unlocked: np.ndarray
    failed: np.ndarray
    stage: np.ndarray
    ttc: int
    n_failed: int
    seed: int
    condition: str


def _setting(condition: str, stage: int, rng: np.random.Generator):
    """Return (xi, t_move, n_unlocked) for this condition and stage."""
    if condition == "S1":
        return XI_SOFT, FAST_MOVE, 3
    if condition == "S2":
        return STAGES[stage]
    if condition == "S3":
        return STAGES[0]
    if condition == "S4":
        _, t_move, n_unlocked = STAGES[stage]
        return XI_SOFT, t_move, n_unlocked
    if condition == "S5":
        return float(rng.uniform(XI_SOFT, 3.0)), FAST_MOVE, 3
    raise ValueError(condition)


def run_seed(condition: str, seed: int, n_trials: int = 60, dt: float = DT) -> TrialLog:
    rng = np.random.default_rng(seed)
    n_t = int(round(T_TRIAL / dt))
    stage = 0
    xi, t_move, n_unlocked = _setting(condition, stage, rng)
    learner = GrayBoxLearner(n_unlocked=n_unlocked, dt=dt)

    err = np.zeros(n_trials)
    pred_rmse = np.full(n_trials, np.nan)
    jac_cond = np.full(n_trials, np.nan)
    xi_log = np.zeros(n_trials)
    tm_log = np.zeros(n_trials)
    unlocked_log = np.zeros(n_trials, dtype=int)
    failed_log = np.zeros(n_trials, dtype=bool)
    stage_log = np.zeros(n_trials, dtype=int)

    streak = 0
    ttc = n_trials + 1
    trials_in_stage = 0
    plateau = 0
    prev_pred = None

    for n in range(n_trials):
        xi, t_move, n_unlocked = _setting(condition, stage, rng)
        learner.unlock(n_unlocked)

        ref = reach_reference(n_t, dt, t_move)
        u = ref + U_EXPLORE * band_limited_noise(n_t, dt, t_move, rng)
        y = cascade(experienced_taus(TAU_TRUE, xi), u, dt) + NOISE_STD * rng.normal(size=n_t)
        y_hat = learner.predict(u, xi)

        tolerance = S_BASE * xi / XI_SOFT
        over = np.flatnonzero(np.abs(y - y_hat) > tolerance)
        failed = over.size > 0
        stop = int(over[0]) if failed else n_t

        if stop >= MIN_USABLE:
            stats = learner.observe(u[:stop], y[:stop], xi)
            pred_rmse[n] = stats["pred_rmse"]
            jac_cond[n] = stats["jac_cond"]

        err[n] = model_error(tuple(learner.taus), dt=dt)
        xi_log[n] = xi
        tm_log[n] = t_move
        unlocked_log[n] = learner.n_unlocked
        failed_log[n] = failed
        stage_log[n] = stage

        if err[n] < CRITERION:
            streak += 1
            if streak >= CRITERION_STREAK and ttc == n_trials + 1:
                ttc = n + 1
        else:
            streak = 0

        # Stage advance is driven by the learner's own progress signal.
        if condition in ("S2", "S4") and stage < len(STAGES) - 1:
            trials_in_stage += 1
            usable = np.isfinite(pred_rmse[n])
            if usable and prev_pred is not None:
                delta = prev_pred - pred_rmse[n]
                plateau = plateau + 1 if delta < PLATEAU_EPS else 0
            if usable:
                prev_pred = pred_rmse[n]
            if trials_in_stage >= MIN_TRIALS_PER_STAGE and plateau >= PLATEAU_STREAK:
                stage += 1
                trials_in_stage = 0
                plateau = 0
                prev_pred = None

    return TrialLog(
        err=err,
        pred_rmse=pred_rmse,
        jac_cond=jac_cond,
        xi=xi_log,
        t_move=tm_log,
        unlocked=unlocked_log,
        failed=failed_log,
        stage=stage_log,
        ttc=int(ttc),
        n_failed=int(failed_log.sum()),
        seed=seed,
        condition=condition,
    )


def run_experiment(n_trials=60, n_seeds=20, seed0=0, conditions=CONDITIONS):
    out = {c: [] for c in conditions}
    for c in conditions:
        for i in range(n_seeds):
            out[c].append(
                run_seed(
                    c,
                    seed=seed0 + 1000 * (CONDITIONS.index(c) + 1) + i,
                    n_trials=n_trials,
                )
            )
    return out
