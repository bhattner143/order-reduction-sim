# -*- coding: utf-8 -*-
"""K1-K5: the same five tuning schedules, run with learner 2.

Nothing here tells the learner what order to use. Learner 1 was handed its
model order at each stage boundary, which meant condition S3 failed by
definition rather than by discovery. Here every condition carries all three
modes at all times and the effective order is whatever the ARD prior retains,
so "staying stiff keeps you at first order" is a result the learner has to
produce on its own.

Relaxation is also earned rather than scheduled. K2 and K4 advance when the
posterior standard deviation on the dominant log time constant falls below a
threshold, that is, when the learner is confident about the part it can
currently see. That is Thrish's "learn that, then relax a little bit more",
expressed as a stopping rule the learner can actually evaluate.

The measurement that matters is no longer trials to criterion. It is the bias
on the dominant time constant, because that is what `bias.py` shows
co-contraction acts on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kalman_learner import TwoTimescaleEKF
from .plant import (
    TAU_TRUE,
    XI_SOFT,
    band_limited_noise,
    cascade_exact,
    experienced_taus,
    reach_reference,
)

DT = 0.01
T_TRIAL = 2.5
NOISE_STD = 0.012
U_EXPLORE = 0.10
S_BASE = 0.15
MIN_USABLE = 20
CRITERION = 0.10
CRITERION_STREAK = 3
MIN_TRIALS_PER_STAGE = 5
CONFIDENCE_SD = 0.04   # posterior sd on log tau_d that licenses relaxing

STAGES = ((3.0, 0.80), (1.3, 0.45), (XI_SOFT, 0.25))
FAST_MOVE = 0.25

CONDITIONS = ("K1", "K2", "K3", "K4", "K5")

CONDITION_META = {
    "K1": {"label": "K1 deep end", "color": "#8C1428"},
    "K2": {"label": "K2 confidence-gated relaxation", "color": "#1B365D"},
    "K3": {"label": "K3 stiff throughout", "color": "#5C5C5C"},
    "K4": {"label": "K4 bandwidth only", "color": "#1E6B3A"},
    "K5": {"label": "K5 random tuning", "color": "#C5761A"},
}


def _impulse(taus, dt: float, n: int) -> np.ndarray:
    u = np.zeros(n)
    u[0] = 1.0 / dt
    return cascade_exact(taus, u, dt) * dt


def model_error(learner, dt: float = DT, n: int = 400) -> float:
    """Relative impulse error of the learner's belief about the *object*."""
    y_true = _impulse(TAU_TRUE, dt, n)
    y_hat = learner.object_impulse(n)
    return float(
        np.sqrt(np.mean((y_true - y_hat) ** 2)) / (np.sqrt(np.mean(y_true**2)) + 1e-12)
    )


@dataclass
class TrialLog:
    err: np.ndarray
    bias_d: np.ndarray       # tau_hat_d - tau_d, signed
    n_eff: np.ndarray        # posterior count of well-determined modes
    sd_log_tau_d: np.ndarray
    tau_hat: np.ndarray      # (n_trials, 3)
    xi: np.ndarray
    t_move: np.ndarray
    failed: np.ndarray
    stage: np.ndarray
    ttc: int
    n_failed: int
    seed: int
    condition: str


def _setting(condition: str, stage: int, rng: np.random.Generator):
    if condition == "K1":
        return XI_SOFT, FAST_MOVE
    if condition == "K2":
        return STAGES[stage]
    if condition == "K3":
        return STAGES[0]
    if condition == "K4":
        return XI_SOFT, STAGES[stage][1]
    if condition == "K5":
        return float(rng.uniform(XI_SOFT, 3.0)), FAST_MOVE
    raise ValueError(condition)


def run_seed(condition: str, seed: int, n_trials: int = 60, dt: float = DT) -> TrialLog:
    rng = np.random.default_rng(seed)
    n_t = int(round(T_TRIAL / dt))
    learner = TwoTimescaleEKF(dt=dt)
    stage = 0

    err = np.zeros(n_trials)
    bias_d = np.zeros(n_trials)
    n_eff = np.zeros(n_trials)
    sd_d = np.full(n_trials, np.nan)
    tau_hat = np.zeros((n_trials, 3))
    xi_log = np.zeros(n_trials)
    tm_log = np.zeros(n_trials)
    failed_log = np.zeros(n_trials, dtype=bool)
    stage_log = np.zeros(n_trials, dtype=int)

    streak, ttc, trials_in_stage = 0, n_trials + 1, 0

    for n in range(n_trials):
        xi, t_move = _setting(condition, stage, rng)

        ref = reach_reference(n_t, dt, t_move)
        u = ref + U_EXPLORE * band_limited_noise(n_t, dt, t_move, rng)
        y = cascade_exact(experienced_taus(TAU_TRUE, xi), u, dt) + NOISE_STD * rng.normal(size=n_t)
        y_hat = learner.predict(u, xi)

        tolerance = S_BASE * xi / XI_SOFT
        over = np.flatnonzero(np.abs(y - y_hat) > tolerance)
        failed = over.size > 0
        stop = int(over[0]) if failed else n_t

        if stop >= MIN_USABLE:
            stats = learner.observe(u[:stop], y[:stop], xi)
            sd_d[n] = stats["sd_log_tau_d"]
        else:
            learner.drift()

        tau_hat[n] = learner.taus
        err[n] = model_error(learner, dt=dt)
        bias_d[n] = float(learner.tau_d - TAU_TRUE[0])
        n_eff[n] = learner.n_eff
        xi_log[n], tm_log[n] = xi, t_move
        failed_log[n], stage_log[n] = failed, stage

        if err[n] < CRITERION:
            streak += 1
            if streak >= CRITERION_STREAK and ttc == n_trials + 1:
                ttc = n + 1
        else:
            streak = 0

        if condition in ("K2", "K4") and stage < len(STAGES) - 1:
            trials_in_stage += 1
            confident = np.isfinite(sd_d[n]) and sd_d[n] < CONFIDENCE_SD
            if trials_in_stage >= MIN_TRIALS_PER_STAGE and confident:
                stage += 1
                trials_in_stage = 0

    return TrialLog(
        err=err,
        bias_d=bias_d,
        n_eff=n_eff,
        sd_log_tau_d=sd_d,
        tau_hat=tau_hat,
        xi=xi_log,
        t_move=tm_log,
        failed=failed_log,
        stage=stage_log,
        ttc=int(ttc),
        n_failed=int(failed_log.sum()),
        seed=seed,
        condition=condition,
    )


def run_experiment(n_trials=60, n_seeds=20, seed0=0, conditions=CONDITIONS):
    return {
        c: [
            run_seed(c, seed=seed0 + 1000 * (CONDITIONS.index(c) + 1) + i, n_trials=n_trials)
            for i in range(n_seeds)
        ]
        for c in conditions
    }
