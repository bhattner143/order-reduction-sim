# -*- coding: utf-8 -*-
"""C1-C5: slosh object, open-loop vs plant-inverse commands.

The cup is now a lag plus a lightly damped resonance (see slosh.py). Scoring
is the same identification metrics as Learner 2: object impulse error, bias
on tau_d, n_eff, failed trials. Tracking error is logged and is never the
criterion. There is no task model.

C1  deep end, open loop     command = reach + explore
C2  curriculum, open loop   confidence-gated xi and speed
C3  stiff throughout        open loop
C4  deep end, closed loop   command = invert(belief, reach) + explore
C5  curriculum, closed loop invert + confidence-gated relaxation

Closed loop uses only the learner's current plant belief to pre-emphasise
the same min-jerk reach. A wrong wn puts the lead at the wrong frequency
and corrupts the next trial's data. That is the information-structure
change. The reach itself is the goal already used in experiment2; it is
not a model of the task dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plant import XI_SOFT, band_limited_noise, reach_reference
from .slosh import TAU_D, WN, ZETA, impulse as slosh_impulse, simulate
from .slosh_learner import SloshEKF

DT = 0.01
T_TRIAL = 2.5
NOISE_STD = 0.012
U_EXPLORE = 0.10
S_BASE = 0.15
MIN_USABLE = 20
CRITERION = 0.10
CRITERION_STREAK = 3
MIN_TRIALS_PER_STAGE = 5
CONFIDENCE_SD = 0.04

STAGES = ((3.0, 0.80), (1.3, 0.45), (XI_SOFT, 0.25))
FAST_MOVE = 0.25

CONDITIONS = ("C1", "C2", "C3", "C4", "C5")

CONDITION_META = {
    "C1": {"label": "C1 deep end, open", "color": "#8C1428", "closed": False},
    "C2": {"label": "C2 curriculum, open", "color": "#1B365D", "closed": False},
    "C3": {"label": "C3 stiff throughout", "color": "#5C5C5C", "closed": False},
    "C4": {"label": "C4 deep end, closed", "color": "#C5761A", "closed": True},
    "C5": {"label": "C5 curriculum, closed", "color": "#1E6B3A", "closed": True},
}


def model_error(learner, dt: float = DT, n: int = 400) -> float:
    """Relative impulse error of the belief about the object. Not tracking."""
    y_true = slosh_impulse(TAU_D, WN, ZETA, dt, n, xi=XI_SOFT, w=1.0)
    y_hat = learner.object_impulse(n)
    return float(
        np.sqrt(np.mean((y_true - y_hat) ** 2)) / (np.sqrt(np.mean(y_true**2)) + 1e-12)
    )


@dataclass
class TrialLog:
    err: np.ndarray
    bias_d: np.ndarray
    wn_hat: np.ndarray
    n_eff: np.ndarray
    sd_log_tau_d: np.ndarray
    xi: np.ndarray
    t_move: np.ndarray
    failed: np.ndarray
    stage: np.ndarray
    track_err: np.ndarray   # logged only; never the criterion
    ttc: int
    n_failed: int
    seed: int
    condition: str


def _setting(condition: str, stage: int, rng: np.random.Generator):
    if condition in ("C1", "C4"):
        return XI_SOFT, FAST_MOVE
    if condition in ("C2", "C5"):
        return STAGES[stage]
    if condition == "C3":
        return STAGES[0]
    raise ValueError(condition)


# Max fraction of (invert - reach) mixed into the command. Full inversion of a
# wrong slosh aborts every trial before MIN_USABLE samples; a cautious mix
# corrupts the data instead of deleting it. Authority also tracks the slosh
# weight and the dominant-mode confidence, which is the dual-control part:
# do not invert a mode you do not yet believe.
CLOSED_AUTHORITY = 0.35


def _command(learner, y_des, xi, explore, closed: bool) -> np.ndarray:
    if not closed:
        return y_des + explore
    u_inv = learner.invert(y_des, xi)
    if not np.all(np.isfinite(u_inv)) or float(np.std(u_inv)) > 8.0:
        return y_des + explore
    sd = float(np.sqrt(max(learner.P[0, 0], 1e-8)))
    conf = float(np.clip(1.0 - sd / 0.45, 0.05, 1.0))
    beta = CLOSED_AUTHORITY * float(np.clip(learner.weight, 0.0, 1.0)) * conf
    u = y_des + beta * (u_inv - y_des)
    return np.clip(u, -4.0, 4.0) + explore


def run_seed(condition: str, seed: int, n_trials: int = 60, dt: float = DT) -> TrialLog:
    rng = np.random.default_rng(seed)
    n_t = int(round(T_TRIAL / dt))
    learner = SloshEKF(dt=dt)
    stage = 0
    closed = CONDITION_META[condition]["closed"]

    err = np.zeros(n_trials)
    bias_d = np.zeros(n_trials)
    wn_hat = np.zeros(n_trials)
    n_eff = np.zeros(n_trials)
    sd_d = np.full(n_trials, np.nan)
    xi_log = np.zeros(n_trials)
    tm_log = np.zeros(n_trials)
    failed_log = np.zeros(n_trials, dtype=bool)
    stage_log = np.zeros(n_trials, dtype=int)
    track = np.zeros(n_trials)

    streak, ttc, trials_in_stage = 0, n_trials + 1, 0

    for n in range(n_trials):
        xi, t_move = _setting(condition, stage, rng)
        y_des = reach_reference(n_t, dt, t_move)
        explore = U_EXPLORE * band_limited_noise(n_t, dt, t_move, rng)
        u = _command(learner, y_des, xi, explore, closed)
        y = simulate(TAU_D, WN, ZETA, u, dt, xi) + NOISE_STD * rng.normal(size=n_t)
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

        err[n] = model_error(learner, dt=dt)
        bias_d[n] = float(learner.tau_d - TAU_D)
        wn_hat[n] = learner.wn
        n_eff[n] = learner.n_eff
        xi_log[n], tm_log[n] = xi, t_move
        failed_log[n], stage_log[n] = failed, stage
        track[n] = float(np.sqrt(np.mean((y - y_des) ** 2)))

        if err[n] < CRITERION:
            streak += 1
            if streak >= CRITERION_STREAK and ttc == n_trials + 1:
                ttc = n + 1
        else:
            streak = 0

        if condition in ("C2", "C5") and stage < len(STAGES) - 1:
            trials_in_stage += 1
            confident = np.isfinite(sd_d[n]) and sd_d[n] < CONFIDENCE_SD
            if trials_in_stage >= MIN_TRIALS_PER_STAGE and confident:
                stage += 1
                trials_in_stage = 0

    return TrialLog(
        err=err,
        bias_d=bias_d,
        wn_hat=wn_hat,
        n_eff=n_eff,
        sd_log_tau_d=sd_d,
        xi=xi_log,
        t_move=tm_log,
        failed=failed_log,
        stage=stage_log,
        track_err=track,
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
