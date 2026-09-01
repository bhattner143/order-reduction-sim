# -*- coding: utf-8 -*-
"""S1-S5 trial loop: curriculum vs deep-end vs stiff vs bandwidth vs random."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete

from .learner import ERAIdentifier
from .plant import CoupledPlant, true_continuous_ss

DT = 0.01
T_TRIAL = 2.5
T_MOVE = 0.35
NOISE_STD = 0.001
CRITERION = 0.12
CRITERION_STREAK = 3
WARMUP = 4
U_EXPLORE = 0.4

STAGE_XI = (3.0, 2.0, 0.7)
STAGE_WC = (3.0, 12.0, 40.0)
STAGE_ORDER = (1, 2, 3)
MIN_TRIALS_PER_STAGE = 10
PROGRESS_EPS = 1e9
PLATEAU_STREAK = 1

CONDITIONS = ("S1", "S2", "S3", "S4", "S5")

CONDITION_META = {
    "S1": {"label": "S1 deep end", "color": "#8C1428"},
    "S2": {"label": "S2 order recruitment", "color": "#1B365D"},
    "S3": {"label": "S3 stiff throughout", "color": "#5C5C5C"},
    "S4": {"label": "S4 bandwidth only", "color": "#1E6B3A"},
    "S5": {"label": "S5 random tuning", "color": "#C5761A"},
}


def _true_discrete(dt: float):
    A, B, C, D = true_continuous_ss()
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="zoh")
    return (
        np.asarray(Ad, dtype=float),
        np.asarray(Bd, dtype=float).reshape(3, 1),
        np.asarray(Cd, dtype=float).reshape(1, 3),
    )


def _true_impulse(dt: float, n: int) -> np.ndarray:
    Ad, Bd, Cd = _true_discrete(dt)
    y_true = np.zeros(n)
    x = np.zeros(3)
    for k in range(n):
        u = 1.0 if k == 0 else 0.0
        x = (Ad @ x.reshape(3, 1) + Bd * u).ravel()
        y_true[k] = float((Cd @ x).ravel()[0])
    return y_true


def _impulse_mismatch(ident, dt: float, n: int = 250) -> float:
    from .learner import simulate_ss_impulse
    y_true = _true_impulse(dt, n)
    y_id = simulate_ss_impulse(ident.A, ident.B, ident.C, ident.D, n)
    denom = float(np.sqrt(np.mean(y_true ** 2)) + 1e-8)
    return float(np.sqrt(np.mean((y_true - y_id) ** 2)) / denom)


@dataclass
class TrialLog:
    rmse_train: np.ndarray
    rmse_true: np.ndarray
    pred_rmse: np.ndarray
    cond_phi: np.ndarray
    xi: np.ndarray
    omega_c: np.ndarray
    order: np.ndarray
    param_err: np.ndarray
    stage: np.ndarray
    ttc: int
    seed: int
    condition: str


def _initial_tuning(condition: str, rng: np.random.Generator):
    if condition == "S1":
        return 0.7, 200.0, 3, 2
    if condition == "S2":
        return STAGE_XI[0], 200.0, STAGE_ORDER[0], 0
    if condition == "S3":
        return 3.0, 200.0, 1, 0
    if condition == "S4":
        return 0.7, STAGE_WC[0], STAGE_ORDER[0], 0
    if condition == "S5":
        xi = float(rng.uniform(0.7, 3.0))
        return xi, 200.0, 3, 2
    raise ValueError(condition)


def _maybe_advance(condition, stage, trial_in_stage, plateau, pred_rmse, prev_pred):
    if condition not in ("S2", "S4"):
        return stage, trial_in_stage + 1, plateau
    if stage >= 2:
        return stage, trial_in_stage + 1, plateau
    if prev_pred is None:
        return stage, trial_in_stage + 1, 0
    delta = prev_pred - pred_rmse
    plateau_next = plateau + 1 if delta < PROGRESS_EPS else 0
    if trial_in_stage + 1 >= MIN_TRIALS_PER_STAGE and plateau_next >= PLATEAU_STREAK:
        return stage + 1, 0, 0
    return stage, trial_in_stage + 1, plateau_next


def run_seed(condition, seed, n_trials=80, dt=DT, noise_std=NOISE_STD):
    rng = np.random.default_rng(seed)
    xi, wc, order, stage = _initial_tuning(condition, rng)
    ident = ERAIdentifier(order=order)
    plant = CoupledPlant(dt=dt, xi=xi, omega_c=wc)

    n_t = int(round(T_TRIAL / dt))
    ref = np.zeros(n_t)

    rmse_train = np.zeros(n_trials)
    rmse_true = np.zeros(n_trials)
    pred_rmse = np.zeros(n_trials)
    cond_phi = np.zeros(n_trials)
    xi_log = np.zeros(n_trials)
    wc_log = np.zeros(n_trials)
    order_log = np.zeros(n_trials, dtype=int)
    param_err = np.zeros(n_trials)
    stage_log = np.zeros(n_trials, dtype=int)

    prev_pred = None
    trial_in_stage = 0
    plateau = 0
    streak = 0
    ttc = n_trials + 1

    for n in range(n_trials):
        if condition == "S5":
            xi = float(rng.uniform(0.7, 3.0))
            wc = 40.0
            order = 3
        plant.set_tuning(xi, wc)

        y = np.zeros(n_t)
        u = np.zeros(n_t)
        x, u_filt = plant.reset()

        for k in range(n_t):
            uk = 1.0 if k == 0 else 0.0
            x, u_filt, yk = plant.step(x, u_filt, uk)
            yk = yk + noise_std * rng.normal()
            y[k] = yk
            u[k] = uk

        stats = ident.update_trial(y, u)
        rmse_train[n] = float(np.sqrt(np.mean((y - ref) ** 2)))
        pred_rmse[n] = stats["pred_rmse"]
        cond_phi[n] = stats["cond_phi"]
        xi_log[n] = xi
        wc_log[n] = wc
        order_log[n] = ident.order
        stage_log[n] = stage
        param_err[n] = _impulse_mismatch(ident, dt, n=n_t)
        rmse_true[n] = param_err[n]

        if n + 1 > WARMUP and rmse_true[n] < CRITERION:
            streak += 1
            if streak >= CRITERION_STREAK and ttc == n_trials + 1:
                ttc = n + 1
        else:
            streak = 0

        stage, trial_in_stage, plateau = _maybe_advance(
            condition, stage, trial_in_stage, plateau, pred_rmse[n], prev_pred
        )
        prev_pred = pred_rmse[n]
        if condition == "S2" and stage <= 2:
            xi, wc, new_order = STAGE_XI[stage], 200.0, STAGE_ORDER[stage]
            if new_order > ident.order:
                ident.expand_order(new_order)
                order = new_order
        elif condition == "S4" and stage <= 2:
            xi, wc, new_order = 0.7, STAGE_WC[stage], STAGE_ORDER[stage]
            if new_order > ident.order:
                ident.expand_order(new_order)
                order = new_order

    return TrialLog(
        rmse_train=rmse_train,
        rmse_true=rmse_true,
        pred_rmse=pred_rmse,
        cond_phi=cond_phi,
        xi=xi_log,
        omega_c=wc_log,
        order=order_log,
        param_err=param_err,
        stage=stage_log,
        ttc=int(ttc),
        seed=seed,
        condition=condition,
    )


def run_experiment(n_trials=80, n_seeds=20, seed0=0, conditions=CONDITIONS):
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
