# -*- coding: utf-8 -*-
"""The bias of a reduced-order fit, and what co-contraction does to it.

This is the result the first learner could not show, because that learner was
correctly specified: it always fitted the true three-lag structure, so its only
error was variance, and variance is not what co-contraction buys.

A learner that fits *fewer* modes than the world has is biased, and the bias
has a closed form. Expanding the object at low frequency,

    G(s) = 1 / ((1+tau_d s)(1+tau_1 s)(1+tau_2 s))
         = 1 - (tau_d + tau_1 + tau_2) s + O(s^2),

and a first-order model 1/(1 + tau_hat s) = 1 - tau_hat s + O(s^2), so matching
the leading term gives

    tau_hat = tau_d + tau_1 + tau_2.

Three lags in a row feel like one lag equal to their sum. The reduced fit is
therefore biased by exactly the neglected time constants. Under co-contraction
the fast lags are compressed by rho(xi), so

    tau_hat(xi) = tau_d + (tau_1 + tau_2) / rho(xi),
    bias(xi)    = (tau_1 + tau_2) / rho(xi).

Co-contraction divides the misspecification bias by rho. That is the precise
sense in which stiffening lets a low-order learner estimate the dominant mode
correctly, and it is a statement about bias, which no amount of extra data
removes.

The functions below state the prediction and check it against an actual least
squares fit on simulated reaches.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .plant import (
    TAU_TRUE,
    band_limited_noise,
    cascade_exact,
    experienced_taus,
    reach_reference,
    rho,
)

DT = 0.01
T_TRIAL = 3.0
T_MOVE = 0.8
U_EXPLORE = 0.10
NOISE_STD = 0.012


def predicted_tau_hat(xi: float, taus=TAU_TRUE) -> float:
    """tau_d + (tau_1 + tau_2)/rho(xi), the leading-order reduced-fit limit."""
    td, t1, t2 = taus
    return td + (t1 + t2) / rho(xi)


def predicted_bias(xi: float, taus=TAU_TRUE) -> float:
    return predicted_tau_hat(xi, taus) - taus[0]


def fit_first_order(xi: float, n_trials: int = 12, seed: int = 0,
                    dt: float = DT, t_move: float = T_MOVE,
                    noise_std: float = NOISE_STD) -> float:
    """Least squares fit of a single lag to reaches recorded at this xi."""
    rng = np.random.default_rng(seed)
    n_t = int(round(T_TRIAL / dt))
    data = []
    for _ in range(n_trials):
        ref = reach_reference(n_t, dt, t_move)
        u = ref + U_EXPLORE * band_limited_noise(n_t, dt, t_move, rng)
        y = cascade_exact(experienced_taus(TAU_TRUE, xi), u, dt) + noise_std * rng.normal(size=n_t)
        data.append((u, y))

    def cost(log_tau: float) -> float:
        tau = float(np.exp(log_tau))
        return sum(float(np.mean((y - cascade_exact((tau, 0.0, 0.0), u, dt)) ** 2)) for u, y in data)

    res = minimize_scalar(cost, bounds=(np.log(0.05), np.log(5.0)), method="bounded")
    return float(np.exp(res.x))


def sweep(xis=(0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0), n_trials: int = 12, seed: int = 0):
    rows = []
    for xi in xis:
        fitted = fit_first_order(xi, n_trials=n_trials, seed=seed)
        rows.append(
            {
                "xi": float(xi),
                "rho": rho(xi),
                "tau_hat_fitted": fitted,
                "tau_hat_predicted": predicted_tau_hat(xi),
                "bias_fitted": fitted - TAU_TRUE[0],
                "bias_predicted": predicted_bias(xi),
            }
        )
    return rows


def report(rows) -> str:
    head = (
        f"{'xi':>5} {'rho':>7} {'tau_hat fit':>12} {'tau_hat pred':>13} "
        f"{'bias fit':>10} {'bias pred':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['xi']:5.1f} {r['rho']:7.1f} {r['tau_hat_fitted']:12.4f} "
            f"{r['tau_hat_predicted']:13.4f} {r['bias_fitted']:10.4f} {r['bias_predicted']:10.4f}"
        )
    return "\n".join(lines)
