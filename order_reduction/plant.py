# -*- coding: utf-8 -*-
"""Fixed third-order object, plus the two things the learner controls.

The object never changes:

    G(s) = 1 / ((1 + tau_d s)(1 + tau_1 s)(1 + tau_2 s))

with (tau_d, tau_1, tau_2) = (1.0, 0.1, 0.05) s, i.e. poles at -1, -10, -20.

Co-contraction xi does not change the object. It changes the *coupled* system
the learner experiences. Following the singular-perturbation reduction in the
meeting note, raising the damping ratio past 1 separates the time scales by

    rho(xi) = (xi + sqrt(xi^2 - 1))^2      for xi > 1,   else 1,

so the two fast time constants are divided by rho and settle inside a fraction
of a movement. At xi = 3 (rho ~ 34) the transients are invisible and the
learner effectively faces the dominant first-order lag alone.

This is the whole point of the design: stiffening buys a *simpler system to
identify*, and it simultaneously *hides* the very transients that must
eventually be learned. Those two consequences are what force a sequence.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

TAU_TRUE = (1.0, 0.1, 0.05)
TRUE_POLES = np.array([-1.0 / t for t in TAU_TRUE])
XI_SOFT = 0.7


def rho(xi: float) -> float:
    """Overdamped separation ratio. 1 below the critical damping ratio."""
    if xi <= 1.0:
        return 1.0
    return float((xi + np.sqrt(xi * xi - 1.0)) ** 2)


def experienced_taus(taus, xi: float):
    """Dominant lag is untouched; the fast lags are compressed by rho(xi)."""
    r = rho(xi)
    td, t1, t2 = taus
    return (td, t1 / r, t2 / r)


def cascade(taus, u: np.ndarray, dt: float) -> np.ndarray:
    """Zero-order-hold cascade of first-order lags. A tau of 0 is a pass-through."""
    y = np.asarray(u, dtype=float)
    for tau in taus:
        if tau <= 1e-6:
            continue
        a = float(np.exp(-dt / tau))
        y = lfilter([0.0, 1.0 - a], [1.0, -a], y)
    return y


def impulse(taus, dt: float, n: int) -> np.ndarray:
    u = np.zeros(n)
    u[0] = 1.0 / dt
    return cascade(taus, u, dt) * dt


def model_error(taus_hat, dt: float = 0.01, n: int = 400) -> float:
    """Relative impulse-response error of a model against the true object."""
    y_true = impulse(TAU_TRUE, dt, n)
    y_hat = impulse(taus_hat, dt, n)
    denom = float(np.sqrt(np.mean(y_true**2)) + 1e-12)
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)) / denom)


def min_jerk(t: np.ndarray, t_move: float, amp: float = 1.0) -> np.ndarray:
    s = np.clip(t / t_move, 0.0, 1.0)
    return amp * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)


def band_limited_noise(n: int, dt: float, t_move: float, rng) -> np.ndarray:
    """Exploration a limb can actually produce: no energy above the movement band.

    This is the single most important modelling choice in the study. Voluntary
    movement is smooth, so the only way to put energy near the 10 and 20 rad/s
    poles is to move fast. An impulsive or white probe would excite every mode
    at once and the sample-complexity argument behind the hypothesis would be
    void -- there would be no penalty for trying to identify all three modes
    from trial one.
    """
    a = float(np.exp(-dt / (t_move / 3.0)))
    w = rng.normal(size=n)
    y = lfilter([1.0 - a], [1.0, -a], w)
    return y / (float(np.std(y)) + 1e-12)


def reach_reference(n: int, dt: float, t_move: float) -> np.ndarray:
    """One out-and-back reach. Shorter t_move means more high-frequency content."""
    t = np.arange(n) * dt
    out = min_jerk(t, t_move)
    back = min_jerk(np.clip(t - (t_move + 0.35), 0.0, None), t_move, amp=-1.0)
    return out + back
