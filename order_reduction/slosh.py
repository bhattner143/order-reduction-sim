# -*- coding: utf-8 -*-
"""Complex-pole slosh object, and a plant inverse that is not a task model.

The object is a slow lag cascaded with a lightly damped resonance:

    G(s) = 1/(1 + tau_d s)  *  wn^2 / (s^2 + 2 zeta wn s + wn^2)

Coffee sloshes; two real lags do not. The dominant lag is the cup's heaviness.
The second-order block is the slosh (natural frequency wn, damping zeta).

Co-contraction does not change the cup. It compresses the slosh in time:

    wn_felt = wn * rho(xi),   zeta_felt = zeta,   tau_d unchanged.

That is the same rule as experienced_taus: fast poles move left by rho. At
xi = 3 the slosh finishes inside one sample and is treated as a pass-through,
which is what "already settled" should mean.

A first-order learner matching the low-frequency expansion inherits

    tau_hat(xi) = tau_d + 2 zeta / (wn * rho(xi))

so the neglected slosh still biases the dominant estimate, and stiffening
divides that bias by rho. Same claim as bias.py, different transient.

The inverse below is used only to build a command from the learner's current
plant belief. The desired output is the same min-jerk reach already used as
an open-loop command. Nothing here is a model of the task, and nothing is
scored on tracking.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import cont2discrete, lfilter

from .plant import band_limited_noise, cascade_exact, reach_reference, rho

# cup heaviness + slosh. wn = 10 rad/s is about 1.6 Hz; zeta = 0.12 rings.
TAU_D = 1.0
WN = 10.0
ZETA = 0.12
SLOSH_TRUE = (TAU_D, WN, ZETA)
XI_SOFT = 0.7

# Internal oversampling target. A felt slosh at xi = 3 sits near 340 rad/s,
# which dt = 0.01 s cannot represent, so the block is integrated on a finer
# grid and decimated back. Without this the only option is an if-statement
# that switches the resonance off, and that is not acceptable here: the
# switch lands at a different xi for the plant than for the learner (their
# wn differ), it puts a cliff in the loss surface, and it makes dh/d log wn
# *exactly* zero rather than small. A mode must become invisible because the
# excitation cannot reach it, not because a branch deleted it.
WN_DT_TARGET = 0.15
MAX_OVERSAMPLE = 48

# Only above this is the resonance genuinely beyond the simulation. With
# oversampling this is far outside the xi range used anywhere in the study.
WN_DT_PASS = MAX_OVERSAMPLE * WN_DT_TARGET


def experienced_slosh(wn: float, zeta: float, xi: float):
    """Fast poles move left by rho; damping ratio of the slosh is unchanged."""
    return float(wn) * rho(xi), float(zeta)


def _lag(tau: float, u: np.ndarray, dt: float) -> np.ndarray:
    if tau <= 1e-6:
        return np.asarray(u, dtype=float)
    a = float(np.exp(-dt / tau))
    return lfilter([1.0 - a], [1.0, -a], np.asarray(u, dtype=float))


def _oversample(wn: float, dt: float) -> int:
    """How many sub-steps this resonance needs to be represented at all."""
    if wn <= 1e-6:
        return 1
    return int(min(MAX_OVERSAMPLE, max(1, np.ceil(wn * dt / WN_DT_TARGET))))


def _second_order(wn: float, zeta: float, u: np.ndarray, dt: float) -> np.ndarray:
    """ZOH discretisation of wn^2 / (s^2 + 2 zeta wn s + wn^2).

    Integrated on a sub-grid when wn dt is large, then decimated. A fast
    resonance driven by a smooth command then contributes essentially
    nothing, which is the correct behaviour, and it does so smoothly in wn
    instead of jumping at a threshold.
    """
    u = np.asarray(u, dtype=float)
    if wn <= 1e-6 or wn * dt > WN_DT_PASS:
        return u
    if zeta >= 1.0:
        disc = float(np.sqrt(max(zeta * zeta - 1.0, 0.0)))
        tau_fast = 1.0 / (wn * (zeta + disc + 1e-12))
        tau_slow = 1.0 / (wn * (zeta - disc + 1e-12))
        return cascade_exact((tau_slow, tau_fast), u, dt)

    os = _oversample(wn, dt)
    num = [wn * wn]
    den = [1.0, 2.0 * zeta * wn, wn * wn]
    bd, ad, _dt = cont2discrete((num, den), dt / os, method="zoh")
    if os == 1:
        return lfilter(np.ravel(bd), np.ravel(ad), u)
    # Hold each command sample across its sub-interval, filter on the fine
    # grid, then take every os-th sample. Offset 0 is the one that lands on
    # t = k dt; offset os-1 reads a sub-step early and biases the low
    # frequency gain (checked against the analytic 2 zeta / wn).
    u_fine = np.repeat(u, os)
    y_fine = lfilter(np.ravel(bd), np.ravel(ad), u_fine)
    return y_fine[::os]


def blend_slosh(wn: float, zeta: float, w: float, x: np.ndarray, dt: float) -> np.ndarray:
    """Slosh present in proportion w. Unit DC gain for any w."""
    if w <= 1e-9:
        return np.asarray(x, dtype=float)
    return (1.0 - w) * x + w * _second_order(wn, zeta, x, dt)


def simulate(tau_d: float, wn: float, zeta: float, u: np.ndarray, dt: float, xi: float,
             w: float = 1.0) -> np.ndarray:
    """Felt response: dominant lag, then blended slosh at the compressed wn."""
    wn_f, zeta_f = experienced_slosh(wn, zeta, xi)
    x = _lag(tau_d, u, dt)
    return blend_slosh(wn_f, zeta_f, w, x, dt)


def impulse(tau_d: float, wn: float, zeta: float, dt: float, n: int,
            xi: float = XI_SOFT, w: float = 1.0) -> np.ndarray:
    u = np.zeros(n)
    u[0] = 1.0 / dt
    return simulate(tau_d, wn, zeta, u, dt, xi, w=w) * dt


def invert_lag(tau: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Exact inverse of the discrete first-order lag used by _lag."""
    y = np.asarray(y, dtype=float)
    if tau <= 1e-6:
        return y.copy()
    a = float(np.exp(-dt / tau))
    den = 1.0 - a
    u = np.empty_like(y)
    y_prev = 0.0
    for k in range(y.size):
        u[k] = (y[k] - a * y_prev) / (den + 1e-12)
        y_prev = y[k]
    return u


def invert_slosh(wn: float, zeta: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Proper inverse: slosh inverse rolled off at wc so it stays causal.

    G_inv(s) * 1/(1 + s/wc)^2, with wc a little above the believed wn.
    A wrong wn puts the lead at the wrong frequency. That is the point.
    """
    y = np.asarray(y, dtype=float)
    # The inverse of a resonance the command cannot reach is a pass-through.
    # Cut off well below the forward model's limit so the inverse never adds
    # lead at a frequency the plant will not respond at.
    if wn <= 1e-6 or wn * dt > 1.0:
        return y.copy()
    wc = max(float(wn) * 2.0, 20.0)
    num = [1.0, 2.0 * zeta * wn, wn * wn]
    den = [wn * wn / (wc * wc), 2.0 * wn * wn / wc, wn * wn]
    bd, ad, _dt = cont2discrete((num, den), dt, method="zoh")
    return lfilter(np.ravel(bd), np.ravel(ad), y)


def invert_felt(tau_d: float, wn: float, zeta: float, w: float,
                y_des: np.ndarray, dt: float, xi: float, u_lim: float = 4.0) -> np.ndarray:
    """Command that would produce y_des if the believed plant were true.

    Invert slosh first (it is downstream), then the dominant lag. Blend the
    slosh inverse with a pass-through using the same weight the forward model
    uses, so w = 0 recovers invert_lag(y_des) exactly.
    """
    wn_f, zeta_f = experienced_slosh(wn, zeta, xi)
    z = np.asarray(y_des, dtype=float)
    if w > 1e-3:
        z_inv = invert_slosh(wn_f, zeta_f, z, dt)
        z = (1.0 - w) * z + w * z_inv
    u = invert_lag(tau_d, z, dt)
    if not np.all(np.isfinite(u)):
        return np.asarray(y_des, dtype=float)
    return np.clip(u, -u_lim, u_lim)


# ---------------------------------------------------------------------------
# Bias law for a first-order fit to the slosh object
# ---------------------------------------------------------------------------

def predicted_tau_hat(xi: float, tau_d: float = TAU_D, wn: float = WN,
                      zeta: float = ZETA) -> float:
    """tau_d + 2 zeta / (wn rho(xi)), leading-order reduced-fit limit."""
    return tau_d + 2.0 * zeta / (wn * rho(xi))


def predicted_bias(xi: float, tau_d: float = TAU_D, wn: float = WN,
                   zeta: float = ZETA) -> float:
    return predicted_tau_hat(xi, tau_d, wn, zeta) - tau_d


# The bias sweep runs finer than the trial loop on purpose. Holding the
# command over a sample adds dt/2 of pure delay, which a first-order fit
# absorbs as extra time constant. At dt = 0.01 s that floor is 5 ms, while
# the slosh bias at xi = 3 is 0.7 ms, so the 1/rho law would be invisible
# under the discretisation. Same failure mode that `cascade` vs
# `cascade_exact` documents for the three-lag plant, one order worse here
# because 2 zeta / wn is a much smaller number than tau_1 + tau_2.
BIAS_DT = 5e-4


# The probe must also be slow. tau_hat = tau_d + 2 zeta / wn is the *leading
# order* term of a low-frequency expansion, so it only describes the fit when
# the excitation stays below the resonance. Probe near wn and the least
# squares fit stops being a low-frequency match at all: it chases the ringing
# instead, and the bias goes *negative* (-0.030 s at t_move = 0.25 against a
# predicted +0.024). Slowing the probe recovers the law monotonically:
# +0.0005 at 0.8 s, +0.011 at 1.5 s, +0.017 at 2.5 s, +0.0195 at 4.0 s.
# The three-lag plant in bias.py never showed this because a cascade of real
# lags cannot ring, so its low-frequency match is always the dominant error.
BIAS_T_MOVE = 2.5
BIAS_T_TRIAL = 9.0


def fit_first_order(xi: float, n_trials: int = 12, seed: int = 0,
                    dt: float = BIAS_DT, t_move: float = BIAS_T_MOVE,
                    noise_std: float = 0.012, t_trial: float = BIAS_T_TRIAL) -> float:
    rng = np.random.default_rng(seed)
    n_t = int(round(t_trial / dt))
    data = []
    for _ in range(n_trials):
        ref = reach_reference(n_t, dt, t_move)
        u = ref + 0.10 * band_limited_noise(n_t, dt, t_move, rng)
        y = simulate(TAU_D, WN, ZETA, u, dt, xi) + noise_std * rng.normal(size=n_t)
        data.append((u, y))

    def cost(log_tau: float) -> float:
        tau = float(np.exp(log_tau))
        return sum(float(np.mean((y - _lag(tau, u, dt)) ** 2)) for u, y in data)

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
                "bias_fitted": fitted - TAU_D,
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
