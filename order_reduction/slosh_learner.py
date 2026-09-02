# -*- coding: utf-8 -*-
"""Learner 3: two-timescale EKF for a lag-plus-slosh object.

Same information-form iterated EKF as Learner 2, different plant. State is

    q = [ log tau_d,  log wn,  log zeta,  w ]

The slosh enters through a participation weight, not by driving wn to
infinity. w = 0 is pass-through after the dominant lag; the derivative
with respect to w stays alive, so a mode pruned while stiff can come back
when the arm relaxes.

Nothing here is a model of the task. invert() turns a desired output into
a command using the current plant belief. That is plant inversion, not
task specification. Scoring stays on the object impulse error.
"""
from __future__ import annotations

import numpy as np

from .slosh import blend_slosh, experienced_slosh, invert_felt, impulse as slosh_impulse
from scipy.signal import lfilter

I_TAU_D, I_WN, I_ZETA, I_W = range(4)

TAU_INIT = 0.30
# Above the spurious basin, not below it. Scanning the loss over log wn on a
# soft fast trial shows two minima, the true one near 10 rad/s and a side
# lobe near 22 rad/s. Starting at 4 rad/s the learner is already downhill of
# the true minimum and walks straight to it, so the multi-basin structure
# that motivated a complex pole in the first place is never encountered.
# Starting at 24 puts the deep end inside the wrong basin and makes the
# curriculum's claim -- arrive at the frequency search with tau_d already
# correct, so the residual is cleanly the resonance -- actually testable.
WN_INIT = 24.0
ZETA_INIT = 0.40
W_INIT = 0.30

LOG_TAU_BOUNDS = (0.05, 5.0)
LOG_WN_BOUNDS = (1.5, 40.0)
LOG_ZETA_BOUNDS = (0.03, 1.5)
W_BOUNDS = (0.0, 1.2)

P_INIT = (0.50, 0.50, 0.40, 0.25)
PROCESS_NOISE = (1e-5, 3e-3, 3e-3, 2e-3)
R_STD = 0.02
THIN = 5
ALPHA_INIT = 1.0
ALPHA_BOUNDS = (1e-3, 1e3)
P_FLOOR = 1e-4
MAX_STEP = 0.4
IEKF_ITERS = 3
FD_EPS = 1e-3


def _lag(tau: float, x: np.ndarray, dt: float) -> np.ndarray:
    if tau <= 1e-6:
        return np.asarray(x, dtype=float)
    a = float(np.exp(-dt / tau))
    return lfilter([1.0 - a], [1.0, -a], np.asarray(x, dtype=float))


class SloshEKF:
    def __init__(self, dt: float = 0.01):
        self.dt = float(dt)
        self.q = np.array(
            [np.log(TAU_INIT), np.log(WN_INIT), np.log(ZETA_INIT), W_INIT]
        )
        self.P = np.diag(np.array(P_INIT, dtype=float))
        self.Q = np.diag(np.array(PROCESS_NOISE, dtype=float))
        self.alpha = np.zeros(4)
        self.alpha[I_W] = ALPHA_INIT
        self.q_target = np.zeros(4)
        self.gamma = np.zeros(4)
        self.lo = np.array(
            [np.log(LOG_TAU_BOUNDS[0]), np.log(LOG_WN_BOUNDS[0]),
             np.log(LOG_ZETA_BOUNDS[0]), W_BOUNDS[0]]
        )
        self.hi = np.array(
            [np.log(LOG_TAU_BOUNDS[1]), np.log(LOG_WN_BOUNDS[1]),
             np.log(LOG_ZETA_BOUNDS[1]), W_BOUNDS[1]]
        )

    @property
    def tau_d(self) -> float:
        return float(np.exp(self.q[I_TAU_D]))

    @property
    def wn(self) -> float:
        return float(np.exp(self.q[I_WN]))

    @property
    def zeta(self) -> float:
        return float(np.exp(self.q[I_ZETA]))

    @property
    def weight(self) -> float:
        return float(self.q[I_W])

    @property
    def n_eff(self) -> float:
        return float(1.0 + self.q[I_W])

    def _h(self, q: np.ndarray, u: np.ndarray, xi: float) -> np.ndarray:
        tau_d = float(np.exp(q[I_TAU_D]))
        wn, zeta = experienced_slosh(float(np.exp(q[I_WN])), float(np.exp(q[I_ZETA])), xi)
        x = _lag(tau_d, u, self.dt)
        return blend_slosh(wn, zeta, float(q[I_W]), x, self.dt)

    def predict(self, u: np.ndarray, xi: float) -> np.ndarray:
        return self._h(self.q, u, xi)

    def invert(self, y_des: np.ndarray, xi: float) -> np.ndarray:
        """Plant inverse of the current belief. Not a task model."""
        return invert_felt(
            self.tau_d, self.wn, self.zeta, self.weight, y_des, self.dt, xi
        )

    def object_impulse(self, n: int) -> np.ndarray:
        """Belief about the object itself, at soft (uncompressed) slosh."""
        return slosh_impulse(
            self.tau_d, self.wn, self.zeta, self.dt, n, xi=0.7, w=self.weight
        )

    def _jacobian(self, q, u, xi, h0=None):
        if h0 is None:
            h0 = self._h(q, u, xi)
        J = np.empty((h0.size, q.size))
        for i in range(q.size):
            qp = q.copy()
            qp[i] += FD_EPS
            J[:, i] = (self._h(qp, u, xi) - h0) / FD_EPS
        return J

    def observe(self, u: np.ndarray, y: np.ndarray, xi: float) -> dict:
        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)
        P_inv = np.linalg.inv(self.P + self.Q)
        q_pred = self.q.copy()
        q = q_pred.copy()
        P_post = self.P

        for _ in range(IEKF_ITERS):
            h0 = self._h(q, u, xi)
            H = self._jacobian(q, u, xi, h0)[::THIN]
            e = (y - h0)[::THIN]
            R = max(R_STD**2, float(np.mean(e**2)))
            A = P_inv + (H.T @ H) / R + np.diag(self.alpha)
            P_post = np.linalg.inv(A)
            rhs = (
                (H.T @ e) / R
                - self.alpha * (q - self.q_target)
                - P_inv @ (q - q_pred)
            )
            q = np.clip(q + np.clip(P_post @ rhs, -MAX_STEP, MAX_STEP), self.lo, self.hi)

        self.q = q
        self.P = P_post + np.eye(4) * P_FLOOR

        g = 1.0 - self.alpha[I_W] * self.P[I_W, I_W]
        self.gamma[I_W] = float(np.clip(g, 0.0, 1.0))
        self.alpha[I_W] = float(
            np.clip(self.gamma[I_W] / max(self.q[I_W] ** 2, 1e-4), *ALPHA_BOUNDS)
        )

        resid = (y - self._h(self.q, u, xi))[::THIN]
        return {
            "pred_rmse": float(np.sqrt(np.mean(resid**2))),
            "n_eff": self.n_eff,
            "sd_log_tau_d": float(np.sqrt(max(self.P[I_TAU_D, I_TAU_D], 0.0))),
        }

    def drift(self) -> None:
        self.P = self.P + self.Q
