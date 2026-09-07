# -*- coding: utf-8 -*-
"""Learner 4: a human-like filter that only estimates the slow lag.

Learners 1--3 always scored a model of the *whole object*. That is why a
stiff first-order learner looked like a failure: its impulse error against
the true third-order cup plateaus near 0.4, even when its estimate of the
dominant lag is essentially perfect. This learner takes the other modelling
choice. The claim, from the 2026-07-01 meeting and from what a person
actually keeps, is that the slow dynamics are what get learned, and that
co-contraction's job is to *filter* the transients so that a first-order
model of a third-order world is no longer misspecified.

The state is a single number, log tau_d. The plant is unchanged: the two
fast lags are still there, and they are still compressed by rho(xi). When
the limb is soft those transients leak into the residual and inflate the
filter's measurement noise, so the update on tau_d is cautious and biased.
When the limb is stiff the transients have already settled, the residual
collapses onto sensor noise, and the same one-parameter filter is both
unbiased and fast. Nothing here tells the learner to go faster under
co-contraction; the speed-up, if it appears, has to come from the residual.

The update rule is the iterated-EKF step of `kalman_learner.TwoTimescaleEKF`,
restricted to the dominant-mode channel and with the same hyperparameters
on that channel, so a comparison against the full third-order EKF is a
comparison of *what is being estimated*, not of how the estimator works.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .kalman_learner import (
    FD_EPS,
    IEKF_ITERS,
    LOG_TAU_BOUNDS,
    MAX_STEP,
    P_FLOOR,
    P_INIT,
    PROCESS_NOISE,
    R_STD,
    TAU_INIT,
    THIN,
)


def _lag(tau: float, x: np.ndarray, dt: float) -> np.ndarray:
    if tau <= 1e-6:
        return np.asarray(x, dtype=float)
    a = float(np.exp(-dt / tau))
    return lfilter([1.0 - a], [1.0, -a], np.asarray(x, dtype=float))


class FirstOrderEKF:
    """EKF on log tau_d only. The transients are the world's problem, not the model's."""

    def __init__(self, dt: float = 0.01):
        self.dt = float(dt)
        self.q = np.array([np.log(TAU_INIT[0])], dtype=float)
        self.P = np.array([[float(P_INIT[0])]])
        self.Q = np.array([[float(PROCESS_NOISE[0])]])
        self.lo = np.array([np.log(LOG_TAU_BOUNDS[0][0])])
        self.hi = np.array([np.log(LOG_TAU_BOUNDS[0][1])])

    @property
    def tau_d(self) -> float:
        return float(np.exp(self.q[0]))

    @property
    def taus(self) -> np.ndarray:
        return np.array([self.tau_d, 0.0, 0.0])

    @property
    def n_eff(self) -> float:
        return 1.0

    def _h(self, q: np.ndarray, u: np.ndarray, xi: float) -> np.ndarray:
        # xi is unused: the dominant lag is not compressed. Kept so the
        # call signature matches TwoTimescaleEKF.predict / observe.
        del xi
        return _lag(float(np.exp(q[0])), u, self.dt)

    def predict(self, u: np.ndarray, xi: float) -> np.ndarray:
        return self._h(self.q, u, xi)

    def object_impulse(self, n: int) -> np.ndarray:
        u = np.zeros(n)
        u[0] = 1.0 / self.dt
        return self._h(self.q, u, 0.7) * self.dt

    def _jacobian(self, q, u, xi, h0=None):
        if h0 is None:
            h0 = self._h(q, u, xi)
        qp = q.copy()
        qp[0] += FD_EPS
        return ((self._h(qp, u, xi) - h0) / FD_EPS).reshape(-1, 1)

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
            A = P_inv + (H.T @ H) / R
            P_post = np.linalg.inv(A)
            rhs = (H.T @ e) / R - P_inv @ (q - q_pred)
            q = np.clip(q + np.clip(P_post @ rhs, -MAX_STEP, MAX_STEP), self.lo, self.hi)

        self.q = q
        self.P = P_post + np.eye(1) * P_FLOOR

        resid = (y - self._h(self.q, u, xi))[::THIN]
        return {
            "pred_rmse": float(np.sqrt(np.mean(resid**2))),
            "n_eff": self.n_eff,
            "sd_log_tau_d": float(np.sqrt(max(self.P[0, 0], 0.0))),
        }

    def drift(self) -> None:
        self.P = self.P + self.Q
