# -*- coding: utf-8 -*-
"""Learner 2: two-timescale extended Kalman filter with ARD mode participation.

Learner 1 (`learner.py`) was fixed-step gradient descent over three time
constants, with the model order switched by hand at stage boundaries. It had
two defects, and this learner is built to remove both.

*It was correctly specified.* It always fitted the true three-lag structure, so
its only error was variance. Co-contraction does not buy variance, it buys
freedom from the misspecification bias derived in `bias.py`, so learner 1 could
not see the effect however it was tuned.

*It did not know what it could not know.* With a fixed step, a learner that
co-contracts receives no information about the fast modes and random-walks on
them, destroying what it knew. Learner 1 hid this by hard-locking the fast time
constants to zero during the stiff stage, which hand-codes the very phenomenon
the study is supposed to demonstrate.

Both are fixed by carrying uncertainty. The state is

    q = [ log tau_d,  log tau_1,  log tau_2,  w_1,  w_2 ]

updated by an iterated EKF in information form. The two transients enter
through a *participation weight* rather than through their time constant,

    blend(tau, w) :  x  ->  (1 - w) x  +  w * lag(tau) x,

so w = 1 is the mode fully present and w = 0 is the mode already settled. That
choice matters more than it looks. The obvious encoding of "this mode is off",
driving tau to zero, also drives the mode's sensitivity to zero, so a pruned
mode becomes invisible to the data and pruning is irreversible: a learner that
switches a mode off while stiff can never switch it back on when it relaxes,
however plainly the soft data show it. A participation weight keeps the
derivative alive at w = 0, so pruning stays reversible and recruitment is
something evidence can cause.

Three properties follow rather than being imposed:

* when co-contraction compresses the fast modes the corresponding Jacobian
  columns vanish, those directions receive no information, and the posterior
  does not move. Freezing the unobservable is a consequence, not a rule;

* the weights carry an automatic-relevance-determination prior shrinking them
  to zero, so a mode is retained only when the data pay for it, and

      n_eff = 1 + w_1 + w_2

  is the effective order the meeting note makes its central observable;

* process noise is anisotropic, small on the dominant mode and larger on the
  transients: a slow, well-retained estimate of the dominant lag and a fast,
  forgetful estimate of the transients.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .plant import rho

# state layout
I_TAU_D, I_TAU_1, I_TAU_2, I_W1, I_W2 = range(5)
W_IDX = (I_W1, I_W2)

TAU_INIT = (0.30, 0.05, 0.02)
W_INIT = 0.25
LOG_TAU_BOUNDS = ((0.02, 5.0), (0.005, 0.6), (0.002, 0.3))
W_BOUNDS = (0.0, 1.2)

P_INIT = (0.50, 0.50, 0.50, 0.25, 0.25)
PROCESS_NOISE = (1e-5, 2e-3, 2e-3, 2e-3, 2e-3)
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
        return x
    a = float(np.exp(-dt / tau))
    return lfilter([1.0 - a], [1.0, -a], x)


def _blend(tau: float, w: float, x: np.ndarray, dt: float) -> np.ndarray:
    """A mode that is present in proportion w. Unit DC gain for any w."""
    if w <= 1e-9:
        return x
    return (1.0 - w) * x + w * _lag(tau, x, dt)


class TwoTimescaleEKF:
    def __init__(self, dt: float = 0.01):
        self.dt = float(dt)
        self.q = np.array(
            [np.log(TAU_INIT[0]), np.log(TAU_INIT[1]), np.log(TAU_INIT[2]), W_INIT, W_INIT]
        )
        self.P = np.diag(np.array(P_INIT, dtype=float))
        self.Q = np.diag(np.array(PROCESS_NOISE, dtype=float))
        self.alpha = np.zeros(5)
        self.alpha[I_W1] = self.alpha[I_W2] = ALPHA_INIT
        self.q_target = np.zeros(5)          # ARD shrinks the weights to zero
        self.gamma = np.zeros(5)
        self.lo = np.array(
            [np.log(LOG_TAU_BOUNDS[0][0]), np.log(LOG_TAU_BOUNDS[1][0]),
             np.log(LOG_TAU_BOUNDS[2][0]), W_BOUNDS[0], W_BOUNDS[0]]
        )
        self.hi = np.array(
            [np.log(LOG_TAU_BOUNDS[0][1]), np.log(LOG_TAU_BOUNDS[1][1]),
             np.log(LOG_TAU_BOUNDS[2][1]), W_BOUNDS[1], W_BOUNDS[1]]
        )

    # -- readouts ----------------------------------------------------------
    @property
    def taus(self) -> np.ndarray:
        return np.exp(self.q[:3])

    @property
    def weights(self) -> np.ndarray:
        return self.q[3:].copy()

    @property
    def n_eff(self) -> float:
        return float(1.0 + self.q[I_W1] + self.q[I_W2])

    @property
    def tau_d(self) -> float:
        return float(np.exp(self.q[I_TAU_D]))

    # -- model -------------------------------------------------------------
    def _h(self, q: np.ndarray, u: np.ndarray, xi: float) -> np.ndarray:
        r = rho(xi)
        x = _lag(float(np.exp(q[I_TAU_D])), np.asarray(u, float), self.dt)
        x = _blend(float(np.exp(q[I_TAU_1])) / r, float(q[I_W1]), x, self.dt)
        x = _blend(float(np.exp(q[I_TAU_2])) / r, float(q[I_W2]), x, self.dt)
        return x

    def predict(self, u: np.ndarray, xi: float) -> np.ndarray:
        """What the learner expects to feel, knowing its own co-contraction."""
        return self._h(self.q, u, xi)

    def object_impulse(self, n: int) -> np.ndarray:
        """The learner's belief about the object itself, at zero co-contraction."""
        u = np.zeros(n)
        u[0] = 1.0 / self.dt
        return self._h(self.q, u, 0.7) * self.dt

    def _jacobian(self, q, u, xi, h0=None):
        if h0 is None:
            h0 = self._h(q, u, xi)
        J = np.empty((h0.size, q.size))
        for i in range(q.size):
            qp = q.copy()
            qp[i] += FD_EPS
            J[:, i] = (self._h(qp, u, xi) - h0) / FD_EPS
        return J

    # -- update ------------------------------------------------------------
    def observe(self, u: np.ndarray, y: np.ndarray, xi: float) -> dict:
        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)
        P_inv = np.linalg.inv(self.P + self.Q)
        q_pred = self.q.copy()
        q = q_pred.copy()
        P_post = self.P

        for _ in range(IEKF_ITERS):
            # Simulate at the true sample rate and thin afterwards. Thinning
            # the input first would run the model at the wrong sample interval
            # and make both residual and Jacobian meaningless.
            h0 = self._h(q, u, xi)
            H = self._jacobian(q, u, xi, h0)[::THIN]
            e = (y - h0)[::THIN]
            # Early residuals are model error, not sensor noise, so a fixed R
            # would make the filter certain long before it is right.
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
        self.P = P_post + np.eye(5) * P_FLOOR

        for i in W_IDX:
            g = 1.0 - self.alpha[i] * self.P[i, i]
            self.gamma[i] = float(np.clip(g, 0.0, 1.0))
            self.alpha[i] = float(
                np.clip(self.gamma[i] / max(self.q[i] ** 2, 1e-4), *ALPHA_BOUNDS)
            )

        resid = (y - self._h(self.q, u, xi))[::THIN]
        return {
            "pred_rmse": float(np.sqrt(np.mean(resid**2))),
            "n_eff": self.n_eff,
            "sd_log_tau_d": float(np.sqrt(max(self.P[I_TAU_D, I_TAU_D], 0.0))),
        }

    def drift(self) -> None:
        """A failed trial teaches nothing but still costs retention."""
        self.P = self.P + self.Q
