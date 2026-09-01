# -*- coding: utf-8 -*-
"""Gray-box learner: estimates the object's time constants, one order at a time.

The learner holds an estimate of (tau_d, tau_1, tau_2) for the *object*. A time
constant that has not been unlocked yet is held at zero, which is exactly what
"the learner currently believes this is a first-order system" means.

Because the learner commands its own co-contraction it knows rho(xi) on every
trial, so a trial recorded while stiff is fitted through the compressed
dynamics it actually produced. Data taken at different stiffness levels
therefore combine correctly into one estimate of the object.

Learning is *incremental gradient descent* with a fixed step, not a batch
solve. That choice is not cosmetic and it is the reason the study has any
content. A batch least-squares solve with the correct model structure is
consistent, so it recovers all three time constants as soon as the data
support them and a curriculum could only ever cost trials. The sample- and
iteration-complexity argument in the meeting note is a statement about
convergence *rate*, N proportional to 1/lambda_min(Phi), and rate is what a
fixed-step gradient learner is limited by. Descent along an ill-conditioned
direction crawls; removing that direction from the problem is precisely what
reducing the effective order does.

Parameters are held in log time constants so that the conditioning measured
here reflects excitation of the modes rather than the arbitrary fact that
tau_d is twenty times larger than tau_2.
"""
from __future__ import annotations

import numpy as np

from .plant import cascade, experienced_taus

TAU_INIT = (0.30, 0.030, 0.015)
TAU_BOUNDS = ((0.02, 5.0), (0.002, 0.6), (0.001, 0.3))
BUFFER = 3
STEPS_PER_TRIAL = 12
LEARNING_RATE = 3.0
FD_EPS = 1e-3


class GrayBoxLearner:
    """Estimates the unlocked subset of (tau_d, tau_1, tau_2) by least squares."""

    def __init__(self, n_unlocked: int = 1, dt: float = 0.01):
        self.dt = float(dt)
        self.n_unlocked = int(n_unlocked)
        self.taus = np.zeros(3)
        for i in range(self.n_unlocked):
            self.taus[i] = TAU_INIT[i]
        self.trials: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.jac_cond = float("inf")

    def unlock(self, n_unlocked: int) -> None:
        """Expose one more mode. Already-learned time constants are kept."""
        if n_unlocked <= self.n_unlocked:
            return
        for i in range(self.n_unlocked, n_unlocked):
            self.taus[i] = TAU_INIT[i]
        self.n_unlocked = int(n_unlocked)

    def predict(self, u: np.ndarray, xi: float) -> np.ndarray:
        """What the learner expects to feel, given its own current stiffness."""
        return cascade(experienced_taus(tuple(self.taus), xi), u, self.dt)

    def observe(self, u: np.ndarray, y: np.ndarray, xi: float) -> dict:
        self.trials.append((np.asarray(u, float), np.asarray(y, float), float(xi)))
        self.trials = self.trials[-BUFFER:]
        return self._fit()

    def _residual(self, q: np.ndarray) -> np.ndarray:
        """Prediction error over the buffered trials, for log time constants q."""
        taus = np.zeros(3)
        taus[: q.size] = np.exp(q)
        out = [
            y - cascade(experienced_taus(tuple(taus), xi), u, self.dt)
            for u, y, xi in self.trials
        ]
        return np.concatenate(out)

    def _jacobian(self, q: np.ndarray, r0: np.ndarray) -> np.ndarray:
        J = np.empty((r0.size, q.size))
        for i in range(q.size):
            qp = q.copy()
            qp[i] += FD_EPS
            J[:, i] = (self._residual(qp) - r0) / FD_EPS
        return J

    def _fit(self) -> dict:
        k = self.n_unlocked
        lo = np.log([TAU_BOUNDS[i][0] for i in range(k)])
        hi = np.log([TAU_BOUNDS[i][1] for i in range(k)])
        q = np.clip(np.log(np.maximum(self.taus[:k], 1e-4)), lo, hi)

        r = self._residual(q)
        m = r.size
        for _ in range(STEPS_PER_TRIAL):
            # cost = 0.5*sum(r^2) with r = y - f(q), so the descent direction
            # is -J^T r with J = dr/dq.
            grad = (self._jacobian(q, r).T @ r) / m
            q = np.clip(q - LEARNING_RATE * grad, lo, hi)
            r = self._residual(q)

        self.taus = np.zeros(3)
        self.taus[:k] = np.exp(q)
        return {"pred_rmse": float(np.sqrt(np.mean(r**2))), "jac_cond": self.full_kappa()}

    def full_kappa(self) -> float:
        """kappa(Phi) of the meeting note, always on all three modes.

        Measured on the full parameter set whatever is currently unlocked, so
        it answers one question for every condition: how hard would it be to
        descend on the complete third-order problem using the data this trial
        just produced?
        """
        if not self.trials:
            return float("nan")
        q = np.log(np.maximum(self.taus, [t[0] for t in TAU_BOUNDS]))
        k = self.n_unlocked
        self.n_unlocked = 3
        try:
            r = self._residual(q)
            J = self._jacobian(q, r)
        finally:
            self.n_unlocked = k
        ev = np.linalg.eigvalsh((J.T @ J) / r.size)
        self.jac_cond = float(ev[-1] / max(ev[0], 1e-18))
        return self.jac_cond
