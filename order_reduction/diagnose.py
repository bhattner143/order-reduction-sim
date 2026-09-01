# -*- coding: utf-8 -*-
"""The identifiability test that decides whether the order-reduction claim holds.

The meeting note argues that co-contraction helps learning because reducing the
effective order reduces sample complexity, via

    N(r, delta)  proportional to  sigma^2 r / (delta lambda_min(Phi_r)).

That inequality is about the error in the *parameters*. A proposal needs a
claim about *behaviour*. This module measures both on the same data, so the
step from one to the other can be checked rather than assumed.

For a given tuning (xi, movement time) it forms the Gauss-Newton curvature
matrix in log time constants at the true parameters, takes its eigenvectors,
and asks two questions about each of them:

  * curvature -- how fast does a gradient learner move along this direction?
  * behavioural cost -- how much does the input-output behaviour of the object
    change if you take a fixed-size step along it and get it wrong?

If badly identified directions are also behaviourally cheap, then poor
conditioning is not a performance problem and the sample-complexity route to
the hypothesis is closed.
"""
from __future__ import annotations

import numpy as np

from .learner import GrayBoxLearner
from .plant import (
    TAU_TRUE,
    band_limited_noise,
    cascade,
    experienced_taus,
    model_error,
    reach_reference,
    rho,
)

STEP = 0.2  # size of the probe step, in log time constants


def curvature_spectrum(xi, t_move, dt=0.01, t_trial=2.0, noise_std=0.012,
                       u_explore=0.10, n_trials=3, seed=0):
    """Eigen-decomposition of the curvature, with the behavioural cost of each mode."""
    rng = np.random.default_rng(seed)
    n_t = int(round(t_trial / dt))
    learner = GrayBoxLearner(n_unlocked=3, dt=dt)
    learner.taus = np.array(TAU_TRUE, dtype=float)
    for _ in range(n_trials):
        ref = reach_reference(n_t, dt, t_move)
        u = ref + u_explore * band_limited_noise(n_t, dt, t_move, rng)
        y = cascade(experienced_taus(TAU_TRUE, xi), u, dt) + noise_std * rng.normal(size=n_t)
        learner.trials.append((u, y, xi))

    q = np.log(np.array(TAU_TRUE))
    r = learner._residual(q)
    J = learner._jacobian(q, r)
    ev, V = np.linalg.eigh((J.T @ J) / r.size)
    cost = np.array([model_error(tuple(np.exp(q + STEP * V[:, i])), dt=dt) for i in range(3)])
    return {
        "xi": float(xi),
        "rho": float(rho(xi)),
        "t_move": float(t_move),
        "curvature": ev,
        "vectors": V,
        "behaviour_cost": cost,
        "kappa": float(ev[-1] / max(ev[0], 1e-18)),
    }


def sweep(xis=(0.7, 1.0, 1.3, 2.0, 3.0, 4.0), t_move=0.25, **kw):
    return [curvature_spectrum(xi, t_move, **kw) for xi in xis]


def report(rows) -> str:
    out = []
    for d in rows:
        out.append(
            f"xi={d['xi']:.1f} rho={d['rho']:6.1f} kappa={d['kappa']:.2e}  "
            + "  ".join(
                f"[lam={d['curvature'][i]:.2e} cost={d['behaviour_cost'][i]:.4f}]"
                for i in range(3)
            )
        )
    return "\n".join(out)
