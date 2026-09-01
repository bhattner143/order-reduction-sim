# -*- coding: utf-8 -*-
"""Third-order object plus body tuning (xi) and command bandwidth (omega_c)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete

TAU_D = 1.0
TAU_1 = 0.1
TAU_2 = 0.05
TRUE_POLES = np.array([-1.0 / TAU_D, -1.0 / TAU_1, -1.0 / TAU_2])


def separation_ratio(xi: float) -> float:
    """rho(xi) from the overdamped quadratic. Equals 1 at and below xi = 1."""
    if xi <= 1.0:
        return 1.0
    return float((xi + np.sqrt(xi * xi - 1.0)) ** 2)


def experienced_time_constants(xi: float) -> tuple[float, float, float]:
    rho = separation_ratio(xi)
    return TAU_D, TAU_1 / rho, TAU_2 / rho


def experienced_poles(xi: float) -> np.ndarray:
    td, t1, t2 = experienced_time_constants(xi)
    return np.array([-1.0 / td, -1.0 / t1, -1.0 / t2])


def _cascade_ss(taus: tuple[float, float, float]):
    td, t1, t2 = taus
    A = np.array(
        [
            [-1.0 / td, 0.0, 0.0],
            [1.0 / t1, -1.0 / t1, 0.0],
            [0.0, 1.0 / t2, -1.0 / t2],
        ],
        dtype=float,
    )
    B = np.array([[1.0 / td], [0.0], [0.0]], dtype=float)
    C = np.array([[0.0, 0.0, 1.0]], dtype=float)
    D = np.array([[0.0]], dtype=float)
    return A, B, C, D


def true_continuous_ss():
    return _cascade_ss((TAU_D, TAU_1, TAU_2))


@dataclass
class CoupledPlant:
    dt: float
    xi: float = 0.7
    omega_c: float = 40.0

    def __post_init__(self) -> None:
        self._rebuild()

    def set_tuning(self, xi: float, omega_c: float) -> None:
        self.xi = float(xi)
        self.omega_c = float(omega_c)
        self._rebuild()

    def _rebuild(self) -> None:
        A, B, C, D = _cascade_ss(experienced_time_constants(self.xi))
        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), self.dt, method="zoh")
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float).reshape(3, 1)
        self.Cd = np.asarray(Cd, dtype=float).reshape(1, 3)
        self.Dd = float(np.asarray(Dd).reshape(()))
        a = float(np.exp(-self.omega_c * self.dt))
        self.filt_a = a
        self.filt_b = 1.0 - a

    @property
    def poles(self) -> np.ndarray:
        return experienced_poles(self.xi)

    def reset(self):
        return np.zeros(3), 0.0

    def step(self, x: np.ndarray, u_filt: float, u_cmd: float):
        u_next = self.filt_a * u_filt + self.filt_b * u_cmd
        x_next = (self.Ad @ x.reshape(3, 1) + self.Bd * u_next).ravel()
        y = float((self.Cd @ x_next).ravel()[0]) + self.Dd * u_next
        return x_next, u_next, y


def min_jerk_reach(t: np.ndarray, t_move: float, y_final: float = 1.0) -> np.ndarray:
    r = np.empty_like(t, dtype=float)
    s = np.clip(t / t_move, 0.0, 1.0)
    r[:] = y_final * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
    r[t >= t_move] = y_final
    return r
