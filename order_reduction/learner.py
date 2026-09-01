# -*- coding: utf-8 -*-
"""Eigensystem realisation (ERA): keep the r largest Hankel singular values."""
from __future__ import annotations

import numpy as np


def _hankel(markov: np.ndarray, rows: int, cols: int) -> np.ndarray:
    H = np.empty((rows, cols), dtype=float)
    for i in range(rows):
        H[i, :] = markov[i : i + cols]
    return H


def era(markov: np.ndarray, order: int, rows: int = 50, cols: int = 50):
    """Return (A, B, C, D) of the given order from an impulse response."""
    need = rows + cols - 1
    if markov.size < need:
        markov = np.pad(markov, (0, need - markov.size))
    D = 0.0
    h = markov[:need]
    H = _hankel(h, rows, cols)
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    r = min(order, int(np.count_nonzero(s > 1e-12)), U.shape[1])
    if r < 1:
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), D, s
    Sig = np.diag(np.sqrt(s[:r]))
    Ok = U[:, :r] @ Sig
    Cr = Sig @ Vt[:r, :]
    O1, O2 = Ok[:-1, :], Ok[1:, :]
    A = np.linalg.pinv(O1) @ O2
    B = Cr[:, :1]
    C = Ok[:1, :]
    return A, B, C, D, s


def simulate_ss_impulse(A, B, C, D, n: int) -> np.ndarray:
    y = np.zeros(n)
    x = np.zeros(A.shape[0])
    for k in range(n):
        u = 1.0 if k == 0 else 0.0
        x = (A @ x.reshape(-1, 1) + B * u).ravel()
        y[k] = float((C @ x).ravel()[0] + D * u)
    return y


class ERAIdentifier:
    def __init__(self, order: int):
        self.order = int(order)
        self.A = np.zeros((max(order, 1), max(order, 1)))
        self.B = np.zeros((max(order, 1), 1))
        self.C = np.zeros((1, max(order, 1)))
        self.D = 0.0
        self.svd = np.ones(6)
        self.imp_hat = np.zeros(1)
        self.n_avg = 0
        self.y_mean = None

    def expand_order(self, new_order: int) -> None:
        self.order = int(new_order)
        self.n_avg = 0
        self.y_mean = None

    def update_trial(self, y: np.ndarray, u: np.ndarray) -> dict:
        y = np.asarray(y, dtype=float)
        if self.y_mean is None:
            self.y_mean = y.copy()
            self.n_avg = 1
        else:
            self.n_avg += 1
            self.y_mean += (y - self.y_mean) / self.n_avg
        markov = self.y_mean
        A, B, C, D, s = era(markov, self.order)
        imp = simulate_ss_impulse(A, B, C, D, markov.size)
        scale = float(np.sum(markov) / (np.sum(imp) + 1e-12))
        C = C * scale
        self.A, self.B, self.C, self.D, self.svd = A, B, C, D, s
        self.imp_hat = simulate_ss_impulse(A, B, C, D, y.size)
        pred_rmse = float(np.sqrt(np.mean((y - self.imp_hat) ** 2)))
        s = np.asarray(s, dtype=float)
        if s.size >= 2 and s[1] > 1e-12:
            cond = float(s[0] / max(s[min(self.order, s.size) - 1], 1e-12))
        else:
            cond = float("inf")
        return {"pred_rmse": pred_rmse, "cond_phi": cond, "n_samples": int(y.size)}
