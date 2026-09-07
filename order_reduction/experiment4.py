# -*- coding: utf-8 -*-
"""L1-L6: do we co-contract in order to learn, and does it learn faster?

The earlier learners asked "how fast can you reconstruct the whole
third-order object?" and concluded that co-contraction does not shorten
that path. This experiment asks the question the meeting note actually
needs, and it asks it of every strategy on the same footing:

    if you do not co-contract you must identify the whole third-order
    plant. If you co-contract, the coupling has already filtered the
    transients and you are identifying a first-order plant. Is that
    second path faster?

Six conditions. Movement time is held at the fast reach in every one,
so any filter is coming from xi and not from slowing down (K4 already
showed that bandwidth staging alone is fast).

    L1  soft throughout,  third-order EKF   no filter: learn all three modes
    L2  stiff throughout, first-order EKF   co-contract, learn the slow part,
                                            treat the transients as noise
    L3  soft throughout,  first-order EKF   same capacity as L2, no filter
    L4  stiff throughout, third-order EKF   filter present, capacity unused
    L5  staged,           third-order EKF   co-contract, then relax and learn
                                            the whole model gradually
    L6  staged,           first -> third    learn the slow lag while stiff,
                                            then warm-start the full model

L1, L2 and L5 are the three strategies a person could actually adopt.
L3 and L4 are the controls that say which half of L2 is doing the work.
L6 is the strongest version of the hypothesis: that a correct slow model,
learned cheaply under co-contraction, is a head start on the full one.

Three scoreboards, because the choice of scoreboard is the whole
methodological point:

    slow      |tau_hat_d - tau_d|, the part a person keeps
    full      impulse error of the belief about the whole object
    test      relative error predicting a soft, fast reach

The test scoreboard is the fair one. Every condition is evaluated on the
same deterministic soft fast probe, whatever stiffness it happens to be
holding, so "I can only do this while braced" shows up as a cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import predicted_tau_hat
from .kalman_learner import I_TAU_D, TwoTimescaleEKF
from .plant import (
    TAU_TRUE,
    XI_SOFT,
    band_limited_noise,
    cascade_exact,
    experienced_taus,
    reach_reference,
)
from .slow_learner import FirstOrderEKF

DT = 0.01
T_TRIAL = 2.5
NOISE_STD = 0.012
U_EXPLORE = 0.10
S_BASE = 0.15
MIN_USABLE = 20
FAST_MOVE = 0.25
XI_STIFF = 3.0

# Staged schedule: stiffness only. Movement time never changes.
XI_STAGES = (XI_STIFF, 1.3, XI_SOFT)
MIN_TRIALS_PER_STAGE = 5
CONFIDENCE_SD = 0.04   # posterior sd on log tau_d that licenses relaxing

# Slow-mode criterion: tighter than the soft first-order bias floor
# (~0.15 s), loose enough that a well-specified filter can hit it.
SLOW_CRITERION = 0.08
FULL_CRITERION = 0.10   # same threshold experiment2 used on object error
TEST_CRITERION = 0.10   # relative error on the common soft fast probe
CRITERION_STREAK = 3

CONDITIONS = ("L1", "L2", "L3", "L4", "L5", "L6")

CONDITION_META = {
    "L1": {
        "label": "L1 soft, learn 3rd",
        "short": "no co-contraction, full model",
        "color": "#8C1428",
        "schedule": "soft",
        "order": 3,
    },
    "L2": {
        "label": "L2 stiff, learn slow only",
        "short": "co-contract, slow model only",
        "color": "#1B365D",
        "schedule": "stiff",
        "order": 1,
    },
    "L3": {
        "label": "L3 soft, learn slow only",
        "short": "control: capacity without filter",
        "color": "#C5761A",
        "schedule": "soft",
        "order": 1,
    },
    "L4": {
        "label": "L4 stiff, learn 3rd",
        "short": "control: filter without restriction",
        "color": "#5C5C5C",
        "schedule": "stiff",
        "order": 3,
    },
    "L5": {
        "label": "L5 staged, learn 3rd",
        "short": "co-contract, then learn the whole model gradually",
        "color": "#1E6B3A",
        "schedule": "staged",
        "order": 3,
    },
    "L6": {
        "label": "L6 staged, slow then recruit",
        "short": "learn the slow lag stiff, then warm-start the full model",
        "color": "#6B2D8C",
        "schedule": "staged",
        "order": "promote",
    },
}


def _make_learner(order, dt: float):
    if order == 1 or order == "promote":
        return FirstOrderEKF(dt=dt)
    if order == 3:
        return TwoTimescaleEKF(dt=dt)
    raise ValueError(order)


def _promote(first_order: FirstOrderEKF, dt: float) -> TwoTimescaleEKF:
    """Hand the slow lag, and how sure we are of it, to a full-order learner.

    This is the explicit form of "learn the slow part first, then learn the
    rest on top of it". Only the dominant channel is transferred; the
    transient time constants and their participation weights start from
    the same prior any fresh third-order learner would use, so L6 gets a
    head start on tau_d and nothing else.
    """
    full = TwoTimescaleEKF(dt=dt)
    full.q[I_TAU_D] = float(first_order.q[0])
    full.P[I_TAU_D, I_TAU_D] = float(first_order.P[0, 0])
    return full


def _impulse(taus, dt: float, n: int) -> np.ndarray:
    u = np.zeros(n)
    u[0] = 1.0 / dt
    return cascade_exact(taus, u, dt) * dt


def object_error(learner, dt: float = DT, n: int = 400) -> float:
    """Full-object impulse error: the scoreboard S3/K3 were failed on."""
    y_true = _impulse(TAU_TRUE, dt, n)
    y_hat = learner.object_impulse(n)
    return float(
        np.sqrt(np.mean((y_true - y_hat) ** 2)) / (np.sqrt(np.mean(y_true**2)) + 1e-12)
    )


def slow_error(tau_hat_d: float, dt: float = DT, n: int = 400) -> float:
    """Impulse error of the slow part alone: 1/(1+tau_hat s) vs 1/(1+tau_d s)."""
    y_true = _impulse((TAU_TRUE[0], 0.0, 0.0), dt, n)
    y_hat = _impulse((tau_hat_d, 0.0, 0.0), dt, n)
    return float(
        np.sqrt(np.mean((y_true - y_hat) ** 2)) / (np.sqrt(np.mean(y_true**2)) + 1e-12)
    )


def make_test_probe(dt: float = DT):
    """One deterministic soft, fast reach, and what it truly feels like.

    Noise-free and identical for every condition and every seed, so the
    test scoreboard compares beliefs and nothing else. The test is always
    taken *soft*: the question is whether you have learned the object, not
    whether you can cope while still braced against it.
    """
    n_t = int(round(T_TRIAL / dt))
    u = reach_reference(n_t, dt, FAST_MOVE)
    y = cascade_exact(experienced_taus(TAU_TRUE, XI_SOFT), u, dt)
    return u, y


def test_error(learner, probe_u: np.ndarray, probe_y: np.ndarray) -> float:
    """Relative error predicting the soft fast probe. The common scoreboard."""
    y_hat = learner.predict(probe_u, XI_SOFT)
    return float(
        np.sqrt(np.mean((probe_y - y_hat) ** 2)) / (np.sqrt(np.mean(probe_y**2)) + 1e-12)
    )


@dataclass
class TrialLog:
    err: np.ndarray            # full-object impulse error
    slow_err: np.ndarray       # slow-part impulse error
    test_err: np.ndarray       # common soft fast probe, the fair scoreboard
    bias_d: np.ndarray         # tau_hat_d - tau_d
    felt_bias: np.ndarray      # first-order cells only; NaN for third-order
    pred_rmse: np.ndarray      # felt prediction RMSE at the held stiffness
    n_eff: np.ndarray
    sd_log_tau_d: np.ndarray
    xi: np.ndarray
    t_move: np.ndarray
    stage: np.ndarray
    failed: np.ndarray
    ttc_slow: int
    ttc_full: int
    ttc_test: int
    n_failed: int
    seed: int
    condition: str


def _xi_for(condition: str, stage: int) -> float:
    schedule = CONDITION_META[condition]["schedule"]
    if schedule == "soft":
        return XI_SOFT
    if schedule == "stiff":
        return XI_STIFF
    if schedule == "staged":
        return float(XI_STAGES[stage])
    raise ValueError(schedule)


def _mark_ttc(value: float, threshold: float, streak: int, ttc: int, n: int, n_trials: int):
    if value < threshold:
        streak += 1
        if streak >= CRITERION_STREAK and ttc == n_trials + 1:
            ttc = n + 1
    else:
        streak = 0
    return streak, ttc


def run_seed(condition: str, seed: int, n_trials: int = 60, dt: float = DT) -> TrialLog:
    meta = CONDITION_META[condition]
    rng = np.random.default_rng(seed)
    n_t = int(round(T_TRIAL / dt))
    t_move = FAST_MOVE
    staged = meta["schedule"] == "staged"
    stage = 0

    learner = _make_learner(meta["order"], dt)
    probe_u, probe_y = make_test_probe(dt) # make_test_probe is a function that makes a test probe, the test probe is a deterministic soft, fast reach, and what it truly feels like

    err = np.zeros(n_trials) # err is is used to store the full-object impulse error
    slow_err = np.zeros(n_trials) # slow_err is used to store the slow-part impulse error
    test_err = np.zeros(n_trials) # test_err is used to store the common soft fast probe, the fair scoreboard
    bias_d = np.zeros(n_trials) # bias_d is used to store the bias of the tau_hat_d - tau_d
    felt_bias = np.full(n_trials, np.nan) # felt_bias is used to store the felt bias of the tau_hat_d - tau_d
    pred_rmse = np.full(n_trials, np.nan) # pred_rmse is used to store the predicted RMSE of the tau_hat_d - tau_d
    n_eff = np.zeros(n_trials) # n_eff is used to store the effective number of parameters
    sd_d = np.full(n_trials, np.nan) # sd_d is used to store the standard deviation of the log of the tau_d
    xi_log = np.zeros(n_trials) # xi_log is used to store the log of the xi
    tm_log = np.full(n_trials, t_move) # tm_log is used to store the time of the move
    stage_log = np.zeros(n_trials, dtype=int) # stage_log is used to store the stage
    failed_log = np.zeros(n_trials, dtype=bool) # failed_log is used to store the failed trials

    streak_s = streak_u = streak_t = 0
    ttc_slow = ttc_full = ttc_test = n_trials + 1
    trials_in_stage = 0

    for n in range(n_trials):
        xi = _xi_for(condition, stage)

        ref = reach_reference(n_t, dt, t_move) # this is the planned motion of hand to the target we send to plant . The experiment asks The experiment asks: given that motion and how hard you grip, can your brain learn what the cup feels like — and is co-contraction a smart way to learn?
        u = ref + U_EXPLORE * band_limited_noise(n_t, dt, t_move, rng) # we always do reach out, pause, acom back reaching, but everytime our path woblles a little.
        y = cascade_exact(experienced_taus(TAU_TRUE, xi), u, dt) + NOISE_STD * rng.normal(size=n_t) # given, the hand motion u, what does the cup feel like? (actually measured)
        y_hat = learner.predict(u, xi) # given the hand motion u, what does the cup feel like? (predicted)

        tolerance = S_BASE * xi / XI_SOFT
        over = np.flatnonzero(np.abs(y - y_hat) > tolerance)
        failed = over.size > 0
        stop = int(over[0]) if failed else n_t

        if stop >= MIN_USABLE:
            stats = learner.observe(u[:stop], y[:stop], xi)
            pred_rmse[n] = stats["pred_rmse"]
            sd_d[n] = stats["sd_log_tau_d"]
        else:
            learner.drift()

        tau_hat = float(learner.tau_d)
        err[n] = object_error(learner, dt=dt)
        slow_err[n] = slow_error(tau_hat, dt=dt)
        test_err[n] = test_error(learner, probe_u, probe_y)
        bias_d[n] = tau_hat - TAU_TRUE[0]
        if isinstance(learner, FirstOrderEKF):
            felt_bias[n] = tau_hat - predicted_tau_hat(xi)
        n_eff[n] = learner.n_eff
        xi_log[n] = xi
        stage_log[n] = stage
        failed_log[n] = failed

        streak_s, ttc_slow = _mark_ttc(
            abs(bias_d[n]), SLOW_CRITERION, streak_s, ttc_slow, n, n_trials
        )
        streak_u, ttc_full = _mark_ttc(
            err[n], FULL_CRITERION, streak_u, ttc_full, n, n_trials
        )
        streak_t, ttc_test = _mark_ttc(
            test_err[n], TEST_CRITERION, streak_t, ttc_test, n, n_trials
        )

        if staged and stage < len(XI_STAGES) - 1:
            trials_in_stage += 1
            confident = np.isfinite(sd_d[n]) and sd_d[n] < CONFIDENCE_SD
            if trials_in_stage >= MIN_TRIALS_PER_STAGE and confident:
                stage += 1
                trials_in_stage = 0
                if meta["order"] == "promote" and isinstance(learner, FirstOrderEKF):
                    learner = _promote(learner, dt)

    return TrialLog(
        err=err,
        slow_err=slow_err,
        test_err=test_err,
        bias_d=bias_d,
        felt_bias=felt_bias,
        pred_rmse=pred_rmse,
        n_eff=n_eff,
        sd_log_tau_d=sd_d,
        xi=xi_log,
        t_move=tm_log,
        stage=stage_log,
        failed=failed_log,
        ttc_slow=int(ttc_slow),
        ttc_full=int(ttc_full),
        ttc_test=int(ttc_test),
        n_failed=int(failed_log.sum()),
        seed=seed,
        condition=condition,
    )


FAILURE_COSTS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def effective_ttc(log: TrialLog, key: str, cost: float) -> float:
    """Trials to criterion when a failed attempt costs `cost` extra trials.

    The simulation charges nothing for a failure beyond losing the rest of
    that trial's data, which is the most generous possible assumption for a
    learner that goes straight to the deep end. For a person it is rarely
    free: a dropped cup has to be retrieved and refilled, a stalled bicycle
    has to be remounted, and in the clinical framing the cost is not paid
    in trials at all. This recomputes each condition's trial count with a
    recovery overhead charged per failure, which is the sweep
    `report/theory.tex` pre-registered as the way the ordering might flip.
    """
    n_trials = int(log.failed.size)
    ttc = int(getattr(log, key))
    reached = ttc <= n_trials
    horizon = min(ttc, n_trials)
    n_fail = float(log.failed[:horizon].sum())
    base = float(ttc if reached else n_trials + 1)
    return base + cost * n_fail


def break_even_cost(logs, faster: str, slower: str, key: str,
                    costs=FAILURE_COSTS) -> float | None:
    """Smallest failure cost at which `faster` overtakes `slower` on `key`."""
    for cost in costs:
        a = np.median([effective_ttc(lg, key, cost) for lg in logs[faster]])
        b = np.median([effective_ttc(lg, key, cost) for lg in logs[slower]])
        if a <= b:
            return float(cost)
    return None


def run_experiment(n_trials=60, n_seeds=20, seed0=0, conditions=CONDITIONS):
    return {
        c: [
            run_seed(c, seed=seed0 + 1000 * (CONDITIONS.index(c) + 1) + i, n_trials=n_trials)
            for i in range(n_seeds)
        ]
        for c in conditions
    }
