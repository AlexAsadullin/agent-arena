"""Sharpe ratio, probabilistic Sharpe ratio (PSR) and deflated Sharpe ratio (DSR).

This is the *local fallback* for when the S6 ledger verdict is unavailable; the leaderboard
labels which source each DSR came from.

References:
    Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier.
    Journal of Risk 15(2). (PSR, eq. 3)
    Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for
    selection bias, backtest overfitting and non-normality. Journal of Portfolio Management
    40(5). (DSR, expected maximum Sharpe ratio, eqs. 2-3)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: Sequence[float] | NDArray[np.float64]) -> float:
    """Per-period Sharpe ratio ``mean / std`` (ddof=1, zero risk-free rate); 0 if undefined."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    return float(arr.mean()) / sd if sd > 0 else 0.0


def annualise(sr_per_period: float, periods_per_year: float) -> float:
    return sr_per_period * math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skewness: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ).

    ``sr`` and ``sr_benchmark`` are per period; ``kurt`` is the (non-excess) kurtosis g4.
    """
    if n_obs < 2:
        return float("nan")
    denom_sq = 1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom_sq <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] over N independent trials with SR variance V (per period):

    sqrt(V) * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))).
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_MASCHERONI) * float(a) + EULER_MASCHERONI * float(b))


def deflated_sharpe_ratio(returns: Sequence[float], trial_sharpes: Sequence[float]) -> float:
    """DSR = PSR(SR0) with SR0 = expected max Sharpe across the season's trials (per period)."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 3 or float(arr.std()) == 0.0:
        return float("nan")
    trials = np.asarray(trial_sharpes, dtype=np.float64)
    sr0 = expected_max_sharpe(int(trials.size), float(trials.var(ddof=1)) if trials.size > 1 else 0.0)
    return probabilistic_sharpe_ratio(
        sharpe_ratio(arr),
        int(arr.size),
        float(skew(arr)),
        float(kurtosis(arr, fisher=False)),
        sr0,
    )
