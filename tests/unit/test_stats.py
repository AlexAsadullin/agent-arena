from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from agent_arena.stats.bootstrap import EpisodeDraws, bootstrap_means, percentile_ci, point_estimate
from agent_arena.stats.pairwise import (
    bootstrap_win_probability,
    bradley_terry,
    bt_win_probability,
    episode_win_matrix,
    holm_adjust,
    paired_bootstrap_pvalue,
)
from agent_arena.stats.sharpe import (
    annualise,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_sharpe_ratio_closed_form() -> None:
    returns = [0.01, -0.01, 0.02, 0.0]
    assert sharpe_ratio(returns) == pytest.approx(np.mean(returns) / np.std(returns, ddof=1))
    assert sharpe_ratio([0.01, 0.01]) == 0.0
    assert sharpe_ratio([0.01]) == 0.0
    assert annualise(0.1, 100) == pytest.approx(1.0)


def test_psr_matches_normal_formula() -> None:
    # Under normality (skew 0, kurtosis 3): PSR = Phi(SR sqrt(T-1) / sqrt(1 + SR^2 / 2)).
    sr, n = 0.1, 250
    expected = norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe_ratio(sr, n, 0.0, 3.0) == pytest.approx(expected)
    assert probabilistic_sharpe_ratio(0.0, n, 0.0, 3.0) == pytest.approx(0.5)
    assert math.isnan(probabilistic_sharpe_ratio(0.1, 1, 0.0, 3.0))


def test_expected_max_sharpe_reference_value() -> None:
    # Closed form for N = 1000, V = 1: 0.4228 * Phi^-1(0.999) + 0.5772 * Phi^-1(1 - 1/(1000 e)).
    reference = (1 - 0.5772156649) * norm.ppf(0.999) + 0.5772156649 * norm.ppf(1 - 1 / (1000 * math.e))
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(reference)
    assert expected_max_sharpe(1000, 4.0) == pytest.approx(2 * reference)
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0


def test_expected_max_sharpe_against_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((20000, 50)).max(axis=1)
    assert expected_max_sharpe(50, 1.0) == pytest.approx(draws.mean(), rel=0.03)


def test_dsr_drops_with_more_trials() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.002, 0.01, 500)
    few = deflated_sharpe_ratio(returns, list(rng.normal(0, 0.05, 3)))
    many = deflated_sharpe_ratio(returns, list(rng.normal(0, 0.05, 300)))
    assert 0 <= many < few <= 1
    assert math.isnan(deflated_sharpe_ratio([0.0, 0.0, 0.0], [0.1, 0.2]))


def test_two_level_bootstrap_ci_covers_mean() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.normal(1.0, 0.5, (40, 6))
    draws = EpisodeDraws.draw(40, 2000, rng)
    boot = bootstrap_means(matrix, draws, rng)
    lo, hi = percentile_ci(boot, 0.95)
    assert lo < point_estimate(matrix) < hi
    assert lo < 1.0 < hi
    # The standard error of the mean of 40 episode means of 6 draws is about 0.5 / sqrt(240).
    assert np.std(boot) == pytest.approx(0.5 / math.sqrt(240), rel=0.35)


def test_bootstrap_handles_nan_padding() -> None:
    matrix = np.array([[1.0, np.nan], [3.0, 3.0]])
    assert point_estimate(matrix) == pytest.approx(2.0)
    rng = np.random.default_rng(3)
    boot = bootstrap_means(matrix, EpisodeDraws.draw(2, 200, rng), rng)
    assert np.all((boot >= 1.0) & (boot <= 3.0))
    assert percentile_ci(np.array([np.nan]), 0.9) != percentile_ci(np.array([1.0]), 0.9)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-5, 5), min_size=2, max_size=12), st.integers(1, 4))
def test_bootstrap_means_stay_in_range(values: list[float], reps: int) -> None:
    rng = np.random.default_rng(4)
    matrix = np.tile(np.asarray(values)[:, None], (1, reps))
    boot = bootstrap_means(matrix, EpisodeDraws.draw(len(values), 100, rng), rng)
    assert boot.min() >= min(values) - 1e-9
    assert boot.max() <= max(values) + 1e-9


def test_win_probability_and_pvalue() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 2.0, 2.0, 5.0])
    assert bootstrap_win_probability(a, b) == pytest.approx(0.625)
    assert math.isnan(bootstrap_win_probability(np.array([np.nan]), np.array([1.0])))
    rng = np.random.default_rng(5)
    strong = rng.normal(1.0, 0.1, 1000)
    assert paired_bootstrap_pvalue(1.0, strong) < 0.01
    null = rng.normal(0.0, 0.1, 1000)
    assert 0.3 < paired_bootstrap_pvalue(0.0, null) < 0.7
    assert math.isnan(paired_bootstrap_pvalue(float("nan"), null))


def test_holm_reference_example() -> None:
    # Holm (1979): p = (0.01, 0.04, 0.03) -> (0.03, 0.06, 0.06).
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm_adjust([float("nan"), 0.02])
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(0.02)


def test_bradley_terry_recovers_strengths() -> None:
    rng = np.random.default_rng(6)
    true = np.array([4.0, 2.0, 1.0])
    wins = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                wins[i, j] = rng.binomial(2000, true[i] / (true[i] + true[j]))
    wins = (wins + (2000 - wins.T)) / 2 * (1 - np.eye(3))
    est = bradley_terry(wins, 0.0, 2000, 1e-12)
    assert est / est[2] == pytest.approx(true, rel=0.1)
    probs = bt_win_probability(est)
    assert probs[0, 1] + probs[1, 0] == pytest.approx(1.0)


def test_bradley_terry_prior_keeps_unbeaten_finite() -> None:
    wins = episode_win_matrix(np.array([[3.0, 3.0, 3.0], [1.0, 2.0, 1.0], [0.0, 2.0, 0.0]]))
    assert wins[0, 1] == 3.0
    assert wins[1, 2] == 2.5
    est = bradley_terry(wins, 0.5, 500, 1e-9)
    assert np.all(np.isfinite(est))
    assert est[0] > est[1] > est[2]
