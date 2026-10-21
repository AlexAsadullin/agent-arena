"""Pairwise comparisons: bootstrap win probabilities, Bradley-Terry strengths and significance.

* Bootstrap win probability ``P(A > B)``: the share of paired bootstrap replicates in which A's
  mean metric exceeds B's (ties count one half).
* Bradley-Terry (Bradley and Terry 1952): ``P(i beats j) = p_i / (p_i + p_j)``, fitted to
  per-episode head-to-head outcomes with the minorisation-maximisation algorithm of Hunter (2004),
  ``p_i <- W_i / sum_j n_ij / (p_i + p_j)``. A symmetric pseudo-count prior on every pair keeps
  the estimate finite when an agent wins or loses every episode.
* Significance against a baseline: one-sided paired bootstrap test of H0 ``mean(A - B) <= 0``,
  with the shifted-null p-value of Efron and Tibshirani (1993, section 16.4), and the Holm (1979)
  step-down correction across all agents tested.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bootstrap_win_probability(boot_a: FloatArray, boot_b: FloatArray) -> float:
    ok = np.isfinite(boot_a) & np.isfinite(boot_b)
    if not ok.any():
        return float("nan")
    a, b = boot_a[ok], boot_b[ok]
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


def episode_win_matrix(episode_means: FloatArray) -> FloatArray:
    """``W[i, j]`` = episodes where agent i beat agent j (ties count one half).

    ``episode_means`` has shape ``(n_agents, n_episodes)``.
    """
    a = episode_means[:, None, :]
    b = episode_means[None, :, :]
    valid = np.isfinite(a) & np.isfinite(b)
    wins = np.where(valid, (a > b) + 0.5 * (a == b), 0.0).sum(axis=2)
    np.fill_diagonal(wins, 0.0)
    return np.asarray(wins, dtype=np.float64)


def bradley_terry(wins: FloatArray, prior: float, max_iter: int, tol: float) -> FloatArray:
    """Bradley-Terry strengths (geometric mean 1) from a win matrix, by Hunter's MM algorithm."""
    n = wins.shape[0]
    w = wins + prior * (1.0 - np.eye(n))
    games = w + w.T
    total_wins = w.sum(axis=1)
    p = np.ones(n)
    for _ in range(max_iter):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = total_wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    return np.asarray(p, dtype=np.float64)


def bt_win_probability(strengths: FloatArray) -> FloatArray:
    """``P[i, j] = p_i / (p_i + p_j)``."""
    return np.asarray(strengths[:, None] / (strengths[:, None] + strengths[None, :]), dtype=np.float64)


def paired_bootstrap_pvalue(observed_diff: float, boot_diff: FloatArray) -> float:
    """One-sided p-value for H0: mean difference <= 0, from the shifted bootstrap null.

    p = (1 + #{b : boot_b - observed >= observed}) / (B + 1).
    """
    finite = boot_diff[np.isfinite(boot_diff)]
    if finite.size == 0 or not np.isfinite(observed_diff):
        return float("nan")
    exceed = int(np.sum(finite - observed_diff >= observed_diff))
    return (1 + exceed) / (finite.size + 1)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaN entries pass through unchanged)."""
    indexed = [(p, i) for i, p in enumerate(pvalues) if np.isfinite(p)]
    indexed.sort()
    m = len(indexed)
    adjusted = list(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted
