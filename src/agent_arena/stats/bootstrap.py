"""Paired two-level bootstrap over episodes and replicates.

Data layout: for each agent a matrix ``M`` of shape ``(n_episodes, n_replicates)`` holding one
metric per run (e.g. the run's net Sharpe). Replicates are the ``seed x prompt variant`` runs of
that agent inside an episode; NaN marks a missing run.

One bootstrap replicate (Davison and Hinkley 1997, section 3.8, two-stage resampling):

1. draw ``n_episodes`` episode indices with replacement - the *same* draw for every agent, so
   comparisons stay paired on the market path;
2. inside each drawn episode, draw replicate indices with replacement for each agent;
3. the statistic is the mean over drawn episodes of the mean over drawn replicates.

Stage 1 captures market (regime) uncertainty, stage 2 captures seed and prompt sensitivity, which
a single-run leaderboard ignores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EpisodeDraws:
    """Episode indices shared by all agents: shape ``(n_boot, n_episodes)``."""

    index: NDArray[np.int64]

    @classmethod
    def draw(cls, n_episodes: int, n_boot: int, rng: np.random.Generator) -> EpisodeDraws:
        return cls(rng.integers(0, n_episodes, size=(n_boot, n_episodes)))


def point_estimate(matrix: FloatArray) -> float:
    """Mean over episodes of the per-episode replicate mean."""
    return float(np.nanmean(np.nanmean(matrix, axis=1)))


def bootstrap_means(matrix: FloatArray, draws: EpisodeDraws, rng: np.random.Generator) -> FloatArray:
    """Bootstrap distribution of :func:`point_estimate` (length ``n_boot``)."""
    n_boot, n_episodes = draws.index.shape
    n_rep = matrix.shape[1]
    picks = rng.integers(0, n_rep, size=(n_boot, n_episodes, n_rep))
    episode_rows = matrix[draws.index]  # (n_boot, n_episodes, n_rep)
    values = np.take_along_axis(episode_rows, picks, axis=2)
    with np.errstate(invalid="ignore"):
        per_episode = np.nanmean(values, axis=2)
        return np.asarray(np.nanmean(per_episode, axis=1), dtype=np.float64)


def percentile_ci(samples: FloatArray, level: float) -> tuple[float, float]:
    """Percentile confidence interval of a bootstrap distribution."""
    tail = (1.0 - level) / 2.0
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(finite, [tail, 1.0 - tail])
    return float(lo), float(hi)
