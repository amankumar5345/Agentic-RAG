"""Bootstrap and proportion intervals for LithoBench results."""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Sequence

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1] (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling at the case level."""
    if not values:
        raise ValueError("bootstrap of an empty sequence")
    generator = rng or random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_resamples):
        sample = [values[generator.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    return percentile(stats, alpha / 2), percentile(stats, 1 - alpha / 2)


def paired_bootstrap_ci(
    arm: Sequence[float],
    baseline: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for the paired difference arm - baseline (same cases)."""
    if len(arm) != len(baseline):
        raise ValueError("paired samples must have the same length")
    differences = [float(left) - float(right) for left, right in zip(arm, baseline, strict=True)]
    return bootstrap_ci(differences, statistic, n_resamples, seed, alpha, rng)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def proportion_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson for n < 50, otherwise the normal approximation."""
    if n <= 0:
        return 0.0, 1.0
    if n < 50:
        return wilson_interval(successes, n, z)
    proportion = successes / n
    margin = z * math.sqrt(proportion * (1 - proportion) / n)
    return max(0.0, proportion - margin), min(1.0, proportion + margin)
