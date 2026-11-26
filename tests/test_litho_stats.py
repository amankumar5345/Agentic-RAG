"""Hand-computed tests for the statistics helpers."""

import pytest

from litho_eval.stats import (
    bootstrap_ci,
    paired_bootstrap_ci,
    percentile,
    proportion_ci,
    wilson_interval,
)


class FixedRng:
    """Always resamples index 0."""

    def randrange(self, n):
        return 0


def test_percentile_hand_values():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.95) == pytest.approx(3.85)


def test_wilson_hand_values():
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert upper == pytest.approx(0.2775, abs=1e-4)

    lower, upper = wilson_interval(10, 10)
    assert lower == pytest.approx(0.7225, abs=1e-4)
    assert upper == 1.0


def test_proportion_ci_uses_wilson_below_fifty():
    assert proportion_ci(0, 10) == wilson_interval(0, 10)
    assert proportion_ci(50, 100) != wilson_interval(50, 100)


def test_bootstrap_constant_values():
    assert bootstrap_ci([5.0, 5.0, 5.0]) == (5.0, 5.0)


def test_bootstrap_with_fixed_rng_returns_first_value():
    assert bootstrap_ci([1.0, 10.0], rng=FixedRng()) == (1.0, 1.0)


def test_paired_bootstrap_constant_difference():
    assert paired_bootstrap_ci([3.0, 3.0], [2.0, 2.0]) == (1.0, 1.0)


def test_paired_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])
