import math
import pytest
from gatekeep.metrics import bootstrap_ci, coverage_curve, ece, macro_f1, pick_tau


def test_ece_is_zero_when_confident_and_right():
    assert ece([1.0] * 10, [1] * 10) == 0


def test_ece_overconfident():
    assert ece([0.95] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.45)


def test_coverage_curve():
    (t, cov, acc), = coverage_curve([0.9, 0.6, 0.8], [1, 0, 1], [0.7])
    assert (t, cov, acc) == (0.7, pytest.approx(2 / 3), 1.0)


def test_coverage_curve_empty_selection_is_nan():
    (_, cov, acc), = coverage_curve([0.5], [1], [0.9])
    assert cov == 0 and math.isnan(acc)


def test_pick_tau_finds_smallest_clean_threshold():
    assert pick_tau([0.6, 0.9, 0.95], [0, 1, 1], target=1.0) == pytest.approx(0.61)


def test_pick_tau_unreachable_target():
    assert pick_tau([0.6, 0.9], [0, 0], target=0.9) == 1.01


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= 0.75 <= hi


def test_macro_f1_perfect():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
