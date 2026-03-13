"""Tests for ExponentialSmoother."""

import math
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.smoother import ExponentialSmoother


def test_first_call_returns_exact_input():
    s = ExponentialSmoother(alpha=0.5)
    x, y = s.smooth(100.0, 200.0)
    assert x == 100.0
    assert y == 200.0


def test_smoothing_converges_toward_target():
    s = ExponentialSmoother(alpha=0.5)
    s.smooth(0.0, 0.0)   # seed
    # feed 100, 100 repeatedly — should converge
    for _ in range(20):
        x, y = s.smooth(100.0, 100.0)
    assert abs(x - 100.0) < 0.01
    assert abs(y - 100.0) < 0.01


def test_reset_clears_state():
    s = ExponentialSmoother(alpha=0.5)
    s.smooth(500.0, 500.0)
    s.reset()
    x, y = s.smooth(10.0, 20.0)
    assert x == 10.0 and y == 20.0   # first call after reset returns raw


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=1.5)


def test_alpha_setter_validation():
    s = ExponentialSmoother(alpha=0.3)
    with pytest.raises(ValueError):
        s.alpha = -0.1


def test_smoothed_output_is_between_seed_and_target():
    s = ExponentialSmoother(alpha=0.4)
    s.smooth(0.0, 0.0)
    x, y = s.smooth(100.0, 100.0)
    assert 0.0 < x < 100.0
    assert 0.0 < y < 100.0
