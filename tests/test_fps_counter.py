"""Tests for FPSCounter."""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.fps_counter import FPSCounter


def test_fps_zero_before_two_ticks():
    c = FPSCounter()
    c.tick()
    assert c.fps == 0.0


def test_fps_positive_after_ticks():
    c = FPSCounter(window=5)
    for _ in range(5):
        c.tick()
        time.sleep(0.01)
    assert c.fps > 0


def test_fps_reasonable_range():
    c = FPSCounter(window=10)
    for _ in range(10):
        c.tick()
        time.sleep(0.005)   # ~200 fps theoretical max
    assert 10 < c.fps < 500
