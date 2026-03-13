"""
Tests for GestureRecognizer — hold-frame stability filter and priority.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock
import pytest

from core.gesture_recognizer import GestureRecognizer
from gestures.base_gesture import BaseGesture, GestureResult
from config import cfg


# ── Stub gesture ──────────────────────────────────────────────────────────

class _StubGesture(BaseGesture):
    """Returns a fixed result when enabled."""
    def __init__(self, name: str, enabled: bool = True):
        self._name = name
        self.enabled = enabled

    @property
    def name(self) -> str:
        return self._name

    def check(self, hand):
        return GestureResult(name=self._name) if self.enabled else None


# ── Tests ─────────────────────────────────────────────────────────────────

def _make_fake_hand():
    return MagicMock()


class TestHoldFrameFilter:
    def test_single_frame_not_enough(self):
        g = _StubGesture("test")
        r = GestureRecognizer([g])
        # hold_frames > 1, so one frame should not fire
        if cfg.detection.gesture_hold_frames > 1:
            result = r.recognize(_make_fake_hand())
            assert result is None

    def test_stable_gesture_fires_after_hold(self):
        g = _StubGesture("stable")
        r = GestureRecognizer([g])
        n = cfg.detection.gesture_hold_frames
        result = None
        for _ in range(n):
            result = r.recognize(_make_fake_hand())
        assert result is not None
        assert result.name == "stable"

    def test_interrupted_gesture_does_not_fire(self):
        g = _StubGesture("flicker")
        r = GestureRecognizer([g])
        n = cfg.detection.gesture_hold_frames

        for i in range(n - 1):
            r.recognize(_make_fake_hand())

        # Interrupt with None gesture
        g.enabled = False
        result = r.recognize(_make_fake_hand())
        assert result is None

        # Re-enable — needs full hold again
        g.enabled = True
        result = r.recognize(_make_fake_hand())
        # Should not fire yet (history broken)
        if n > 1:
            assert result is None


class TestPriority:
    def test_higher_priority_wins(self):
        high = _StubGesture("high_priority")
        low  = _StubGesture("low_priority")
        r = GestureRecognizer([high, low])

        n = cfg.detection.gesture_hold_frames
        result = None
        for _ in range(n):
            result = r.recognize(_make_fake_hand())

        assert result is not None
        assert result.name == "high_priority"

    def test_fallback_when_high_priority_disabled(self):
        high = _StubGesture("high", enabled=False)
        low  = _StubGesture("low",  enabled=True)
        r = GestureRecognizer([high, low])

        n = cfg.detection.gesture_hold_frames
        result = None
        for _ in range(n):
            result = r.recognize(_make_fake_hand())

        assert result is not None
        assert result.name == "low"


class TestRegister:
    def test_register_appends_gesture(self):
        r = GestureRecognizer([])
        g = _StubGesture("new")
        r.register(g)
        n = cfg.detection.gesture_hold_frames
        result = None
        for _ in range(n):
            result = r.recognize(_make_fake_hand())
        assert result is not None

    def test_register_at_index_zero_is_highest_priority(self):
        low  = _StubGesture("low")
        r = GestureRecognizer([low])
        high = _StubGesture("high")
        r.register(high, index=0)

        n = cfg.detection.gesture_hold_frames
        result = None
        for _ in range(n):
            result = r.recognize(_make_fake_hand())

        assert result is not None
        assert result.name == "high"
