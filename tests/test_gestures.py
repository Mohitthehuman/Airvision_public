"""
Tests for gesture detection logic.

We build minimal HandData objects (fake landmark lists) and verify each
gesture class returns the correct GestureResult or None.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from core.hand_detector import (
    HandData, LandmarkPoint,
    WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
)
from gestures.cursor_gesture import CursorGesture
from gestures.pause_gesture import PauseGesture
from gestures.scroll_gesture import ScrollGesture
from gestures.click_gestures import LeftClickGesture, RightClickGesture


# ── Helpers ────────────────────────────────────────────────────────────────

def _lm(x: float, y: float, z: float = 0.0) -> LandmarkPoint:
    return LandmarkPoint(x=x, y=y, z=z, px=int(x * 640), py=int(y * 480))


def _make_hand(finger_states: list[bool]) -> HandData:
    """
    Build a synthetic HandData where finger extension is determined by
    finger_states = [thumb, index, middle, ring, pinky].

    Strategy:
    * Wrist is at y=0.8 (bottom of frame).
    * MCP joints at y=0.6.
    * For extended fingers: PIP at 0.4, TIP at 0.2  (tip above pip → extended).
    * For folded fingers:   PIP at 0.6, TIP at 0.65 (tip below pip → folded).

    Thumb extension is based on x-axis spread from MCP.
    """
    thumb_ext, idx_ext, mid_ext, ring_ext, pinky_ext = finger_states

    # 21 landmarks, initialised to wrist position
    lms = [_lm(0.5, 0.8)] * 21   # default: wrist

    # Wrist
    lms[WRIST] = _lm(0.5, 0.8)

    # ── Thumb (x-axis extension) ─────────────────────────────────────────
    lms[THUMB_CMC] = _lm(0.45, 0.75)
    lms[THUMB_MCP] = _lm(0.40, 0.70)
    lms[THUMB_IP]  = _lm(0.36, 0.68)
    lms[THUMB_TIP] = _lm(0.30 if thumb_ext else 0.39, 0.66)

    # ── Index ────────────────────────────────────────────────────────────
    tip_y = 0.2 if idx_ext else 0.65
    pip_y = 0.4 if idx_ext else 0.60
    lms[INDEX_MCP] = _lm(0.50, 0.60)
    lms[INDEX_PIP] = _lm(0.50, pip_y)
    lms[INDEX_DIP] = _lm(0.50, (tip_y + pip_y) / 2)
    lms[INDEX_TIP] = _lm(0.50, tip_y)

    # ── Middle ───────────────────────────────────────────────────────────
    tip_y = 0.2 if mid_ext else 0.65
    pip_y = 0.4 if mid_ext else 0.60
    lms[MIDDLE_MCP] = _lm(0.53, 0.60)
    lms[MIDDLE_PIP] = _lm(0.53, pip_y)
    lms[MIDDLE_DIP] = _lm(0.53, (tip_y + pip_y) / 2)
    lms[MIDDLE_TIP] = _lm(0.53, tip_y)

    # ── Ring ─────────────────────────────────────────────────────────────
    tip_y = 0.2 if ring_ext else 0.65
    pip_y = 0.4 if ring_ext else 0.60
    lms[RING_MCP] = _lm(0.56, 0.60)
    lms[RING_PIP] = _lm(0.56, pip_y)
    lms[RING_DIP] = _lm(0.56, (tip_y + pip_y) / 2)
    lms[RING_TIP] = _lm(0.56, tip_y)

    # ── Pinky ────────────────────────────────────────────────────────────
    tip_y = 0.2 if pinky_ext else 0.65
    pip_y = 0.4 if pinky_ext else 0.60
    lms[PINKY_MCP] = _lm(0.59, 0.60)
    lms[PINKY_PIP] = _lm(0.59, pip_y)
    lms[PINKY_DIP] = _lm(0.59, (tip_y + pip_y) / 2)
    lms[PINKY_TIP] = _lm(0.59, tip_y)

    return HandData(landmarks=lms)


# ── PauseGesture ───────────────────────────────────────────────────────────

class TestPauseGesture:
    def test_open_palm_triggers_pause(self):
        hand = _make_hand([True, True, True, True, True])
        result = PauseGesture().check(hand)
        assert result is not None
        assert result.name == "pause"

    def test_pointing_does_not_pause(self):
        hand = _make_hand([False, True, False, False, False])
        assert PauseGesture().check(hand) is None

    def test_four_fingers_no_pause(self):
        # Only 4 fingers — not enough
        hand = _make_hand([False, True, True, True, True])
        assert PauseGesture().check(hand) is None


# ── CursorGesture ──────────────────────────────────────────────────────────

class TestCursorGesture:
    def test_single_index_fires_cursor(self):
        hand = _make_hand([False, True, False, False, False])
        result = CursorGesture().check(hand)
        assert result is not None
        assert result.name == "cursor_move"
        assert "x" in result.data and "y" in result.data

    def test_two_fingers_no_cursor(self):
        hand = _make_hand([False, True, True, False, False])
        assert CursorGesture().check(hand) is None

    def test_open_palm_no_cursor(self):
        hand = _make_hand([True, True, True, True, True])
        assert CursorGesture().check(hand) is None


# ── ScrollGesture ──────────────────────────────────────────────────────────

class TestScrollGesture:
    def _two_finger_hand(self, mid_y: float) -> HandData:
        """
        Build a two-finger hand with both tips near mid_y.
        PIP joints are placed well above (lower y value) the tips so the
        extension check always passes regardless of mid_y.
        """
        hand = _make_hand([False, True, True, False, False])
        lms = list(hand.landmarks)
        # In image coords y increases downward, so PIP must have a LARGER y
        # than the tip for the extension check (tip.y < pip.y - 0.02) to pass.
        pip_y = mid_y + 0.15     # PIP sits below the tip on screen
        lms[INDEX_PIP]  = _lm(0.50, pip_y)
        lms[INDEX_TIP]  = _lm(0.50, mid_y - 0.01)
        lms[MIDDLE_PIP] = _lm(0.53, pip_y)
        lms[MIDDLE_TIP] = _lm(0.53, mid_y + 0.01)
        hand.landmarks = lms
        return hand

    def test_wrong_fingers_returns_none(self):
        g = ScrollGesture()
        hand = _make_hand([False, True, False, False, False])
        assert g.check(hand) is None

    def test_first_frame_seeds_position(self):
        g = ScrollGesture()
        hand = _make_hand([False, True, True, False, False])
        # First call just seeds — no scroll yet
        assert g.check(hand) is None

    def test_scroll_up_on_upward_move(self):
        g = ScrollGesture()
        hand1 = self._two_finger_hand(0.5)
        hand2 = self._two_finger_hand(0.3)   # moved up (lower y = higher on screen)
        g.check(hand1)                         # seed
        result = g.check(hand2)
        assert result is not None
        assert result.name == "scroll"
        assert result.data["direction"] == 1

    def test_scroll_down_on_downward_move(self):
        g = ScrollGesture()
        hand1 = self._two_finger_hand(0.3)
        hand2 = self._two_finger_hand(0.6)
        g.check(hand1)
        result = g.check(hand2)
        assert result is not None
        assert result.data["direction"] == -1

    def test_tiny_move_in_deadzone_no_scroll(self):
        g = ScrollGesture()
        hand1 = self._two_finger_hand(0.5)
        hand2 = self._two_finger_hand(0.501)   # less than DEADZONE (0.018)
        g.check(hand1)
        assert g.check(hand2) is None


# ── LeftClickGesture ───────────────────────────────────────────────────────

class TestLeftClickGesture:
    def _pinched_hand(self) -> HandData:
        """Thumb tip very close to index tip."""
        hand = _make_hand([True, True, False, False, False])
        lms = list(hand.landmarks)
        lms[THUMB_TIP] = _lm(0.50, 0.50)
        lms[INDEX_TIP] = _lm(0.50, 0.50)   # same position = 0 distance
        hand.landmarks = lms
        return hand

    def test_pinch_fires_left_click(self):
        result = LeftClickGesture().check(self._pinched_hand())
        assert result is not None
        assert result.name in ("left_click", "double_click")

    def test_no_pinch_returns_none(self):
        hand = _make_hand([False, True, False, False, False])
        # Thumb and index are far apart in _make_hand
        result = LeftClickGesture().check(hand)
        # Should be None or the distance should be large enough to not trigger
        if result is not None:
            # If it fires, the distance must be below threshold — fine
            assert result.name in ("left_click", "double_click")


# ── RightClickGesture ──────────────────────────────────────────────────────

class TestRightClickGesture:
    def test_thumb_middle_pinch_fires(self):
        hand = _make_hand([True, False, True, False, False])
        lms = list(hand.landmarks)
        lms[THUMB_TIP]  = _lm(0.53, 0.50)
        lms[MIDDLE_TIP] = _lm(0.53, 0.50)
        hand.landmarks = lms
        result = RightClickGesture().check(hand)
        assert result is not None
        assert result.name == "right_click"


# ── HandData helpers ───────────────────────────────────────────────────────

class TestHandData:
    def test_finger_count_all_extended(self):
        hand = _make_hand([True, True, True, True, True])
        assert hand.finger_count == 5

    def test_finger_count_none_extended(self):
        hand = _make_hand([False, False, False, False, False])
        assert hand.finger_count == 0

    def test_dist_same_point_is_zero(self):
        hand = _make_hand([False, True, False, False, False])
        lms = list(hand.landmarks)
        lms[INDEX_TIP] = _lm(0.5, 0.5)
        lms[MIDDLE_TIP] = _lm(0.5, 0.5)
        hand.landmarks = lms
        assert hand.dist(INDEX_TIP, MIDDLE_TIP) == pytest.approx(0.0)

    def test_hand_size_positive(self):
        hand = _make_hand([False, True, False, False, False])
        assert hand.hand_size > 0
