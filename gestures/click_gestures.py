"""
Click gestures — pinch-based left click, right click, and double click.

Gesture definitions
-------------------
Left click   : Thumb tip touches Index tip  (pinch)
Right click  : Thumb tip touches Middle tip (pinch)
Double click : Two rapid left-click pinches within double_click_interval_sec

All distances are normalised by hand size so the threshold is
resolution-independent.
"""

from __future__ import annotations

import time
from typing import Optional

from config import cfg
from core.hand_detector import (
    HandData,
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP,
)
from gestures.base_gesture import BaseGesture, GestureResult


class LeftClickGesture(BaseGesture):
    """
    Thumb-index pinch → left click.

    Tracks rapid pinch repeats so a double-click can be detected without
    a separate gesture class.
    """

    def __init__(self) -> None:
        self._last_pinch_time: float = 0.0
        self._pending_double: bool = False

    @property
    def name(self) -> str:
        return "left_click"

    def check(self, hand: HandData) -> Optional[GestureResult]:
        dist = hand.normalised_dist(THUMB_TIP, INDEX_TIP)
        threshold = cfg.gesture.click_pinch_threshold

        if dist < threshold:
            now = time.time()
            interval = now - self._last_pinch_time
            self._last_pinch_time = now

            if interval < cfg.gesture.double_click_interval_sec:
                return GestureResult(name="double_click")
            return GestureResult(
                name=self.name,
                data={"dist": dist},
            )
        return None


class RightClickGesture(BaseGesture):
    """Thumb-middle pinch → right click."""

    @property
    def name(self) -> str:
        return "right_click"

    def check(self, hand: HandData) -> Optional[GestureResult]:
        dist = hand.normalised_dist(THUMB_TIP, MIDDLE_TIP)
        if dist < cfg.gesture.right_click_pinch_threshold:
            return GestureResult(name=self.name, data={"dist": dist})
        return None
