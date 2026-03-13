"""
PauseGesture — open palm (all five fingers extended).

When active, cursor movement and clicks are suspended.  This gives the user
a "safe" state to reposition their hand without accidentally moving the cursor.
"""

from __future__ import annotations

from typing import Optional

from core.hand_detector import HandData
from gestures.base_gesture import BaseGesture, GestureResult
from config import cfg


class PauseGesture(BaseGesture):
    """All fingers open → pause all cursor interaction."""

    @property
    def name(self) -> str:
        return "pause"

    def check(self, hand: HandData) -> Optional[GestureResult]:
        if hand.finger_count >= cfg.gesture.pause_finger_count:
            return GestureResult(name=self.name)
        return None
