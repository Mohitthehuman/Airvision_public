"""
CursorGesture — index finger pointing (index up, all others folded).

The index fingertip position is used as the cursor control point.
"""

from __future__ import annotations

from typing import Optional

from core.hand_detector import (
    HandData,
    INDEX_TIP, INDEX_PIP,
    MIDDLE_TIP, MIDDLE_PIP,
    RING_TIP, RING_PIP,
    PINKY_TIP, PINKY_PIP,
)
from gestures.base_gesture import BaseGesture, GestureResult


class CursorGesture(BaseGesture):
    """
    Fires when only the index finger is clearly extended.

    Payload
    -------
    data['x'], data['y'] — normalised tip position for cursor mapping.
    """

    @property
    def name(self) -> str:
        return "cursor_move"

    def check(self, hand: HandData) -> Optional[GestureResult]:
        ext = hand.extended_fingers   # [thumb, index, middle, ring, pinky]

        index_up  = ext[1]
        middle_up = ext[2]
        ring_up   = ext[3]
        pinky_up  = ext[4]

        if index_up and not middle_up and not ring_up and not pinky_up:
            tip = hand.index_tip
            return GestureResult(
                name=self.name,
                data={"x": tip.x, "y": tip.y},
            )
        return None
