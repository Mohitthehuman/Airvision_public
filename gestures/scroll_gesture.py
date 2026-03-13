"""
ScrollGesture — two fingers up (index + middle), move hand vertically.

When index and middle are extended and ring/pinky are folded, we track the
vertical position of the midpoint between the two fingertips.

* Moving the hand upward   → scroll up
* Moving the hand downward → scroll down

A deadzone prevents accidental scrolling from small hand tremors.
"""

from __future__ import annotations

from typing import Optional

from core.hand_detector import (
    HandData,
    INDEX_TIP, MIDDLE_TIP,
)
from gestures.base_gesture import BaseGesture, GestureResult


_DEADZONE = 0.018          # minimum normalised movement before scroll fires
_HISTORY_LEN = 5           # frames to average for velocity estimation


class ScrollGesture(BaseGesture):
    """
    Index + middle extended, others folded.

    Payload
    -------
    data['direction']  +1 (up) or -1 (down)
    data['velocity']   magnitude of vertical movement
    """

    def __init__(self) -> None:
        self._prev_y: Optional[float] = None

    @property
    def name(self) -> str:
        return "scroll"

    def check(self, hand: HandData) -> Optional[GestureResult]:
        ext = hand.extended_fingers   # [thumb, index, middle, ring, pinky]

        index_up  = ext[1]
        middle_up = ext[2]
        ring_up   = ext[3]
        pinky_up  = ext[4]

        if not (index_up and middle_up and not ring_up and not pinky_up):
            self._prev_y = None
            return None

        # Midpoint of index and middle tips
        mid_y = (hand.index_tip.y + hand.middle_tip.y) / 2.0

        if self._prev_y is None:
            self._prev_y = mid_y
            return None

        delta = self._prev_y - mid_y   # positive = moved up
        self._prev_y = mid_y

        if abs(delta) < _DEADZONE:
            return None

        direction = 1 if delta > 0 else -1
        return GestureResult(
            name=self.name,
            data={"direction": direction, "velocity": abs(delta)},
        )
