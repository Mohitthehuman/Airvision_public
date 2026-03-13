"""
GestureRecognizer — interprets HandData into high-level gesture names.

Design
------
Each concrete gesture class (in gestures/) implements BaseGesture.check().
The recognizer tries them in priority order and returns the first match.

A gesture must be stable for `detection.gesture_hold_frames` consecutive
frames before it is declared active (prevents flicker).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

from config import cfg
from core.hand_detector import HandData
from gestures.base_gesture import BaseGesture, GestureResult

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """
    Manages a prioritised list of gesture detectors and handles the
    hold-frames stability filter.
    """

    def __init__(self, gestures: List[BaseGesture]) -> None:
        self._gestures = gestures
        self._hold_len = cfg.detection.gesture_hold_frames
        self._history: deque[str] = deque(maxlen=self._hold_len)

    def recognize(self, hand: HandData) -> Optional[GestureResult]:
        """
        Run all registered gestures against *hand* and return the result
        for whichever is stable, or None.
        """
        # Find the first matching gesture
        raw_result: Optional[GestureResult] = None
        for gesture in self._gestures:
            result = gesture.check(hand)
            if result is not None:
                raw_result = result
                break   # gestures are priority-ordered

        label = raw_result.name if raw_result else "none"
        self._history.append(label)

        # Require hold_len identical labels in a row
        if len(self._history) == self._hold_len and len(set(self._history)) == 1:
            return raw_result

        return None

    def register(self, gesture: BaseGesture, index: int = -1) -> None:
        """Add a gesture at the given priority index (-1 = lowest priority)."""
        if index == -1:
            self._gestures.append(gesture)
        else:
            self._gestures.insert(index, gesture)
