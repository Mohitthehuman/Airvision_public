"""
FPSCounter — rolling-window FPS estimator.

Keeps the last N frame timestamps and computes the average FPS from them.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Rolling-window frames-per-second counter."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a new frame and return the current FPS."""
        self._timestamps.append(time.monotonic())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
