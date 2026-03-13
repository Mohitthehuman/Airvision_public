"""
Exponential Moving Average (EMA) smoother for cursor coordinates.

Formula:  smoothed = alpha * raw + (1 - alpha) * previous_smoothed

* alpha close to 1.0  → very responsive, but more jitter
* alpha close to 0.0  → very smooth, but sluggish / laggy
"""

from __future__ import annotations

from typing import Optional, Tuple


class ExponentialSmoother:
    """
    Two-dimensional EMA cursor smoother.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  Recommended range: 0.15 – 0.40.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        if not 0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        """Apply EMA and return the smoothed (x, y)."""
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            self._sx = self._alpha * x + (1 - self._alpha) * self._sx
            self._sy = self._alpha * y + (1 - self._alpha) * self._sy
        return self._sx, self._sy

    def reset(self) -> None:
        """Forget state (call when cursor was paused)."""
        self._sx = None
        self._sy = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0 < value <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {value}")
        self._alpha = value
