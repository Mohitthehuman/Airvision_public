"""
BaseGesture — abstract interface every gesture must implement.

A GestureResult carries:
  name   — string identifier used in UI labels and routing
  data   — arbitrary payload (e.g. scroll direction, click type)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from core.hand_detector import HandData


@dataclass
class GestureResult:
    name: str
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GestureResult(name={self.name!r}, data={self.data})"


class BaseGesture(ABC):
    """Every gesture must implement check()."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this gesture."""

    @abstractmethod
    def check(self, hand: HandData) -> Optional[GestureResult]:
        """
        Inspect *hand* and return a GestureResult if this gesture is
        detected, or None otherwise.
        """
