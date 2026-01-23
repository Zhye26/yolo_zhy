"""
Base class for violation rules.
Each rule evaluates a frame context and returns violation candidates.
"""
from abc import ABC, abstractmethod
from typing import List
from app.core.types import FrameContext, ViolationCandidate


class ViolationRule(ABC):
    """Abstract base class for violation detection rules."""

    def __init__(self, rule_id: str, enabled: bool = True, priority: int = 0):
        self.rule_id = rule_id
        self.enabled = enabled
        self.priority = priority

    @abstractmethod
    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        """
        Evaluate the rule against the frame context.

        Args:
            context: Frame context with detections and tracks

        Returns:
            List of violation candidates
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.rule_id}, enabled={self.enabled})"
