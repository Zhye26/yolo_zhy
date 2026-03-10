"""
Rule engine that orchestrates violation detection rules.
"""
from typing import Dict, List
from app.core.types import FrameContext, ViolationCandidate
from app.rules.base import ViolationRule
from app.rules.passenger import PassengerRule
from app.rules.helmet import HelmetRule
from app.config import settings


class RuleEngine:
    """
    Orchestrates violation detection rules.
    Evaluates all enabled rules and returns violation candidates.
    """

    def __init__(self):
        self.rules: List[ViolationRule] = []
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Initialize default rules based on settings."""
        if settings.rules.passenger_rule_enabled:
            self.rules.append(PassengerRule(enabled=True))
        if settings.rules.helmet_rule_enabled and settings.detection.helmet_detection_enabled:
            self.rules.append(HelmetRule(enabled=True))
        self.rules.sort(key=lambda rule: rule.priority)

    def add_rule(self, rule: ViolationRule) -> None:
        """Add a custom rule to the engine."""
        self.rules.append(rule)
        self.rules.sort(key=lambda item: item.priority)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for index, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                self.rules.pop(index)
                return True
        return False

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a rule by ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = enabled
                return True
        return False

    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        """
        Evaluate all enabled rules against the frame context.

        Args:
            context: Frame context with detections and tracks

        Returns:
            List of all violation candidates from all rules
        """
        all_candidates = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            all_candidates.extend(rule.evaluate(context))
        return all_candidates

    def get_rules_info(self) -> List[Dict]:
        """Get information about all registered rules."""
        return [
            {
                "rule_id": rule.rule_id,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "class": rule.__class__.__name__,
            }
            for rule in self.rules
        ]
