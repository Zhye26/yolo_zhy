"""
Rule engine that orchestrates violation detection rules.
"""
from typing import List, Dict
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
        self.rules.append(PassengerRule(enabled=settings.rules.passenger_rule_enabled))
        self.rules.append(HelmetRule(enabled=settings.rules.helmet_rule_enabled))
        self.rules.sort(key=lambda r: r.priority)

    def add_rule(self, rule: ViolationRule) -> None:
        """Add a custom rule to the engine."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                self.rules.pop(i)
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
            if rule.enabled:
                candidates = rule.evaluate(context)
                all_candidates.extend(candidates)
        return all_candidates

    def get_rules_info(self) -> List[Dict]:
        """Get information about all registered rules."""
        return [
            {
                "rule_id": r.rule_id,
                "enabled": r.enabled,
                "priority": r.priority,
                "class": r.__class__.__name__
            }
            for r in self.rules
        ]
