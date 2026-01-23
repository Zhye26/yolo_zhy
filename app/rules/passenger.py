"""
Passenger violation rule.
Detects when an e-bike is carrying a passenger.
"""
from typing import List
from app.core.types import (
    FrameContext,
    ViolationCandidate,
    ViolationType,
    Detection,
    Track,
)
from app.rules.base import ViolationRule
from app.config import settings


class PassengerRule(ViolationRule):
    """Rule for detecting passenger violations on e-bikes."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="passenger_violation",
            enabled=enabled,
            priority=1
        )
        self.passenger_class_id = settings.detection.passenger_class_id

    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        if not self.enabled:
            return []

        candidates = []
        sources = context.tracks if context.tracks else [
            Track(track_id=-1, detection=d) for i, d in enumerate(context.detections)
        ]

        for item in sources:
            det = item.detection if isinstance(item, Track) else item
            if det.class_id == self.passenger_class_id:
                track_id = item.track_id if isinstance(item, Track) else None
                candidates.append(ViolationCandidate(
                    rule_id=self.rule_id,
                    violation_type=ViolationType.PASSENGER,
                    entity_ids=[track_id] if track_id else [],
                    bbox=det.bbox,
                    confidence=det.confidence,
                    evidence={"class": "passenger", "track_id": track_id},
                    description="电动车载人违规"
                ))

        return candidates
