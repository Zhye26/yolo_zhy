"""
Helmet violation rule.
Detects when a driver or passenger is not wearing a helmet.
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


class HelmetRule(ViolationRule):
    """Rule for detecting no-helmet violations."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="no_helmet_violation",
            enabled=enabled,
            priority=2
        )
        self.driver_class_id = settings.detection.driver_class_id
        self.passenger_class_id = settings.detection.passenger_class_id
        self.helmet_class_id = settings.detection.helmet_class_id
        self.head_ratio = settings.rules.helmet_head_ratio
        self.overlap_threshold = settings.rules.helmet_overlap_threshold

    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        if not self.enabled:
            return []

        candidates = []
        helmets = []
        persons = []

        sources = context.tracks if context.tracks else [
            Track(track_id=i, detection=d) for i, d in enumerate(context.detections)
        ]

        for item in sources:
            det = item.detection if isinstance(item, Track) else item
            if det.class_id == self.helmet_class_id:
                helmets.append(det)
            elif det.class_id in (self.driver_class_id, self.passenger_class_id):
                persons.append((item, det))

        for item, person_det in persons:
            has_helmet = self._check_helmet_on_person(person_det, helmets)
            if not has_helmet:
                track_id = item.track_id if isinstance(item, Track) else None
                candidates.append(ViolationCandidate(
                    rule_id=self.rule_id,
                    violation_type=ViolationType.NO_HELMET,
                    entity_ids=[track_id] if track_id else [],
                    bbox=person_det.bbox,
                    confidence=person_det.confidence,
                    evidence={
                        "class": "driver" if person_det.class_id == self.driver_class_id else "passenger",
                        "track_id": track_id
                    },
                    description="未佩戴头盔"
                ))

        return candidates

    def _check_helmet_on_person(self, person: Detection, helmets: List[Detection]) -> bool:
        """Check if any helmet is on the person's head region."""
        px1, py1, px2, py2 = person.bbox
        head_y2 = py1 + (py2 - py1) * self.head_ratio

        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet.bbox
            if hy2 > head_y2:
                continue
            overlap_x = max(0, min(hx2, px2) - max(hx1, px1))
            helmet_width = hx2 - hx1
            if helmet_width > 0 and overlap_x / helmet_width > self.overlap_threshold:
                return True

        return False
