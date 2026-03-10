"""
Overload violation rule.
Counts how many riders are associated with a single e-bike and flags overload.
"""
from typing import Dict, List, Optional, Tuple
from app.core.types import (
    Detection,
    FrameContext,
    Track,
    ViolationCandidate,
    ViolationType,
)
from app.rules.base import ViolationRule
from app.config import settings


class PassengerRule(ViolationRule):
    """Rule for detecting overloaded e-bikes by rider count."""

    def __init__(self, enabled: bool = True):
        super().__init__(
            rule_id="passenger_violation",
            enabled=enabled,
            priority=1,
        )
        self.ebike_class_id = settings.detection.ebike_class_id
        self.driver_class_id = settings.detection.driver_class_id
        self.passenger_class_id = settings.detection.passenger_class_id

    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        if not self.enabled:
            return []

        tracked_sources = [
            item for item in context.tracks
            if getattr(item, "state", "tracked") == "tracked"
        ]
        sources = tracked_sources if tracked_sources else [
            Track(track_id=index, detection=detection)
            for index, detection in enumerate(context.detections)
        ]

        ebikes = self._canonicalize_ebikes(
            [item for item in sources if item.detection.class_id == self.ebike_class_id]
        )
        riders = [
            item for item in sources
            if item.detection.class_id in {self.driver_class_id, self.passenger_class_id}
        ]
        if not ebikes or not riders:
            return []

        grouped_riders: Dict[int, Dict[str, object]] = {}
        for ebike_index, ebike_track in enumerate(ebikes):
            grouped_riders[ebike_index] = {
                "ebike": ebike_track,
                "riders": [],
            }

        for rider in riders:
            match = self._find_best_ebike_match(rider.detection, ebikes)
            if match is None:
                continue
            ebike_index, _ = match
            grouped_riders[ebike_index]["riders"].append(rider)

        candidates: List[ViolationCandidate] = []
        for group in grouped_riders.values():
            ebike_track = group["ebike"]
            matched_riders: List[Track] = group["riders"]
            if not matched_riders:
                continue

            overload_count = len(matched_riders)
            merged_overload = False
            if overload_count < 2:
                merged_overload = self._looks_like_merged_double_rider(
                    rider=matched_riders[0].detection,
                    ebike=ebike_track.detection,
                )
                if not merged_overload:
                    continue
                overload_count = 2

            violation_track_id = self._resolve_violation_track_id(ebike_track, matched_riders)
            violation_bbox = self._merge_boxes(
                [ebike_track.detection.bbox] + [rider.detection.bbox for rider in matched_riders]
            )
            confidence = max([
                ebike_track.detection.confidence,
                *[rider.detection.confidence for rider in matched_riders],
            ])

            candidates.append(ViolationCandidate(
                rule_id=self.rule_id,
                violation_type=ViolationType.PASSENGER,
                entity_ids=[violation_track_id] if violation_track_id is not None else [],
                bbox=violation_bbox,
                confidence=confidence,
                evidence={
                    "class": "overload_merged" if merged_overload else "overload_count",
                    "rider_count": overload_count,
                    "track_id": violation_track_id,
                },
                description="电动车超载",
            ))

        return candidates

    def _canonicalize_ebikes(self, ebikes: List[Track]) -> List[Track]:
        kept: List[Track] = []
        ordered = sorted(
            ebikes,
            key=lambda item: (item.detection.area, item.detection.confidence),
            reverse=True,
        )
        for candidate in ordered:
            if any(self._same_vehicle(candidate.detection.bbox, existing.detection.bbox) for existing in kept):
                continue
            kept.append(candidate)
        return kept

    def _same_vehicle(self, candidate_bbox: List[float], existing_bbox: List[float]) -> bool:
        overlap = self._overlap_ratio(candidate_bbox, existing_bbox)
        area_ratio = min(self._area(candidate_bbox), self._area(existing_bbox)) / max(self._area(candidate_bbox), self._area(existing_bbox), 1.0)
        if overlap > 0.62:
            return True
        if area_ratio < 0.45 and (self._contains(existing_bbox, candidate_bbox, margin=0.08) or self._contains(candidate_bbox, existing_bbox, margin=0.08)):
            return True
        return False

    def _find_best_ebike_match(
        self,
        rider: Detection,
        ebikes: List[Track],
    ) -> Optional[Tuple[int, float]]:
        best_index: Optional[int] = None
        best_score = 0.0
        best_area = 0.0

        for ebike_index, ebike_track in enumerate(ebikes):
            score = self._rider_ebike_score(rider.bbox, ebike_track.detection.bbox)
            ebike_area = ebike_track.detection.area
            if (
                score > best_score + 0.08
                or (abs(score - best_score) <= 0.12 and ebike_area > best_area * 1.2)
            ):
                best_score = score
                best_area = ebike_area
                best_index = ebike_index

        if best_index is None or best_score < 0.16:
            return None
        return best_index, best_score

    def _looks_like_merged_double_rider(
        self,
        rider: Detection,
        ebike: Detection,
    ) -> bool:
        rider_w = max(1.0, rider.bbox[2] - rider.bbox[0])
        rider_h = max(1.0, rider.bbox[3] - rider.bbox[1])
        ebike_w = max(1.0, ebike.bbox[2] - ebike.bbox[0])
        ebike_h = max(1.0, ebike.bbox[3] - ebike.bbox[1])
        overlap = self._overlap_ratio(rider.bbox, ebike.bbox)
        width_ratio = rider_w / ebike_w
        height_ratio = rider_h / ebike_h
        foot_in_box = self._foot_point_inside(rider.bbox, ebike.bbox, margin_x=0.18, margin_top=0.22, margin_bottom=0.18)

        return (
            ebike_w >= 140.0
            and ebike_h >= 140.0
            and rider_w >= 120.0
            and rider_h >= 240.0
            and width_ratio >= 0.68
            and height_ratio >= 1.45
            and overlap >= 0.55
            and foot_in_box
        )

    def _resolve_violation_track_id(
        self,
        ebike_track: Track,
        riders: List[Track],
    ) -> Optional[int]:
        if ebike_track.track_id >= 0:
            return ebike_track.track_id
        for rider in riders:
            if rider.track_id >= 0:
                return rider.track_id
        return None

    def _rider_ebike_score(self, rider_bbox: List[float], ebike_bbox: List[float]) -> float:
        px1, py1, px2, py2 = rider_bbox
        ex1, ey1, ex2, ey2 = ebike_bbox
        rider_w = max(1.0, px2 - px1)
        rider_h = max(1.0, py2 - py1)
        ebike_w = max(1.0, ex2 - ex1)
        ebike_h = max(1.0, ey2 - ey1)

        foot_x = (px1 + px2) / 2
        foot_y = py2
        margin_x = ebike_w * 0.18
        margin_top = ebike_h * 0.22
        margin_bottom = ebike_h * 0.16

        score = 0.0
        if (
            ex1 - margin_x <= foot_x <= ex2 + margin_x
            and ey1 - margin_top <= foot_y <= ey2 + margin_bottom
        ):
            score += 0.75

        lower_half = [px1, py1 + (py2 - py1) * 0.45, px2, py2]
        score += self._iou(lower_half, ebike_bbox)
        score += self._horizontal_overlap(lower_half, ebike_bbox) * 0.35
        score += min(ebike_w / rider_w, 6.0) * 0.05
        score += min(ebike_h / rider_h, 2.5) * 0.10
        return score

    def _foot_point_inside(
        self,
        rider_bbox: List[float],
        ebike_bbox: List[float],
        margin_x: float,
        margin_top: float,
        margin_bottom: float,
    ) -> bool:
        px1, _, px2, py2 = rider_bbox
        ex1, ey1, ex2, ey2 = ebike_bbox
        foot_x = (px1 + px2) / 2
        foot_y = py2
        ebike_w = ex2 - ex1
        ebike_h = ey2 - ey1
        return (
            ex1 - ebike_w * margin_x <= foot_x <= ex2 + ebike_w * margin_x
            and ey1 - ebike_h * margin_top <= foot_y <= ey2 + ebike_h * margin_bottom
        )

    def _horizontal_overlap(self, box_a: List[float], box_b: List[float]) -> float:
        overlap = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        width = max(1.0, min(box_a[2] - box_a[0], box_b[2] - box_b[0]))
        return overlap / width

    def _merge_boxes(self, boxes: List[List[float]]) -> List[float]:
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    def _contains(self, outer_bbox: List[float], inner_bbox: List[float], margin: float = 0.0) -> bool:
        ox1, oy1, ox2, oy2 = outer_bbox
        ix1, iy1, ix2, iy2 = inner_bbox
        outer_w = ox2 - ox1
        outer_h = oy2 - oy1
        return (
            ox1 - outer_w * margin <= ix1
            and oy1 - outer_h * margin <= iy1
            and ox2 + outer_w * margin >= ix2
            and oy2 + outer_h * margin >= iy2
        )

    def _area(self, bbox: List[float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _intersection_area(self, box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    def _overlap_ratio(self, box_a: List[float], box_b: List[float]) -> float:
        inter_area = self._intersection_area(box_a, box_b)
        if inter_area <= 0:
            return 0.0
        return inter_area / max(1.0, min(self._area(box_a), self._area(box_b)))

    def _iou(self, box_a: List[float], box_b: List[float]) -> float:
        inter_area = self._intersection_area(box_a, box_b)
        if inter_area <= 0:
            return 0.0
        area_a = self._area(box_a)
        area_b = self._area(box_b)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0
