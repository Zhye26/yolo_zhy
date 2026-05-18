"""
Overload violation rule.
Counts how many riders are associated with a single e-bike and flags overload.
"""
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.core.types import (
    Detection,
    FrameContext,
    Track,
    ViolationCandidate,
    ViolationType,
)
from app.rules.base import ViolationRule
from app.rules.match_classifier import PersonVehicleMatchClassifier
from app.rules.match_features import legacy_rider_vehicle_score


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
        self._track_history: Dict[int, Dict[str, int]] = {}
        self._activation_frames = 2
        self._temporal_hold_frames = 2
        self._edge_temporal_hold_frames = 24
        self._weak_support_hold_frames = 1
        self._edge_weak_support_hold_frames = 16
        self._match_classifier = self._build_match_classifier()

    def reset(self) -> None:
        self._track_history.clear()

    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        if not self.enabled:
            return []

        tracked_sources = [
            item for item in context.tracks
            if getattr(item, "state", "tracked") == "tracked" and item.track_id >= 0
        ]
        ebikes = self._canonicalize_ebikes(
            [item for item in tracked_sources if item.detection.class_id == self.ebike_class_id]
        )
        if not ebikes:
            self._prune_track_history(set())
            return []

        riders = self._build_rider_sources(context)

        grouped_riders: Dict[int, Dict[str, object]] = {}
        for ebike_index, ebike_track in enumerate(ebikes):
            grouped_riders[ebike_index] = {
                "ebike": ebike_track,
                "riders": [],
            }

        for rider in riders:
            match = self._find_best_ebike_match(rider.detection, ebikes, context.meta.width)
            if match is None:
                continue
            ebike_index, _ = match
            grouped_riders[ebike_index]["riders"].append(rider)

        candidates: List[ViolationCandidate] = []
        active_track_ids = set()
        for group in grouped_riders.values():
            ebike_track = group["ebike"]
            edge_partial = self._is_edge_partial_ebike(ebike_track.detection.bbox, context.meta.width)
            matched_riders: List[Track] = self._filter_matched_riders(
                self._deduplicate_riders(group["riders"]),
                ebike_track.detection.bbox,
                edge_partial=edge_partial,
            )
            active_track_ids.add(ebike_track.track_id)

            if ebike_track.hits < max(2, settings.violations.min_frames_to_confirm):
                self._update_track_history(
                    ebike_track.track_id,
                    strong_overload=False,
                    weak_support=False,
                    rider_count=0,
                    edge_partial=edge_partial,
                )
                continue

            overload_count = len(matched_riders)
            merged_overload = False
            merged_overload_class: Optional[str] = None
            strong_overload = overload_count >= 2
            if not strong_overload and overload_count == 1:
                merged_overload_class = self._merged_overload_class(
                    rider=matched_riders[0].detection,
                    ebike=ebike_track.detection,
                    edge_partial=edge_partial,
                )
                merged_overload = merged_overload_class is not None
                if merged_overload:
                    overload_count = 2
                    strong_overload = True

            weak_support = overload_count >= 1
            is_active, using_strong_evidence = self._update_track_history(
                ebike_track.track_id,
                strong_overload=strong_overload,
                weak_support=weak_support,
                rider_count=overload_count,
                edge_partial=edge_partial,
                immediate_confirm=bool(
                    merged_overload_class == "overload_merged_side_view" and ebike_track.hits >= 5
                ),
            )
            if not is_active:
                continue
            if not using_strong_evidence and not edge_partial:
                continue
            if not using_strong_evidence and not edge_partial and ebike_track.hits < 20:
                continue

            if edge_partial:
                tracked_rider_count = sum(1 for rider in matched_riders if rider.track_id >= 0)
                if tracked_rider_count < 2 and not using_strong_evidence and not self._has_edge_partial_proxy_support(
                    matched_riders,
                    ebike_track.detection.bbox,
                ):
                    continue

            violation_track_id = self._resolve_violation_track_id(ebike_track, matched_riders)
            if violation_track_id is None:
                continue

            confidence = max([
                ebike_track.detection.confidence,
                *[rider.detection.confidence for rider in matched_riders],
            ])
            candidates.append(
                ViolationCandidate(
                    rule_id=self.rule_id,
                    violation_type=ViolationType.PASSENGER,
                    entity_ids=[violation_track_id],
                    bbox=list(ebike_track.detection.bbox),
                    confidence=confidence,
                    evidence={
                        "class": (
                            merged_overload_class
                            if merged_overload_class is not None
                            else "overload_count" if using_strong_evidence else "overload_temporal_hold"
                        ),
                        "rider_count": overload_count,
                        "track_id": violation_track_id,
                    },
                    description="电动车超载",
                )
            )

        self._prune_track_history(active_track_ids)
        return candidates

    def _build_rider_sources(self, context: FrameContext) -> List[Track]:
        rider_class_ids = {self.driver_class_id, self.passenger_class_id}
        tracked_riders = [
            item for item in context.tracks
            if getattr(item, "state", "tracked") == "tracked"
            and item.track_id >= 0
            and item.detection.class_id in rider_class_ids
        ]
        rider_sources: List[Track] = []

        for detection in context.detections:
            if detection.class_id not in rider_class_ids:
                continue
            rider_sources.append(
                Track(
                    track_id=self._match_track_id(detection, tracked_riders),
                    detection=detection,
                )
            )

        return rider_sources

    def _update_track_history(
        self,
        track_id: int,
        strong_overload: bool,
        weak_support: bool,
        rider_count: int,
        edge_partial: bool,
        immediate_confirm: bool = False,
    ) -> Tuple[bool, bool]:
        entry = self._track_history.setdefault(
            track_id,
            {
                "positive_streak": 0,
                "hold_frames": 0,
                "recent_max": 0,
                "confirmed": 0,
            },
        )

        if strong_overload:
            entry["positive_streak"] += 1
            entry["hold_frames"] = self._edge_temporal_hold_frames if edge_partial else self._temporal_hold_frames
            entry["recent_max"] = max(entry["recent_max"], rider_count)
            if immediate_confirm:
                entry["positive_streak"] = max(entry["positive_streak"], self._activation_frames)
        elif weak_support:
            if entry["confirmed"]:
                if edge_partial:
                    entry["hold_frames"] = max(entry["hold_frames"], self._edge_weak_support_hold_frames)
                elif entry["hold_frames"] > 0:
                    entry["hold_frames"] -= 1
            elif entry["positive_streak"] > 0:
                entry["positive_streak"] = max(1, entry["positive_streak"] - 1)
        else:
            if entry["confirmed"] and entry["hold_frames"] > 0 and entry["recent_max"] >= 2:
                entry["hold_frames"] -= 1
            else:
                entry["positive_streak"] = 0
                entry["hold_frames"] = 0
                entry["recent_max"] = 0
                entry["confirmed"] = 0

        if entry["positive_streak"] >= self._activation_frames:
            entry["confirmed"] = 1

        active = bool(
            entry["confirmed"]
            and (
                strong_overload
                or weak_support
                or entry["hold_frames"] > 0
            )
        )
        return active, strong_overload

    def _filter_matched_riders(
        self,
        riders: List[Track],
        ebike_bbox: List[float],
        edge_partial: bool,
    ) -> List[Track]:
        if edge_partial:
            return riders
        return [
            rider for rider in riders
            if self._is_viable_non_edge_rider(rider.detection.bbox, ebike_bbox)
        ]

    def _deduplicate_riders(self, riders: List[Track]) -> List[Track]:
        kept: List[Track] = []
        ordered = sorted(
            riders,
            key=lambda item: (
                item.track_id >= 0,
                self._area(item.detection.bbox),
                item.detection.confidence,
            ),
            reverse=True,
        )
        for rider in ordered:
            if any(self._same_person(rider.detection.bbox, existing.detection.bbox) for existing in kept):
                continue
            kept.append(rider)
        return kept

    def _same_person(self, rider_bbox: List[float], existing_bbox: List[float]) -> bool:
        overlap = self._overlap_ratio(rider_bbox, existing_bbox)
        center_score = self._center_proximity_score(rider_bbox, existing_bbox)
        return overlap >= 0.72 or (overlap >= 0.55 and center_score >= 0.82)

    def _is_viable_non_edge_rider(self, rider_bbox: List[float], ebike_bbox: List[float]) -> bool:
        rx1, ry1, rx2, ry2 = rider_bbox
        ex1, ey1, ex2, ey2 = ebike_bbox
        rider_w = max(1.0, rx2 - rx1)
        rider_h = max(1.0, ry2 - ry1)
        rider_area = self._area(rider_bbox)
        ebike_h = max(1.0, ey2 - ey1)
        ebike_w = max(1.0, ex2 - ex1)
        ebike_area = max(1.0, self._area(ebike_bbox))
        area_ratio = rider_area / ebike_area
        rider_bottom_ratio = (ry2 - ey1) / ebike_h
        rider_center_x = (rx1 + rx2) / 2
        horizontal_margin = ebike_w * 0.04
        center_inside = ex1 - horizontal_margin <= rider_center_x <= ex2 + horizontal_margin

        if rider_bottom_ratio >= 0.32 and center_inside:
            return True
        if area_ratio >= 0.20 and rider_h >= ebike_h * 0.48 and center_inside:
            return True
        return False

    def _has_edge_partial_proxy_support(self, riders: List[Track], ebike_bbox: List[float]) -> bool:
        if not riders:
            return False
        ebike_w = max(1.0, ebike_bbox[2] - ebike_bbox[0])
        ebike_h = max(1.0, ebike_bbox[3] - ebike_bbox[1])
        best_score = max(
            self._rider_ebike_score(rider.detection.bbox, ebike_bbox)
            for rider in riders
        )
        if ebike_w >= 170.0 and ebike_h >= 240.0:
            return best_score >= 1.45
        return ebike_w >= 170.0 and ebike_h >= 145.0 and (ebike_w / ebike_h) >= 1.18 and best_score >= 1.62

    def _center_proximity_score(self, box_a: List[float], box_b: List[float]) -> float:
        ax = (box_a[0] + box_a[2]) / 2
        ay = (box_a[1] + box_a[3]) / 2
        bx = (box_b[0] + box_b[2]) / 2
        by = (box_b[1] + box_b[3]) / 2
        aw = max(1.0, box_a[2] - box_a[0])
        ah = max(1.0, box_a[3] - box_a[1])
        bw = max(1.0, box_b[2] - box_b[0])
        bh = max(1.0, box_b[3] - box_b[1])
        norm = max((aw + bw) * 0.5, (ah + bh) * 0.5, 1.0)
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return max(0.0, 1.0 - distance / (norm * 1.5))

    def _prune_track_history(self, active_track_ids: set[int]) -> None:
        stale = [track_id for track_id in self._track_history if track_id not in active_track_ids]
        for track_id in stale:
            self._track_history.pop(track_id, None)

    def _match_track_id(self, detection: Detection, tracked_items: List[Track]) -> int:
        best_track_id = -1
        best_iou = 0.0
        for item in tracked_items:
            if item.detection.class_id != detection.class_id:
                continue
            iou = self._iou(detection.bbox, item.detection.bbox)
            if iou > 0.35 and iou > best_iou:
                best_iou = iou
                best_track_id = item.track_id
        return best_track_id

    def _is_edge_partial_ebike(self, bbox: List[float], frame_width: int) -> bool:
        x1, _, x2, _ = bbox
        box_w = max(1.0, x2 - x1)
        return x1 <= 4 or x2 >= frame_width - 4 or box_w < 48.0

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
        area_ratio = min(self._area(candidate_bbox), self._area(existing_bbox)) / max(
            self._area(candidate_bbox),
            self._area(existing_bbox),
            1.0,
        )
        if overlap > 0.62:
            return True
        if area_ratio < 0.45 and (
            self._contains(existing_bbox, candidate_bbox, margin=0.08)
            or self._contains(candidate_bbox, existing_bbox, margin=0.08)
        ):
            return True
        return False

    def _find_best_ebike_match(
        self,
        rider: Detection,
        ebikes: List[Track],
        frame_width: int = 0,
    ) -> Optional[Tuple[int, float]]:
        best_index: Optional[int] = None
        best_score = 0.0
        best_area = 0.0

        for ebike_index, ebike_track in enumerate(ebikes):
            score = self._match_score(rider, ebike_track.detection, frame_width)
            if score is None:
                continue
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

    def _match_score(
        self,
        rider: Detection,
        ebike: Detection,
        frame_width: int,
    ) -> Optional[float]:
        if self._match_classifier is not None and self._match_classifier.available:
            score = self._match_classifier.score(rider, ebike, frame_width=frame_width)
            if score is None or score < self._match_classifier.threshold:
                return None
            return score
        return self._rider_ebike_score(rider.bbox, ebike.bbox)

    def _build_match_classifier(self) -> Optional[PersonVehicleMatchClassifier]:
        if not settings.rules.match_classifier_enabled:
            return None
        try:
            classifier = PersonVehicleMatchClassifier(
                settings.rules.match_classifier_path,
                threshold=settings.rules.match_classifier_threshold,
            )
        except Exception:
            return None
        return classifier if classifier.available else None

    def _merged_overload_class(
        self,
        rider: Detection,
        ebike: Detection,
        edge_partial: bool,
    ) -> Optional[str]:
        rider_w = max(1.0, rider.bbox[2] - rider.bbox[0])
        rider_h = max(1.0, rider.bbox[3] - rider.bbox[1])
        ebike_w = max(1.0, ebike.bbox[2] - ebike.bbox[0])
        ebike_h = max(1.0, ebike.bbox[3] - ebike.bbox[1])
        overlap = self._overlap_ratio(rider.bbox, ebike.bbox)
        width_ratio = rider_w / ebike_w
        height_ratio = rider_h / ebike_h
        foot_in_box = self._foot_point_inside(
            rider.bbox,
            ebike.bbox,
            margin_x=0.18,
            margin_top=0.22,
            margin_bottom=0.18,
        )

        classic_merged = (
            ebike_w >= 140.0
            and ebike_h >= 140.0
            and rider_w >= 120.0
            and rider_h >= 260.0
            and width_ratio >= 0.78
            and height_ratio >= 1.52
            and overlap >= 0.62
            and foot_in_box
        )
        if classic_merged:
            return "overload_merged"
        if edge_partial:
            return None
        if self._looks_like_side_view_merged_double_rider(
            rider_bbox=rider.bbox,
            ebike_bbox=ebike.bbox,
        ):
            return "overload_merged_side_view"
        return None

    def _looks_like_side_view_merged_double_rider(
        self,
        rider_bbox: List[float],
        ebike_bbox: List[float],
    ) -> bool:
        rider_w = max(1.0, rider_bbox[2] - rider_bbox[0])
        rider_h = max(1.0, rider_bbox[3] - rider_bbox[1])
        ebike_w = max(1.0, ebike_bbox[2] - ebike_bbox[0])
        ebike_h = max(1.0, ebike_bbox[3] - ebike_bbox[1])
        overlap = self._overlap_ratio(rider_bbox, ebike_bbox)
        width_ratio = rider_w / ebike_w
        height_ratio = rider_h / ebike_h
        aspect_ratio = ebike_w / ebike_h
        foot_in_box = self._foot_point_inside(
            rider_bbox,
            ebike_bbox,
            margin_x=0.18,
            margin_top=0.22,
            margin_bottom=0.18,
        )

        return (
            ebike_w >= 180.0
            and ebike_h >= 145.0
            and aspect_ratio >= 1.28
            and width_ratio >= 0.46
            and width_ratio <= 0.64
            and height_ratio >= 1.32
            and overlap >= 0.66
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
        return legacy_rider_vehicle_score(rider_bbox, ebike_bbox)

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
