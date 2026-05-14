"""
Feature extraction for person-to-vehicle association.

The same feature contract is used by:
- PassengerRule at inference time.
- tools/export_match_samples.py for human correction CSVs.
- tools/train_match_classifier.py for lightweight model training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List

if TYPE_CHECKING:
    from app.core.types import Detection


FEATURE_NAMES = [
    "person_center_relative_x",
    "person_center_relative_y",
    "person_bottom_relative_y",
    "bbox_iou",
    "lower_half_iou",
    "horizontal_overlap",
    "vertical_overlap",
    "overlap_ratio",
    "area_ratio",
    "height_ratio",
    "width_ratio",
    "bottom_distance_ratio",
    "center_distance_ratio",
    "person_conf",
    "vehicle_conf",
    "is_edge_vehicle",
    "legacy_match_score",
]


@dataclass(frozen=True)
class MatchFeatureSet:
    """Named feature vector for a person/vehicle pair."""

    values: Dict[str, float]

    def as_vector(self, feature_names: Iterable[str] = FEATURE_NAMES) -> List[float]:
        return [float(self.values.get(name, 0.0)) for name in feature_names]


def extract_match_features(
    person: Detection,
    vehicle: Detection,
    frame_width: int = 0,
) -> MatchFeatureSet:
    """Build numeric features for one person-to-vehicle candidate pair."""
    px1, py1, px2, py2 = person.bbox
    vx1, vy1, vx2, vy2 = vehicle.bbox

    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)
    vehicle_w = max(1.0, vx2 - vx1)
    vehicle_h = max(1.0, vy2 - vy1)

    person_cx = (px1 + px2) / 2.0
    person_cy = (py1 + py2) / 2.0
    vehicle_cx = (vx1 + vx2) / 2.0
    vehicle_cy = (vy1 + vy2) / 2.0

    lower_half = [px1, py1 + person_h * 0.45, px2, py2]
    diag = max((vehicle_w**2 + vehicle_h**2) ** 0.5, 1.0)

    values = {
        "person_center_relative_x": (person_cx - vx1) / vehicle_w,
        "person_center_relative_y": (person_cy - vy1) / vehicle_h,
        "person_bottom_relative_y": (py2 - vy1) / vehicle_h,
        "bbox_iou": _iou(person.bbox, vehicle.bbox),
        "lower_half_iou": _iou(lower_half, vehicle.bbox),
        "horizontal_overlap": _axis_overlap(px1, px2, vx1, vx2),
        "vertical_overlap": _axis_overlap(py1, py2, vy1, vy2),
        "overlap_ratio": _overlap_ratio(person.bbox, vehicle.bbox),
        "area_ratio": _area(person.bbox) / max(1.0, _area(vehicle.bbox)),
        "height_ratio": person_h / vehicle_h,
        "width_ratio": person_w / vehicle_w,
        "bottom_distance_ratio": (py2 - vy2) / vehicle_h,
        "center_distance_ratio": (((person_cx - vehicle_cx) ** 2 + (person_cy - vehicle_cy) ** 2) ** 0.5) / diag,
        "person_conf": float(person.confidence),
        "vehicle_conf": float(vehicle.confidence),
        "is_edge_vehicle": 1.0 if _is_edge_vehicle(vehicle.bbox, frame_width) else 0.0,
        "legacy_match_score": legacy_rider_vehicle_score(person.bbox, vehicle.bbox),
    }
    return MatchFeatureSet(values)


def legacy_rider_vehicle_score(person_bbox: List[float], vehicle_bbox: List[float]) -> float:
    """
    Keep the original geometric score as an explicit feature.

    This lets a lightweight classifier learn when to trust or override the old
    heuristic instead of replacing it with an opaque signal.
    """
    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox
    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)
    vehicle_w = max(1.0, vx2 - vx1)
    vehicle_h = max(1.0, vy2 - vy1)

    foot_x = (px1 + px2) / 2.0
    foot_y = py2
    margin_x = vehicle_w * 0.18
    margin_top = vehicle_h * 0.22
    margin_bottom = vehicle_h * 0.16
    foot_supported = (
        vx1 - margin_x <= foot_x <= vx2 + margin_x
        and vy1 - margin_top <= foot_y <= vy2 + margin_bottom
    )

    if not foot_supported and person_h > vehicle_h * 1.18 and foot_y > vy2 + vehicle_h * 0.26:
        return 0.0

    lower_half = [px1, py1 + (py2 - py1) * 0.45, px2, py2]
    score = 0.0
    if foot_supported:
        score += 0.75
    score += _iou(lower_half, vehicle_bbox)
    score += _axis_overlap(lower_half[0], lower_half[2], vehicle_bbox[0], vehicle_bbox[2]) * 0.35
    score += min(vehicle_w / person_w, 6.0) * 0.05
    score += min(vehicle_h / person_h, 2.5) * 0.10
    return score


def _is_edge_vehicle(bbox: List[float], frame_width: int) -> bool:
    if frame_width <= 0:
        return False
    x1, _, x2, _ = bbox
    box_w = max(1.0, x2 - x1)
    return x1 <= 4 or x2 >= frame_width - 4 or box_w < 48.0


def _axis_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    width = max(1.0, min(a2 - a1, b2 - b1))
    return overlap / width


def _intersection_area(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)


def _area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _overlap_ratio(box_a: List[float], box_b: List[float]) -> float:
    inter_area = _intersection_area(box_a, box_b)
    if inter_area <= 0:
        return 0.0
    return inter_area / max(1.0, min(_area(box_a), _area(box_b)))


def _iou(box_a: List[float], box_b: List[float]) -> float:
    inter_area = _intersection_area(box_a, box_b)
    if inter_area <= 0:
        return 0.0
    area_a = _area(box_a)
    area_b = _area(box_b)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0
