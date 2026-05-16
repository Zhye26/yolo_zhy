#!/usr/bin/env python3
"""
Yolov8n-only overload validation on real video.

This script uses only COCO classes from yolov8n.pt:
- 0: person
- 3: motorcycle

It treats motorcycle detections as two-wheeler candidates and flags a possible
overload when two or more person boxes match one two-wheeler across frames.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from methods.mot_trackers_ljt import AVAILABLE_TRACKERS, create_mot_tracker


PERSON_CLASS_ID = 0
MOTORCYCLE_CLASS_ID = 3
TARGET_CLASSES = [PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID]
CLASS_NAMES = {
    PERSON_CLASS_ID: "person",
    MOTORCYCLE_CLASS_ID: "motorcycle",
}


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: List[float]

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class TrackedDetection(Detection):
    raw_track_id: int = 0
    stable_id: str = ""


@dataclass
class VehicleTrack:
    track_id: int
    bbox: List[float]
    class_id: int
    confidence: float
    hits: int = 1
    missed: int = 0
    age: int = 1
    positive_streak: int = 0
    confirmed_overload: bool = False
    last_rider_count: int = 0
    ever_confirmed: bool = False
    speed_px: float = 0.0
    observed_frames: int = 0
    high_quality_observed_frames: int = 0
    overload_frame_count: int = 0
    high_quality_overload_frames: int = 0
    high_quality_non_overload_frames: int = 0
    overload_evidence_score: float = 0.0
    non_overload_evidence_score: float = 0.0
    overload_state_switches: int = 0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def center_distance(box_a: List[float], box_b: List[float]) -> float:
    ax, ay = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    bx, by = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def mean_confidence(detections: List[Detection]) -> float:
    if not detections:
        return 0.0
    return sum(det.confidence for det in detections) / len(detections)


def update_overload_evidence_state(
    state: Dict[str, int | float | bool],
    *,
    rider_count: int,
    vehicle_confidence: float,
    matched_people: List[Detection],
    match_scores: List[float],
    speed_px: float,
    confirm_frames: int,
) -> bool:
    """Update lifecycle-level overload evidence and return confirmed state."""

    avg_person_conf = mean_confidence(matched_people)
    avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
    has_rider_evidence = rider_count > 0
    vehicle_quality = clamp((vehicle_confidence - 0.20) / 0.50, 0.0, 1.0)
    person_quality = 1.0 if not has_rider_evidence else clamp((avg_person_conf - 0.20) / 0.50, 0.0, 1.0)
    association_quality = 1.0 if not has_rider_evidence else clamp(avg_match_score / 1.20, 0.0, 1.0)
    speed_factor = clamp(speed_px / 8.0, 0.50, 1.50)
    frame_quality = vehicle_quality * person_quality * association_quality
    weighted_evidence = frame_quality * speed_factor
    high_quality = vehicle_confidence >= 0.45 and (not has_rider_evidence or avg_person_conf >= 0.35)
    strong_overload_frame = (
        rider_count >= 2
        and vehicle_confidence >= 0.45
        and avg_person_conf >= 0.35
        and avg_match_score >= 0.95
    )

    state["observed_frames"] = int(state.get("observed_frames", 0)) + 1
    if high_quality:
        state["high_quality_observed_frames"] = int(state.get("high_quality_observed_frames", 0)) + 1

    if rider_count >= 2:
        state["overload_frame_count"] = int(state.get("overload_frame_count", 0)) + 1
        state["overload_evidence_score"] = float(state.get("overload_evidence_score", 0.0)) + weighted_evidence
        if high_quality:
            state["high_quality_overload_frames"] = int(state.get("high_quality_overload_frames", 0)) + 1
    else:
        state["non_overload_evidence_score"] = float(state.get("non_overload_evidence_score", 0.0)) + weighted_evidence
        if high_quality and speed_px >= 4.0:
            state["high_quality_non_overload_frames"] = int(state.get("high_quality_non_overload_frames", 0)) + 1

    min_observed = max(2, confirm_frames)
    observed = int(state.get("observed_frames", 0))
    high_quality_observed = int(state.get("high_quality_observed_frames", 0))
    overload_frames = int(state.get("overload_frame_count", 0))
    overload_evidence = float(state.get("overload_evidence_score", 0.0))
    non_overload_evidence = float(state.get("non_overload_evidence_score", 0.0))
    high_quality_non_overload = int(state.get("high_quality_non_overload_frames", 0))

    time_ratio = overload_frames / observed if observed else 0.0
    evidence_ratio = overload_evidence / (overload_evidence + non_overload_evidence) if (overload_evidence + non_overload_evidence) > 0 else 0.0
    previous = bool(state.get("confirmed_overload", False))

    confirm_threshold = 0.55
    clear_threshold = 0.35
    confirmed = previous
    if not previous:
        if strong_overload_frame and (observed >= 2 or speed_px >= 4.0):
            confirmed = True
        elif observed >= min_observed and high_quality_observed >= 1 and overload_evidence >= 0.65:
            confirmed = time_ratio >= confirm_threshold or evidence_ratio >= confirm_threshold
    else:
        if observed >= min_observed and time_ratio < clear_threshold and evidence_ratio < clear_threshold:
            confirmed = False

    if (
        confirmed
        and high_quality_non_overload >= max(2, confirm_frames)
        and non_overload_evidence > overload_evidence * 1.20
    ):
        confirmed = False

    if confirmed != previous:
        state["overload_state_switches"] = int(state.get("overload_state_switches", 0)) + 1
    state["confirmed_overload"] = confirmed
    state["last_overload_time_ratio"] = time_ratio
    state["last_overload_evidence_ratio"] = evidence_ratio
    state["last_speed_px"] = speed_px
    return confirmed


def update_simple_overload_stats(
    state: Dict[str, int | float | bool],
    *,
    rider_count: int,
    vehicle_confidence: float,
    matched_people: List[Detection],
    match_scores: List[float],
    speed_px: float,
    ratio_threshold: float = 0.25,
    min_observed_frames: int = 8,
) -> bool:
    avg_person_conf = mean_confidence(matched_people)
    avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
    raw_overload = rider_count >= 2
    high_conf = vehicle_confidence >= 0.40 and (rider_count == 0 or avg_person_conf >= 0.30)
    moving = speed_px >= 3.0

    state["observed_frames"] = int(state.get("observed_frames", 0)) + 1
    state["raw_overload_frames"] = int(state.get("raw_overload_frames", 0)) + int(raw_overload)
    state["high_conf_observed_frames"] = int(state.get("high_conf_observed_frames", 0)) + int(high_conf)
    state["high_conf_overload_frames"] = int(state.get("high_conf_overload_frames", 0)) + int(high_conf and raw_overload)
    state["moving_observed_frames"] = int(state.get("moving_observed_frames", 0)) + int(moving)
    state["moving_overload_frames"] = int(state.get("moving_overload_frames", 0)) + int(moving and raw_overload)
    state["max_rider_count"] = max(int(state.get("max_rider_count", 0)), rider_count)
    state["last_rider_count"] = rider_count
    state["sum_vehicle_conf"] = float(state.get("sum_vehicle_conf", 0.0)) + vehicle_confidence
    state["sum_match_score"] = float(state.get("sum_match_score", 0.0)) + avg_match_score
    state["sum_speed_px"] = float(state.get("sum_speed_px", 0.0)) + speed_px

    observed = int(state["observed_frames"])
    raw_ratio = int(state["raw_overload_frames"]) / observed if observed else 0.0
    high_conf_observed = int(state["high_conf_observed_frames"])
    high_conf_ratio = (
        int(state["high_conf_overload_frames"]) / high_conf_observed if high_conf_observed else 0.0
    )
    moving_observed = int(state["moving_observed_frames"])
    moving_ratio = int(state["moving_overload_frames"]) / moving_observed if moving_observed else 0.0

    state["raw_overload_ratio"] = raw_ratio
    state["high_conf_overload_ratio"] = high_conf_ratio
    state["moving_overload_ratio"] = moving_ratio
    state["avg_vehicle_conf"] = float(state["sum_vehicle_conf"]) / observed
    state["avg_match_score"] = float(state["sum_match_score"]) / observed
    state["avg_speed_px"] = float(state["sum_speed_px"]) / observed

    status = overload_status_from_stats(
        state,
        ratio_threshold=ratio_threshold,
        min_observed_frames=min_observed_frames,
    )
    confirmed = status == "CONFIRMED"
    state["overload_status"] = status
    state["suspected_overload"] = status in {"SUSPECTED", "CONFIRMED"}
    state["confirmed_overload"] = confirmed
    state["ever_confirmed_overload"] = bool(state.get("ever_confirmed_overload", False)) or confirmed
    return confirmed


def overload_status_from_stats(
    state: Dict[str, int | float | bool],
    *,
    ratio_threshold: float = 0.25,
    min_observed_frames: int = 8,
) -> str:
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    observed = int(state.get("observed_frames", 0) or 0)
    raw_frames = int(state.get("raw_overload_frames", 0) or 0)
    high_conf_observed = int(state.get("high_conf_observed_frames", 0) or 0)
    high_conf_frames = int(state.get("high_conf_overload_frames", 0) or 0)
    moving_observed = int(state.get("moving_observed_frames", 0) or 0)
    moving_frames = int(state.get("moving_overload_frames", 0) or 0)
    max_rider_count = int(state.get("max_rider_count", 0) or 0)
    raw_ratio = float(state.get("raw_overload_ratio", ratio(raw_frames, observed)) or 0.0)
    high_conf_ratio = float(state.get("high_conf_overload_ratio", ratio(high_conf_frames, high_conf_observed)) or 0.0)
    moving_ratio = float(state.get("moving_overload_ratio", ratio(moving_frames, moving_observed)) or 0.0)

    has_enough_history = observed >= min_observed_frames
    repeated_raw = raw_frames >= 3 and raw_ratio >= ratio_threshold
    reliable_high_conf = high_conf_frames >= 3 and high_conf_ratio >= max(0.20, ratio_threshold * 0.80)
    reliable_moving = moving_observed >= min_observed_frames and moving_frames >= 3 and moving_ratio >= ratio_threshold
    if has_enough_history and max_rider_count >= 2 and (repeated_raw or reliable_high_conf or reliable_moving):
        return "CONFIRMED"
    if raw_frames > 0 or max_rider_count >= 2:
        return "SUSPECTED"
    return "NORMAL"


@dataclass
class FrameVehicleResult:
    track_id: int | str
    detection: Detection
    matched_people: List[Detection]
    match_scores: List[float]
    raw_overload: bool
    confirmed_overload: bool


def sync_cuda_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def overlap_ratio(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0


def expand_box(box: List[float], x_ratio: float, top_ratio: float, bottom_ratio: float) -> List[float]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [
        x1 - w * x_ratio,
        y1 - h * top_ratio,
        x2 + w * x_ratio,
        y2 + h * bottom_ratio,
    ]


def point_in_box(x: float, y: float, box: List[float]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def person_vehicle_match_score(person: Detection, vehicle: Detection) -> float:
    px1, py1, px2, py2 = person.bbox
    vx1, vy1, vx2, vy2 = vehicle.bbox
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    vw = max(1.0, vx2 - vx1)
    vh = max(1.0, vy2 - vy1)

    foot_x = (px1 + px2) / 2
    foot_y = py2
    person_center_x = (px1 + px2) / 2
    vehicle_center_x = (vx1 + vx2) / 2

    support_box = expand_box(vehicle.bbox, x_ratio=0.45, top_ratio=1.35, bottom_ratio=0.35)
    lower_body = [px1, py1 + ph * 0.45, px2, py2]
    lower_vehicle = [vx1 - vw * 0.20, vy1 - vh * 0.20, vx2 + vw * 0.20, vy2 + vh * 0.35]

    if not point_in_box(foot_x, foot_y, support_box):
        return 0.0
    if foot_y < vy1 - vh * 0.35 or foot_y > vy2 + vh * 0.45:
        return 0.0
    if not (vx1 - vw * 0.55 <= person_center_x <= vx2 + vw * 0.55):
        return 0.0

    score = 0.0
    score += 0.45
    if vx1 - vw * 0.45 <= person_center_x <= vx2 + vw * 0.45:
        score += 0.25
    score += min(iou(lower_body, lower_vehicle) * 3.0, 0.25)
    score += min(overlap_ratio(lower_body, lower_vehicle) * 0.25, 0.25)

    horizontal_gap = abs(person_center_x - vehicle_center_x) / max(vw, 1.0)
    if horizontal_gap > 1.0:
        score -= min((horizontal_gap - 1.0) * 0.35, 0.35)
    if pw > vw * 1.8 and overlap_ratio(person.bbox, support_box) < 0.25:
        score -= 0.20

    return max(0.0, score)


def result_to_detections(result, min_area: float) -> List[Detection]:
    detections: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(boxes, confs, clss):
        det = Detection(class_id=int(cls_id), confidence=float(conf), bbox=[float(v) for v in box.tolist()])
        if det.area >= min_area:
            detections.append(det)
    return detections


class VehicleTracker:
    def __init__(self, iou_thresh: float, max_missed: int, confirm_frames: int):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.confirm_frames = confirm_frames
        self.next_track_id = 1
        self.tracks: Dict[int, VehicleTrack] = {}
        self.confirmed_track_ids: set[int] = set()

    def update(
        self,
        vehicles: List[Detection],
        rider_counts: Dict[int, int],
        grouped_people: Optional[Dict[int, List[Detection]]] = None,
        match_scores: Optional[Dict[int, List[float]]] = None,
    ) -> Dict[int, int]:
        matches: Dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            if track.missed > self.max_missed:
                continue
            for det_idx, det in enumerate(vehicles):
                score = iou(track.bbox, det.bbox)
                if score >= self.iou_thresh:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True)

        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(det_idx)
            matches[det_idx] = track_id

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track.missed += 1
                track.age += 1

        for det_idx, det in enumerate(vehicles):
            if det_idx in matches:
                track = self.tracks[matches[det_idx]]
                track.speed_px = center_distance(track.bbox, det.bbox)
                track.bbox = det.bbox
                track.class_id = det.class_id
                track.confidence = det.confidence
                track.hits += 1
                track.missed = 0
                track.age += 1
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = VehicleTrack(
                    track_id=track_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    confidence=det.confidence,
                )
                matches[det_idx] = track_id

        for det_idx, track_id in matches.items():
            count = rider_counts.get(det_idx, 0)
            track = self.tracks[track_id]
            track.last_rider_count = count
            track.confirmed_overload = update_overload_evidence_state(
                track.__dict__,
                rider_count=count,
                vehicle_confidence=track.confidence,
                matched_people=(grouped_people or {}).get(det_idx, []),
                match_scores=(match_scores or {}).get(det_idx, []),
                speed_px=track.speed_px,
                confirm_frames=self.confirm_frames,
            )
            if track.confirmed_overload:
                track.ever_confirmed = True
                self.confirmed_track_ids.add(track_id)

        expired = [track_id for track_id, track in self.tracks.items() if track.missed > self.max_missed]
        for track_id in expired:
            self.tracks.pop(track_id, None)

        return matches

    def is_confirmed(self, track_id: int) -> bool:
        track = self.tracks.get(track_id)
        return bool(track and track.confirmed_overload)


class ByteVehicleTracker:
    def __init__(
        self,
        track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        confirm_frames: int,
        max_missed: int,
    ):
        try:
            from boxmot import BYTETracker
        except ImportError as exc:
            raise RuntimeError("ByteTrack requires boxmot. Install project tracking dependencies first.") from exc

        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=30,
        )
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.frame_index = 0
        self.confirmed_track_ids: set[int] = set()
        self.states: Dict[int, Dict[str, int | bool]] = {}

    def update(
        self,
        vehicles: List[Detection],
        rider_counts: Dict[int, int],
        frame,
        grouped_people: Optional[Dict[int, List[Detection]]] = None,
        match_scores: Optional[Dict[int, List[float]]] = None,
    ) -> Dict[int, int]:
        self.frame_index += 1
        matches: Dict[int, int] = {}
        if not vehicles:
            self._decay_missing_tracks(set())
            return matches

        det_array = np.array(
            [
                [
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[2],
                    det.bbox[3],
                    det.confidence,
                    det.class_id,
                ]
                for det in vehicles
            ],
            dtype=np.float32,
        )
        tracks_output = self.tracker.update(det_array, frame)
        matches = self._match_outputs_to_detections(tracks_output, vehicles)

        current_track_ids = set(matches.values())
        for det_idx, track_id in matches.items():
            state = self.states.setdefault(
                track_id,
                {
                    "observed_frames": 0,
                    "high_quality_observed_frames": 0,
                    "overload_frame_count": 0,
                    "high_quality_overload_frames": 0,
                    "high_quality_non_overload_frames": 0,
                    "overload_evidence_score": 0.0,
                    "non_overload_evidence_score": 0.0,
                    "overload_state_switches": 0,
                    "last_seen": self.frame_index,
                    "confirmed_overload": False,
                },
            )
            count = rider_counts.get(det_idx, 0)
            prev_center = state.get("last_center")
            speed_px = 0.0
            if prev_center:
                speed_px = ((vehicles[det_idx].center[0] - float(prev_center[0])) ** 2 + (vehicles[det_idx].center[1] - float(prev_center[1])) ** 2) ** 0.5
            state["last_center"] = vehicles[det_idx].center
            state["last_seen"] = self.frame_index
            update_overload_evidence_state(
                state,
                rider_count=count,
                vehicle_confidence=vehicles[det_idx].confidence,
                matched_people=(grouped_people or {}).get(det_idx, []),
                match_scores=(match_scores or {}).get(det_idx, []),
                speed_px=speed_px,
                confirm_frames=self.confirm_frames,
            )
            if state["confirmed_overload"]:
                self.confirmed_track_ids.add(track_id)

        self._decay_missing_tracks(current_track_ids)
        for det_idx in range(len(vehicles)):
            matches.setdefault(det_idx, -(det_idx + 1))
        return matches

    def _match_outputs_to_detections(self, tracks_output, vehicles: List[Detection]) -> Dict[int, int]:
        if tracks_output is None or len(tracks_output) == 0:
            return {}

        candidates: List[Tuple[float, int, int]] = []
        for row in tracks_output:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, track_id = row[:5]
            class_id = int(row[6]) if len(row) > 6 else None
            track_box = [float(x1), float(y1), float(x2), float(y2)]
            for det_idx, det in enumerate(vehicles):
                if class_id is not None and class_id != det.class_id:
                    continue
                score = iou(track_box, det.bbox)
                if score > 0.05:
                    candidates.append((score, int(track_id), det_idx))

        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matches: Dict[int, int] = {}
        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            matches[det_idx] = track_id
        return matches

    def _decay_missing_tracks(self, current_track_ids: set[int]) -> None:
        expired = []
        for track_id, state in self.states.items():
            if track_id in current_track_ids:
                continue
            missed = self.frame_index - int(state["last_seen"])
            if missed > self.max_missed:
                expired.append(track_id)
                continue
            state["confirmed_overload"] = False
        for track_id in expired:
            self.states.pop(track_id, None)

    def is_confirmed(self, track_id: int) -> bool:
        state = self.states.get(track_id)
        return bool(state and state["confirmed_overload"])


class GenericMotVehicleTracker:
    def __init__(self, tracker_name: str, args):
        self.tracker, self.tracker_name = create_mot_tracker(tracker_name, args)
        self.confirm_frames = args.confirm_frames
        self.max_missed = args.max_missed
        self.frame_index = 0
        self.confirmed_track_ids: set[int] = set()
        self.states: Dict[int, Dict[str, int | bool]] = {}

    def update(
        self,
        vehicles: List[Detection],
        rider_counts: Dict[int, int],
        frame,
        grouped_people: Optional[Dict[int, List[Detection]]] = None,
        match_scores: Optional[Dict[int, List[float]]] = None,
    ) -> Dict[int, int]:
        self.frame_index += 1
        matches: Dict[int, int] = {}
        if not vehicles:
            self._decay_missing_tracks(set())
            return matches

        det_array = np.array(
            [
                [
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[2],
                    det.bbox[3],
                    det.confidence,
                    det.class_id,
                ]
                for det in vehicles
            ],
            dtype=np.float32,
        )
        tracks_output = self.tracker.update(det_array, frame)
        matches = self._match_outputs_to_detections(tracks_output, vehicles)

        current_track_ids = set(matches.values())
        for det_idx, track_id in matches.items():
            state = self.states.setdefault(
                track_id,
                {
                    "observed_frames": 0,
                    "high_quality_observed_frames": 0,
                    "overload_frame_count": 0,
                    "high_quality_overload_frames": 0,
                    "high_quality_non_overload_frames": 0,
                    "overload_evidence_score": 0.0,
                    "non_overload_evidence_score": 0.0,
                    "overload_state_switches": 0,
                    "last_seen": self.frame_index,
                    "confirmed_overload": False,
                },
            )
            count = rider_counts.get(det_idx, 0)
            prev_center = state.get("last_center")
            speed_px = 0.0
            if prev_center:
                speed_px = ((vehicles[det_idx].center[0] - float(prev_center[0])) ** 2 + (vehicles[det_idx].center[1] - float(prev_center[1])) ** 2) ** 0.5
            state["last_center"] = vehicles[det_idx].center
            state["last_seen"] = self.frame_index
            update_overload_evidence_state(
                state,
                rider_count=count,
                vehicle_confidence=vehicles[det_idx].confidence,
                matched_people=(grouped_people or {}).get(det_idx, []),
                match_scores=(match_scores or {}).get(det_idx, []),
                speed_px=speed_px,
                confirm_frames=self.confirm_frames,
            )
            if state["confirmed_overload"]:
                self.confirmed_track_ids.add(track_id)

        self._decay_missing_tracks(current_track_ids)
        for det_idx in range(len(vehicles)):
            matches.setdefault(det_idx, -(det_idx + 1))
        return matches

    def _match_outputs_to_detections(self, tracks_output, vehicles: List[Detection]) -> Dict[int, int]:
        if tracks_output is None or len(tracks_output) == 0:
            return {}

        candidates: List[Tuple[float, int, int]] = []
        for row in tracks_output:
            if len(row) < 5:
                continue
            x1, y1, x2, y2, track_id = row[:5]
            class_id = int(row[6]) if len(row) > 6 else None
            track_box = [float(x1), float(y1), float(x2), float(y2)]
            for det_idx, det in enumerate(vehicles):
                if class_id is not None and class_id != det.class_id:
                    continue
                score = iou(track_box, det.bbox)
                if score > 0.05:
                    candidates.append((score, int(track_id), det_idx))

        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matches: Dict[int, int] = {}
        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            matches[det_idx] = track_id
        return matches

    def _decay_missing_tracks(self, current_track_ids: set[int]) -> None:
        expired = []
        for track_id, state in self.states.items():
            if track_id in current_track_ids:
                continue
            missed = self.frame_index - int(state["last_seen"])
            if missed > self.max_missed:
                expired.append(track_id)
                continue
            state["confirmed_overload"] = False
        for track_id in expired:
            self.states.pop(track_id, None)

    def is_confirmed(self, track_id: int) -> bool:
        state = self.states.get(track_id)
        return bool(state and state["confirmed_overload"])


# Association-specific tracker implementation is in methods/association_ljt.py.

ASSOCIATION_TRACKER_PREFIX = "assoc_"
ASSOCIATION_MOT_TRACKERS = tuple(f"{ASSOCIATION_TRACKER_PREFIX}{name}" for name in sorted(AVAILABLE_TRACKERS) if name != "byte")


def is_association_mot_tracker_name(name: str) -> bool:
    return name.strip().lower().startswith(ASSOCIATION_TRACKER_PREFIX)


def association_base_tracker_name(name: str) -> str:
    return name.strip().lower()[len(ASSOCIATION_TRACKER_PREFIX) :]


def uses_association_scene(tracker_name: str) -> bool:
    return tracker_name == "association" or is_association_mot_tracker_name(tracker_name)


def create_vehicle_tracker(args):
    if args.tracker == "association":
        from methods.association_ljt import AssociationByteTracker

        tracker = AssociationByteTracker(
            track_thresh=args.byte_track_thresh,
            match_thresh=args.byte_match_thresh,
            track_buffer=args.byte_track_buffer,
            confirm_frames=args.confirm_frames,
            max_missed=args.max_missed,
            association_min_hits=args.association_min_hits,
            association_lock_frames=args.association_lock_frames,
            association_unbind_frames=args.association_unbind_frames,
            association_switch_margin=args.association_switch_margin,
        )
        print("[tracker] association")
        return tracker, "association"

    if is_association_mot_tracker_name(args.tracker):
        from methods.association_ljt import AssociationMotTracker

        base_tracker_name = association_base_tracker_name(args.tracker)
        if base_tracker_name == "byte":
            raise RuntimeError("Use 'association' for ByteTrack with association constraints.")
        person_tracker, person_name = create_mot_tracker(base_tracker_name, args)
        vehicle_tracker, vehicle_name = create_mot_tracker(base_tracker_name, args)
        tracker = AssociationMotTracker(
            person_tracker=person_tracker,
            vehicle_tracker=vehicle_tracker,
            confirm_frames=args.confirm_frames,
            max_missed=args.max_missed,
            association_min_hits=args.association_min_hits,
            association_lock_frames=args.association_lock_frames,
            association_unbind_frames=args.association_unbind_frames,
            association_switch_margin=args.association_switch_margin,
        )
        tracker_name = f"{ASSOCIATION_TRACKER_PREFIX}{vehicle_name}"
        print(f"[tracker] {tracker_name}")
        return tracker, tracker_name

    if args.tracker == "auto":
        try:
            tracker = GenericMotVehicleTracker("byte", args)
            print("[tracker] byte")
            return tracker, "byte"
        except RuntimeError:
            print("[tracker] byte unavailable, falling back to iou")

    if args.tracker in AVAILABLE_TRACKERS:
        tracker = GenericMotVehicleTracker(args.tracker, args)
        print(f"[tracker] {tracker.tracker_name}")
        return tracker, tracker.tracker_name

    print("[tracker] iou")
    return (
        VehicleTracker(
            iou_thresh=args.track_iou,
            max_missed=args.max_missed,
            confirm_frames=args.confirm_frames,
        ),
        "iou",
    )


def match_people_to_vehicles(
    people: List[Detection],
    vehicles: List[Detection],
    match_thresh: float,
) -> Tuple[Dict[int, List[Detection]], Dict[int, List[float]]]:
    grouped: Dict[int, List[Detection]] = {idx: [] for idx in range(len(vehicles))}
    scores: Dict[int, List[float]] = {idx: [] for idx in range(len(vehicles))}
    candidates: List[Tuple[float, int, int]] = []

    for person_idx, person in enumerate(people):
        for vehicle_idx, vehicle in enumerate(vehicles):
            score = person_vehicle_match_score(person, vehicle)
            if score >= match_thresh:
                candidates.append((score, person_idx, vehicle_idx))
    candidates.sort(reverse=True)

    used_people: set[int] = set()
    for score, person_idx, vehicle_idx in candidates:
        if person_idx in used_people:
            continue
        used_people.add(person_idx)
        grouped[vehicle_idx].append(people[person_idx])
        scores[vehicle_idx].append(score)

    return grouped, scores


def draw_frame(
    frame,
    people: List[Detection],
    vehicle_results: List[FrameVehicleResult],
    frame_index: int,
    timestamp_sec: float,
    fps_text: str,
):
    vis = frame.copy()

    for person in people:
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 1)
        cv2.putText(vis, f"person {person.confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    for item in vehicle_results:
        det = item.detection
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = (0, 0, 255) if item.confirmed_overload else (0, 165, 255) if item.raw_overload else (255, 160, 0)
        label = "OVERLOAD" if item.confirmed_overload else "candidate" if item.raw_overload else det.class_name
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"#{item.track_id} {label} riders={len(item.matched_people)} {det.confidence:.2f}",
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        vcx = int((x1 + x2) / 2)
        vcy = int((y1 + y2) / 2)
        for person in item.matched_people:
            pcx = int((person.bbox[0] + person.bbox[2]) / 2)
            pcy = int(person.bbox[3])
            cv2.line(vis, (vcx, vcy), (pcx, pcy), color, 2)
            stable_person_id = getattr(person, "stable_id", "")
            if stable_person_id:
                cv2.putText(
                    vis,
                    stable_person_id,
                    (pcx + 4, pcy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 38), (0, 0, 0), -1)
    header = f"yolov8n-ljt frame={frame_index} t={timestamp_sec:.2f}s {fps_text}"
    cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Yolov8n-only e-bike overload validation")
    parser.add_argument("--video", default="data/test_video/docker-compose.cpu.mp4")
    parser.add_argument("--out", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "weights" / "yolov8n.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-thresh", type=float, default=1.05)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument(
        "--tracker",
        choices=["auto", "iou", "association", *ASSOCIATION_MOT_TRACKERS, *sorted(AVAILABLE_TRACKERS)],
        default="association",
    )
    parser.add_argument("--track-iou", type=float, default=0.18)
    parser.add_argument("--max-missed", type=int, default=6)
    parser.add_argument("--byte-track-thresh", type=float, default=0.25)
    parser.add_argument("--byte-match-thresh", type=float, default=0.8)
    parser.add_argument("--byte-track-buffer", type=int, default=30)
    parser.add_argument("--mot-min-hits", type=int, default=3)
    parser.add_argument("--reid-weights", default=str(Path(__file__).resolve().parents[2] / "weights" / "osnet_x0_25_msmt17.pt"))
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--reid-fp16", action="store_true")
    parser.add_argument("--association-min-hits", type=int, default=4)
    parser.add_argument("--association-lock-frames", type=int, default=20)
    parser.add_argument("--association-unbind-frames", type=int, default=15)
    parser.add_argument("--association-switch-margin", type=float, default=0.35)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else Path(f"data/model_compare_video_vis/yolov8n-overload-{args.tracker}-ljt.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_path.with_name(f"{out_path.stem}_frames.csv")
    summary_path = Path(args.summary_csv) if args.summary_csv else out_path.with_name(f"{out_path.stem}_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tracker, tracker_name = create_vehicle_tracker(args)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = source_fps / max(1, args.frame_stride)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open output video writer: {out_path}")

    model = YOLO(args.model)
    ok, first_frame = cap.read()
    if not ok:
        writer.release()
        cap.release()
        raise SystemExit(f"Video has no readable frames: {video_path}")

    for _ in range(args.warmup):
        model.predict(first_frame, classes=TARGET_CLASSES, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
    sync_cuda_if_available()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows: List[Dict[str, object]] = []
    timing_ms: List[float] = []
    processed = 0
    frame_index = -1
    frames_with_raw_overload = 0
    frames_with_confirmed_overload = 0
    total_people = 0
    total_vehicles = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % max(1, args.frame_stride) != 0:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        timestamp_sec = frame_index / source_fps if source_fps > 0 else 0.0

        sync_cuda_if_available()
        start = time.perf_counter()
        result = model.predict(
            frame,
            classes=TARGET_CLASSES,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        sync_cuda_if_available()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timing_ms.append(elapsed_ms)

        detections = result_to_detections(result, min_area=args.min_area)
        people = [det for det in detections if det.class_id == PERSON_CLASS_ID]
        vehicles = [det for det in detections if det.class_id == MOTORCYCLE_CLASS_ID]
        total_people += len(people)
        total_vehicles += len(vehicles)

        if uses_association_scene(tracker_name):
            track_matches, grouped_people, match_scores = tracker.update_scene(
                people,
                vehicles,
                frame,
                args.match_thresh,
            )
        else:
            grouped_people, match_scores = match_people_to_vehicles(people, vehicles, args.match_thresh)
            rider_counts = {idx: len(grouped_people[idx]) for idx in range(len(vehicles))}
            if tracker_name in AVAILABLE_TRACKERS:
                track_matches = tracker.update(vehicles, rider_counts, frame, grouped_people, match_scores)
            else:
                track_matches = tracker.update(vehicles, rider_counts, grouped_people, match_scores)

        frame_results: List[FrameVehicleResult] = []
        has_raw_overload = False
        has_confirmed_overload = False
        for vehicle_idx, vehicle in enumerate(vehicles):
            track_id = track_matches[vehicle_idx]
            matched_people = grouped_people[vehicle_idx]
            raw_overload = len(matched_people) >= 2
            confirmed = tracker.is_confirmed(track_id)
            has_raw_overload = has_raw_overload or raw_overload
            has_confirmed_overload = has_confirmed_overload or confirmed
            frame_results.append(
                FrameVehicleResult(
                    track_id=track_id,
                    detection=vehicle,
                    matched_people=matched_people,
                    match_scores=match_scores[vehicle_idx],
                    raw_overload=raw_overload,
                    confirmed_overload=confirmed,
                )
            )
            rows.append(
                {
                    "frame": frame_index,
                    "timestamp_sec": f"{timestamp_sec:.3f}",
                    "vehicle_track_id": track_id,
                    "vehicle_class": vehicle.class_name,
                    "vehicle_conf": f"{vehicle.confidence:.3f}",
                    "vehicle_bbox": " ".join(f"{v:.1f}" for v in vehicle.bbox),
                    "matched_person_ids": " ".join(getattr(person, "stable_id", "") for person in matched_people),
                    "matched_person_count": len(matched_people),
                    "match_scores": " ".join(f"{score:.3f}" for score in match_scores[vehicle_idx]),
                    "raw_overload": int(raw_overload),
                    "confirmed_overload": int(confirmed),
                    "elapsed_ms": f"{elapsed_ms:.3f}",
                    "fps": f"{1000.0 / elapsed_ms:.3f}" if elapsed_ms > 0 else "",
                }
            )

        if has_raw_overload:
            frames_with_raw_overload += 1
        if has_confirmed_overload:
            frames_with_confirmed_overload += 1

        avg_ms = sum(timing_ms[-30:]) / min(len(timing_ms), 30)
        fps_text = f"fps={1000.0 / avg_ms:.1f}" if avg_ms > 0 else ""
        writer.write(draw_frame(frame, people, frame_results, frame_index, timestamp_sec, fps_text))

        processed += 1
        if processed % 50 == 0:
            print(f"[processed] {processed} frames")

    writer.release()
    cap.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "timestamp_sec",
            "vehicle_track_id",
            "vehicle_class",
            "vehicle_conf",
            "vehicle_bbox",
            "matched_person_ids",
            "matched_person_count",
            "match_scores",
            "raw_overload",
            "confirmed_overload",
            "elapsed_ms",
            "fps",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    avg_ms = sum(timing_ms) / len(timing_ms) if timing_ms else 0.0
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "model",
                "tracker",
                "processed_frames",
                "source_frames",
                "source_fps",
                "imgsz",
                "conf",
                "match_thresh",
                "confirm_frames",
                "mot_min_hits",
                "reid_weights",
                "reid_device",
                "reid_fp16",
                "total_people_detections",
                "total_two_wheeler_detections",
                "frames_with_raw_overload",
                "frames_with_confirmed_overload",
                "confirmed_overload_tracks",
                "avg_elapsed_ms",
                "avg_fps",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerow(
            {
                "video": str(video_path),
                "model": args.model,
                "tracker": tracker_name,
                "processed_frames": processed,
                "source_frames": total_frames,
                "source_fps": f"{source_fps:.3f}",
                "imgsz": args.imgsz,
                "conf": args.conf,
                "match_thresh": args.match_thresh,
                "confirm_frames": args.confirm_frames,
                "mot_min_hits": args.mot_min_hits,
                "reid_weights": args.reid_weights,
                "reid_device": args.reid_device,
                "reid_fp16": int(args.reid_fp16),
                "total_people_detections": total_people,
                "total_two_wheeler_detections": total_vehicles,
                "frames_with_raw_overload": frames_with_raw_overload,
                "frames_with_confirmed_overload": frames_with_confirmed_overload,
                "confirmed_overload_tracks": len(tracker.confirmed_track_ids),
                "avg_elapsed_ms": f"{avg_ms:.3f}",
                "avg_fps": f"{1000.0 / avg_ms:.3f}" if avg_ms > 0 else "",
            }
        )

    print(f"[video] {out_path}")
    print(f"[frame csv] {csv_path}")
    print(f"[summary csv] {summary_path}")
    print(f"[summary] source_frames={total_frames} processed_frames={processed} source_fps={source_fps:.3f} output_fps={output_fps:.3f}")


if __name__ == "__main__":
    main()
