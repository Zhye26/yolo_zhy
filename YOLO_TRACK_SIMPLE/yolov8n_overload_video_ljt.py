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

    def update(self, vehicles: List[Detection], rider_counts: Dict[int, int]) -> Dict[int, int]:
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
            if count >= 2:
                track.positive_streak += 1
            else:
                track.positive_streak = max(0, track.positive_streak - 1)
            track.confirmed_overload = track.positive_streak >= self.confirm_frames
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

    def update(self, vehicles: List[Detection], rider_counts: Dict[int, int], frame) -> Dict[int, int]:
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
                    "positive_streak": 0,
                    "last_seen": self.frame_index,
                    "confirmed_overload": False,
                },
            )
            count = rider_counts.get(det_idx, 0)
            if count >= 2:
                state["positive_streak"] = int(state["positive_streak"]) + 1
            else:
                state["positive_streak"] = max(0, int(state["positive_streak"]) - 1)
            state["last_seen"] = self.frame_index
            state["confirmed_overload"] = int(state["positive_streak"]) >= self.confirm_frames
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
            state["positive_streak"] = max(0, int(state["positive_streak"]) - 1)
            state["confirmed_overload"] = False
        for track_id in expired:
            self.states.pop(track_id, None)

    def is_confirmed(self, track_id: int) -> bool:
        state = self.states.get(track_id)
        return bool(state and state["confirmed_overload"])


class AssociationByteTracker:
    def __init__(
        self,
        track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        confirm_frames: int,
        max_missed: int,
        association_min_hits: int,
        association_lock_frames: int,
        association_unbind_frames: int,
        association_switch_margin: float,
    ):
        try:
            from boxmot import BYTETracker
        except ImportError as exc:
            raise RuntimeError("Association ByteTrack requires boxmot. Install project tracking dependencies first.") from exc

        self.person_tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=30,
        )
        self.vehicle_tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=30,
        )
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.association_min_hits = association_min_hits
        self.association_lock_frames = association_lock_frames
        self.association_unbind_frames = association_unbind_frames
        self.association_switch_margin = association_switch_margin
        self.frame_index = 0
        self.confirmed_track_ids: set[str] = set()
        self.raw_to_stable: Dict[str, Dict[int, str]] = {"person": {}, "vehicle": {}}
        self.stable_objects: Dict[str, Dict[str, object]] = {}
        self.next_person_id = 1
        self.next_vehicle_id = 1
        self.bindings: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.vehicle_states: Dict[str, Dict[str, int | bool]] = {}

    def update_scene(
        self,
        people: List[Detection],
        vehicles: List[Detection],
        frame,
        match_thresh: float,
    ) -> Tuple[Dict[int, str], Dict[int, List[TrackedDetection]], Dict[int, List[float]]]:
        self.frame_index += 1
        tracked_people = self._track_objects("person", self.person_tracker, people, frame)
        tracked_vehicles = self._track_objects("vehicle", self.vehicle_tracker, vehicles, frame)
        vehicle_matches = {idx: obj.stable_id for idx, obj in tracked_vehicles.items()}
        for vehicle_idx in range(len(vehicles)):
            vehicle_matches.setdefault(vehicle_idx, f"U{vehicle_idx + 1:03d}")

        current_pairs = self._score_current_pairs(tracked_people, tracked_vehicles, match_thresh)
        self._update_bindings(current_pairs)

        grouped: Dict[int, List[TrackedDetection]] = {idx: [] for idx in range(len(vehicles))}
        scores: Dict[int, List[float]] = {idx: [] for idx in range(len(vehicles))}
        vehicle_idx_by_stable = {obj.stable_id: idx for idx, obj in tracked_vehicles.items()}
        person_by_stable = {obj.stable_id: obj for obj in tracked_people.values()}

        for (person_id, vehicle_id), binding in self.bindings.items():
            if binding["state"] not in {"bound", "locked"}:
                continue
            if vehicle_id not in vehicle_idx_by_stable or person_id not in person_by_stable:
                continue
            vehicle_idx = vehicle_idx_by_stable[vehicle_id]
            grouped[vehicle_idx].append(person_by_stable[person_id])
            scores[vehicle_idx].append(float(binding["last_score"]))

        self._update_vehicle_overload_states(grouped, vehicle_matches)
        return vehicle_matches, grouped, scores

    def _track_objects(self, kind: str, tracker, detections: List[Detection], frame) -> Dict[int, TrackedDetection]:
        if not detections:
            return {}
        det_array = np.array(
            [
                [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence, det.class_id]
                for det in detections
            ],
            dtype=np.float32,
        )
        output = tracker.update(det_array, frame)
        raw_matches = self._match_outputs_to_detections(output, detections)
        tracked: Dict[int, TrackedDetection] = {}
        for det_idx, raw_track_id in raw_matches.items():
            det = detections[det_idx]
            stable_id = self._stable_id_for(kind, raw_track_id, det)
            tracked[det_idx] = TrackedDetection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=det.bbox,
                raw_track_id=raw_track_id,
                stable_id=stable_id,
            )
        return tracked

    def _match_outputs_to_detections(self, tracks_output, detections: List[Detection]) -> Dict[int, int]:
        if tracks_output is None or len(tracks_output) == 0:
            return {}
        candidates: List[Tuple[float, int, int]] = []
        for row in tracks_output:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, track_id = row[:5]
            class_id = int(row[6]) if len(row) > 6 else None
            track_box = [float(x1), float(y1), float(x2), float(y2)]
            for det_idx, det in enumerate(detections):
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

    def _stable_id_for(self, kind: str, raw_track_id: int, det: Detection) -> str:
        mapped = self.raw_to_stable[kind].get(raw_track_id)
        if mapped:
            self._touch_stable(mapped, det)
            return mapped

        prefix = "P" if kind == "person" else "M"
        candidate_id = self._find_reusable_stable_id(kind, det)
        if candidate_id is None:
            if kind == "person":
                candidate_id = f"{prefix}{self.next_person_id:03d}"
                self.next_person_id += 1
            else:
                candidate_id = f"{prefix}{self.next_vehicle_id:03d}"
                self.next_vehicle_id += 1
        self.raw_to_stable[kind][raw_track_id] = candidate_id
        self._touch_stable(candidate_id, det, kind=kind)
        return candidate_id

    def _touch_stable(self, stable_id: str, det: Detection, kind: Optional[str] = None) -> None:
        prev = self.stable_objects.get(stable_id)
        center = det.center
        velocity = (0.0, 0.0)
        if prev is not None:
            px, py = prev["center"]
            velocity = (center[0] - float(px), center[1] - float(py))
        self.stable_objects[stable_id] = {
            "kind": kind or (prev["kind"] if prev else ""),
            "bbox": det.bbox,
            "center": center,
            "velocity": velocity,
            "last_seen": self.frame_index,
        }

    def _find_reusable_stable_id(self, kind: str, det: Detection) -> Optional[str]:
        best_score = 0.0
        best_id = None
        for stable_id, state in self.stable_objects.items():
            if state["kind"] != kind:
                continue
            missed = self.frame_index - int(state["last_seen"])
            if missed > self.max_missed + self.association_unbind_frames:
                continue
            old_box = state["bbox"]
            score = iou(old_box, det.bbox)
            if score > best_score:
                best_score = score
                best_id = stable_id
        return best_id if best_score >= 0.25 else None

    def _score_current_pairs(
        self,
        tracked_people: Dict[int, TrackedDetection],
        tracked_vehicles: Dict[int, TrackedDetection],
        match_thresh: float,
    ) -> Dict[Tuple[str, str], float]:
        candidates: List[Tuple[float, str, str]] = []
        for person in tracked_people.values():
            current_binding = self._active_vehicle_for_person(person.stable_id)
            for vehicle in tracked_vehicles.values():
                score = person_vehicle_match_score(person, vehicle)
                if score <= 0:
                    continue
                score += self._motion_score(person.stable_id, vehicle.stable_id)
                if (person.stable_id, vehicle.stable_id) in self.bindings:
                    score += 0.25
                if current_binding and current_binding != vehicle.stable_id:
                    current_score = float(self.bindings[(person.stable_id, current_binding)]["last_score"])
                    lock_until = int(self.bindings[(person.stable_id, current_binding)]["lock_until"])
                    if self.frame_index <= lock_until:
                        score -= 1.0
                    elif score < current_score + self.association_switch_margin:
                        score -= 0.5
                if score >= match_thresh:
                    candidates.append((score, person.stable_id, vehicle.stable_id))

        candidates.sort(reverse=True)
        used_people: set[str] = set()
        selected: Dict[Tuple[str, str], float] = {}
        for score, person_id, vehicle_id in candidates:
            if person_id in used_people:
                continue
            used_people.add(person_id)
            selected[(person_id, vehicle_id)] = score
        return selected

    def _motion_score(self, person_id: str, vehicle_id: str) -> float:
        person = self.stable_objects.get(person_id)
        vehicle = self.stable_objects.get(vehicle_id)
        if not person or not vehicle:
            return 0.0
        pv = person["velocity"]
        vv = vehicle["velocity"]
        pnorm = (float(pv[0]) ** 2 + float(pv[1]) ** 2) ** 0.5
        vnorm = (float(vv[0]) ** 2 + float(vv[1]) ** 2) ** 0.5
        if pnorm < 1.0 or vnorm < 1.0:
            return 0.0
        cosine = (float(pv[0]) * float(vv[0]) + float(pv[1]) * float(vv[1])) / (pnorm * vnorm)
        return max(0.0, cosine) * 0.15

    def _active_vehicle_for_person(self, person_id: str) -> Optional[str]:
        active = []
        for (bound_person_id, vehicle_id), binding in self.bindings.items():
            if bound_person_id != person_id:
                continue
            if binding["state"] in {"bound", "locked", "lost"}:
                active.append((int(binding["hits"]), vehicle_id))
        if not active:
            return None
        active.sort(reverse=True)
        return active[0][1]

    def _update_bindings(self, current_pairs: Dict[Tuple[str, str], float]) -> None:
        for key, score in current_pairs.items():
            binding = self.bindings.setdefault(
                key,
                {
                    "hits": 0,
                    "misses": 0,
                    "state": "candidate",
                    "lock_until": 0,
                    "last_score": 0.0,
                },
            )
            binding["hits"] = int(binding["hits"]) + 1
            binding["misses"] = 0
            binding["last_score"] = score
            if int(binding["hits"]) >= self.association_min_hits:
                binding["state"] = "locked" if self.frame_index <= int(binding["lock_until"]) else "bound"
                if int(binding["lock_until"]) < self.frame_index:
                    binding["lock_until"] = self.frame_index + self.association_lock_frames
                    binding["state"] = "locked"

        expired = []
        for key, binding in self.bindings.items():
            if key in current_pairs:
                continue
            binding["misses"] = int(binding["misses"]) + 1
            if binding["state"] in {"bound", "locked"}:
                binding["state"] = "lost"
            if int(binding["misses"]) > self.association_unbind_frames:
                expired.append(key)
        for key in expired:
            self.bindings.pop(key, None)

    def _update_vehicle_overload_states(
        self,
        grouped: Dict[int, List[TrackedDetection]],
        vehicle_matches: Dict[int, str],
    ) -> None:
        current_vehicle_ids = set(vehicle_matches.values())
        for vehicle_idx, stable_id in vehicle_matches.items():
            state = self.vehicle_states.setdefault(
                stable_id,
                {"positive_streak": 0, "last_seen": self.frame_index, "confirmed_overload": False},
            )
            count = len(grouped.get(vehicle_idx, []))
            if count >= 2:
                state["positive_streak"] = int(state["positive_streak"]) + 1
            else:
                state["positive_streak"] = max(0, int(state["positive_streak"]) - 1)
            state["last_seen"] = self.frame_index
            state["confirmed_overload"] = int(state["positive_streak"]) >= self.confirm_frames
            if state["confirmed_overload"]:
                self.confirmed_track_ids.add(stable_id)

        for stable_id, state in list(self.vehicle_states.items()):
            if stable_id in current_vehicle_ids:
                continue
            if self.frame_index - int(state["last_seen"]) > self.max_missed:
                self.vehicle_states.pop(stable_id, None)
            else:
                state["confirmed_overload"] = False

    def is_confirmed(self, track_id: int | str) -> bool:
        state = self.vehicle_states.get(str(track_id))
        return bool(state and state["confirmed_overload"])


def create_vehicle_tracker(args):
    if args.tracker == "association":
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

    if args.tracker in {"auto", "byte"}:
        try:
            tracker = ByteVehicleTracker(
                track_thresh=args.byte_track_thresh,
                match_thresh=args.byte_match_thresh,
                track_buffer=args.byte_track_buffer,
                confirm_frames=args.confirm_frames,
                max_missed=args.max_missed,
            )
            print("[tracker] byte")
            return tracker, "byte"
        except RuntimeError:
            if args.tracker == "byte":
                raise
            print("[tracker] byte unavailable, falling back to iou")

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
    parser.add_argument("--out", default="data/model_compare_video_vis/yolov8n-overload-ljt.mp4")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-thresh", type=float, default=1.05)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--tracker", choices=["auto", "byte", "iou", "association"], default="association")
    parser.add_argument("--track-iou", type=float, default=0.18)
    parser.add_argument("--max-missed", type=int, default=6)
    parser.add_argument("--byte-track-thresh", type=float, default=0.25)
    parser.add_argument("--byte-match-thresh", type=float, default=0.8)
    parser.add_argument("--byte-track-buffer", type=int, default=30)
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
    out_path = Path(args.out)
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

        if tracker_name == "association":
            track_matches, grouped_people, match_scores = tracker.update_scene(
                people,
                vehicles,
                frame,
                args.match_thresh,
            )
        else:
            grouped_people, match_scores = match_people_to_vehicles(people, vehicles, args.match_thresh)
            rider_counts = {idx: len(grouped_people[idx]) for idx in range(len(vehicles))}
            if tracker_name == "byte":
                track_matches = tracker.update(vehicles, rider_counts, frame)
            else:
                track_matches = tracker.update(vehicles, rider_counts)

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
