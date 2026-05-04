#!/usr/bin/env python3
"""
Yolov8n-only overload validation on real video.

This script uses only COCO classes from yolov8n.pt:
- 0: person
- 1: bicycle
- 3: motorcycle

It treats bicycle/motorcycle as two-wheeler candidates and flags a possible
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
import torch
from ultralytics import YOLO


PERSON_CLASS_ID = 0
BICYCLE_CLASS_ID = 1
MOTORCYCLE_CLASS_ID = 3
TARGET_CLASSES = [PERSON_CLASS_ID, BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID]
CLASS_NAMES = {
    PERSON_CLASS_ID: "person",
    BICYCLE_CLASS_ID: "bicycle",
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
    track_id: int
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
    parser.add_argument("--track-iou", type=float, default=0.18)
    parser.add_argument("--max-missed", type=int, default=6)
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

    tracker = VehicleTracker(iou_thresh=args.track_iou, max_missed=args.max_missed, confirm_frames=args.confirm_frames)
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
        vehicles = [det for det in detections if det.class_id in {BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID}]
        total_people += len(people)
        total_vehicles += len(vehicles)

        grouped_people, match_scores = match_people_to_vehicles(people, vehicles, args.match_thresh)
        rider_counts = {idx: len(grouped_people[idx]) for idx in range(len(vehicles))}
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
