#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.gui.gui_wjh_ljt import (
    HEAD_CLASS_ID,
    HELMET_CLASS_ID,
    RIDER_CLASS_ID,
    TARGET_CLASSES as WJH_CLASSES,
    build_person_evidence_for_rider,
)
from core.pipeline.pipeline_ljt import (
    MOTORCYCLE_CLASS_ID,
    PERSON_CLASS_ID,
    Detection,
    iou as bbox_iou,
    match_people_to_vehicles,
    result_to_detections,
    sync_cuda_if_available,
)

YOLOV8N_CLASSES = [PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID]


class IouIdTracker:
    def __init__(self, iou_thresh: float = 0.25, max_missed: int = 10, prefix: str = "T"):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.prefix = prefix
        self.next_id = 1
        self.tracks: Dict[str, Dict[str, object]] = {}

    def update(self, detections: List[Detection]) -> Dict[int, str]:
        matches: Dict[int, str] = {}
        used_tracks: set[str] = set()
        candidates: List[tuple[float, str, int]] = []
        for track_id, state in self.tracks.items():
            if int(state["missed"]) > self.max_missed:
                continue
            for det_idx, det in enumerate(detections):
                score = bbox_iou(state["bbox"], det.bbox)
                if score >= self.iou_thresh:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True)

        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in matches:
                continue
            used_tracks.add(track_id)
            matches[det_idx] = track_id

        for track_id, state in list(self.tracks.items()):
            if track_id not in used_tracks:
                state["missed"] = int(state["missed"]) + 1

        for det_idx, det in enumerate(detections):
            track_id = matches.get(det_idx)
            if track_id is None:
                track_id = f"{self.prefix}{self.next_id:03d}"
                self.next_id += 1
                matches[det_idx] = track_id
            self.tracks[track_id] = {"bbox": det.bbox, "missed": 0, "confidence": det.confidence}
        return matches


def read_region_bbox(region_csv: Path) -> Tuple[int, int, int, int]:
    with region_csv.open("r", newline="", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    vals = [float(x) for x in str(row["region_bbox"]).split()]
    return int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])


def center_in_roi(det: Detection, roi_rect: Tuple[int, int, int, int] | None) -> bool:
    if roi_rect is None:
        return True
    x1, y1, x2, y2 = roi_rect
    cx, cy = det.center
    return x1 <= cx <= x2 and y1 <= cy <= y2


def match_cross_targets(wjh_items, yolo_items, cross_iou_thresh: float):
    candidates = []
    for w_idx, w_item in enumerate(wjh_items):
        if not w_item["in_roi"]:
            continue
        for y_idx, y_item in enumerate(yolo_items):
            if not y_item["in_roi"]:
                continue
            score = bbox_iou(w_item["det"].bbox, y_item["det"].bbox)
            if score >= cross_iou_thresh:
                candidates.append((score, w_idx, y_idx))
    candidates.sort(reverse=True)

    matched = []
    used_w: set[int] = set()
    used_y: set[int] = set()
    for score, w_idx, y_idx in candidates:
        if w_idx in used_w or y_idx in used_y:
            continue
        used_w.add(w_idx)
        used_y.add(y_idx)
        matched.append((wjh_items[w_idx], yolo_items[y_idx], score))
    return matched


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Headless cross-validate runner")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--wjh-model", type=Path, default=root / "weights" / "wjh.pt")
    parser.add_argument("--yolo-model", type=Path, default=root / "weights" / "yolov8n.pt")
    parser.add_argument("--region-csv", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--head-conf", type=float, default=0.25)
    parser.add_argument("--helmet-conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--wjh-min-area", type=float, default=12.0)
    parser.add_argument("--yolo-min-area", type=float, default=20.0)
    parser.add_argument("--yolo-match-thresh", type=float, default=1.05)
    parser.add_argument("--cross-iou-thresh", type=float, default=0.20)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video
    roi_rect = read_region_bbox(args.region_csv)

    wjh_model = YOLO(str(args.wjh_model))
    yolo_model = YOLO(str(args.yolo_model))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = video.parent
    stem = f"{video.stem}-cross-validate-ljt"
    csv_path = out_dir / f"{stem}_frames.csv"
    target_csv_path = out_dir / f"{stem}_targets.csv"

    rows: List[Dict[str, object]] = []
    target_rows: List[Dict[str, object]] = []
    frame_index = -1
    processed = 0

    wjh_tracker = IouIdTracker(iou_thresh=0.25, max_missed=10, prefix="W")
    yolo_tracker = IouIdTracker(iou_thresh=0.25, max_missed=10, prefix="Y")
    target_states: Dict[str, Dict[str, object]] = {}

    start_wall = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % max(1, args.frame_stride) != 0:
            continue

        sync_cuda_if_available()
        wjh_result = wjh_model.predict(frame, classes=WJH_CLASSES, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        yolo_result = yolo_model.predict(frame, classes=YOLOV8N_CLASSES, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        sync_cuda_if_available()

        wjh_dets = result_to_detections(wjh_result, min_area=args.wjh_min_area)
        yolo_dets = result_to_detections(yolo_result, min_area=args.yolo_min_area)

        riders_all = [det for det in wjh_dets if det.class_id == RIDER_CLASS_ID]
        wjh_ids = wjh_tracker.update(riders_all)
        heads = [det for det in wjh_dets if det.class_id == HEAD_CLASS_ID and det.confidence >= args.head_conf]
        helmets = [det for det in wjh_dets if det.class_id == HELMET_CLASS_ID and det.confidence >= args.helmet_conf]

        wjh_items = []
        for rider_idx, rider in enumerate(riders_all):
            people, stats = build_person_evidence_for_rider(rider, heads, helmets)
            in_roi = center_in_roi(rider, roi_rect)
            overload = in_roi and int(stats["person_evidence_count"]) >= 2
            helmet_status = "NO_HELMET" if int(stats["unpaired_head_count"]) > 0 else "HELMETED" if int(stats["helmet_count"]) > 0 else "UNKNOWN"
            wjh_items.append({
                "det": rider,
                "id": wjh_ids[rider_idx],
                "people": people,
                "stats": stats,
                "overload": overload,
                "helmet_status": helmet_status,
                "in_roi": in_roi,
            })

        people = [det for det in yolo_dets if det.class_id == PERSON_CLASS_ID]
        motorcycles_all = [det for det in yolo_dets if det.class_id == MOTORCYCLE_CLASS_ID]
        yolo_ids = yolo_tracker.update(motorcycles_all)
        motorcycles = [det for det in motorcycles_all if center_in_roi(det, roi_rect)]
        grouped_people, _ = match_people_to_vehicles(people, motorcycles, args.yolo_match_thresh)

        yolo_items = []
        for moto_all_idx, motorcycle in enumerate(motorcycles_all):
            in_roi = center_in_roi(motorcycle, roi_rect)
            roi_idx = motorcycles.index(motorcycle) if motorcycle in motorcycles else None
            matched = grouped_people[roi_idx] if roi_idx is not None else []
            yolo_items.append(
                {
                    "det": motorcycle,
                    "id": yolo_ids[moto_all_idx],
                    "people": matched,
                    "overload": in_roi and len(matched) >= 2,
                    "in_roi": in_roi,
                }
            )

        target_matches = match_cross_targets(wjh_items, yolo_items, args.cross_iou_thresh)
        final_items = []
        for w_item, y_item, cross_iou in target_matches:
            target_id = f"{w_item['id']}|{y_item['id']}"
            final_now = bool(w_item["overload"] and y_item["overload"])
            state = target_states.setdefault(
                target_id,
                {
                    "ever_overload": False,
                    "helmet_status": "UNKNOWN",
                },
            )
            if final_now:
                state["ever_overload"] = True
            if w_item["helmet_status"] != "UNKNOWN":
                state["helmet_status"] = w_item["helmet_status"]
            final_items.append((target_id, w_item, y_item, cross_iou, final_now, state))
            target_rows.append(
                {
                    "frame": frame_index,
                    "target_id": target_id,
                    "wjh_id": w_item["id"],
                    "yolo_id": y_item["id"],
                    "cross_iou": f"{cross_iou:.4f}",
                    "wjh_overload": int(w_item["overload"]),
                    "yolov8n_overload": int(y_item["overload"]),
                    "final_overload": int(final_now),
                    "ever_final_overload": int(bool(state["ever_overload"])),
                    "helmet_status": state["helmet_status"],
                    "wjh_person_evidence_count": int(w_item["stats"]["person_evidence_count"]),
                    "wjh_no_helmet_count": int(w_item["stats"]["unpaired_head_count"]),
                    "yolov8n_person_count": len(y_item["people"]),
                }
            )

        rows.append(
            {
                "frame": frame_index,
                "wjh_overload_count": sum(1 for item in wjh_items if item["overload"]),
                "yolov8n_overload_count": sum(1 for item in yolo_items if item["overload"]),
                "matched_target_count": len(final_items),
                "final_overload": int(any(item[4] for item in final_items)),
                "final_target_ids": " ".join(item[0] for item in final_items if item[4]),
                "helmet_statuses": " ".join(f"{item[0]}:{item[5]['helmet_status']}" for item in final_items),
                "roi": " ".join(str(v) for v in roi_rect),
            }
        )

        processed += 1
        if processed % 50 == 0:
            wall_fps = processed / max(1e-6, (time.perf_counter() - start_wall))
            print(f"[cross] frame={frame_index}/{total_frames} processed={processed} wall_fps={wall_fps:.2f}")

    cap.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "wjh_overload_count",
                "yolov8n_overload_count",
                "matched_target_count",
                "final_overload",
                "final_target_ids",
                "helmet_statuses",
                "roi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with target_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "target_id",
                "wjh_id",
                "yolo_id",
                "cross_iou",
                "wjh_overload",
                "yolov8n_overload",
                "final_overload",
                "ever_final_overload",
                "helmet_status",
                "wjh_person_evidence_count",
                "wjh_no_helmet_count",
                "yolov8n_person_count",
            ],
        )
        writer.writeheader()
        writer.writerows(target_rows)

    elapsed = max(1e-6, time.perf_counter() - start_wall)
    print(f"[done] cross rows={len(rows)} targets={len(target_rows)} wall_fps={processed/elapsed:.3f}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {target_csv_path}")


if __name__ == "__main__":
    main()
