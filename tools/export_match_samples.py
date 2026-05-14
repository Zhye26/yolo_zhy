#!/usr/bin/env python3
"""
Export person-to-vehicle association samples for manual correction.

The output CSV is intentionally editable: fill `match_label` with 1 when the
person belongs to the vehicle, 0 otherwise, and optionally fill `error_type`
and `notes` for later analysis.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.rules.match_features import FEATURE_NAMES
except ModuleNotFoundError:
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


BASE_COLUMNS = [
    "video",
    "frame_index",
    "timestamp",
    "vehicle_track_id",
    "person_track_id",
    "vehicle_bbox",
    "person_bbox",
    "vehicle_conf",
    "person_conf",
    "heuristic_match",
    "correct_vehicle_id",
    "match_label",
    "error_type",
    "notes",
]


def export_samples(args) -> int:
    # Author: You Pinzhen - export candidate person-vehicle pairs for manual correction CSVs.
    import cv2

    from ultralytics import YOLO
    from yolov8n_overload_video_ljt import (
        BICYCLE_CLASS_ID,
        MOTORCYCLE_CLASS_ID,
        PERSON_CLASS_ID,
        TARGET_CLASSES,
        VehicleTracker,
        match_people_to_vehicles,
        result_to_detections,
        sync_cuda_if_available,
    )
    extract_match_features = _load_extract_match_features()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    tracker = VehicleTracker(
        iou_thresh=args.track_iou,
        max_missed=args.max_missed,
        confirm_frames=args.confirm_frames,
    )

    fieldnames = BASE_COLUMNS + FEATURE_NAMES
    frame_index = -1
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            if args.max_frames and frame_index > args.max_frames:
                break
            if frame_index % args.sample_every != 0:
                # Keep tracker reasonably warm even on skipped frames.
                _update_tracker_on_frame(
                    model=model,
                    tracker=tracker,
                    frame=frame,
                    args=args,
                    export_pairs=False,
                    writer=None,
                    fps=fps,
                    frame_index=frame_index,
                    video_path=args.video,
                    rows_written=rows_written,
                )
                continue

            rows_written = _update_tracker_on_frame(
                model=model,
                tracker=tracker,
                frame=frame,
                args=args,
                export_pairs=True,
                writer=writer,
                fps=fps,
                frame_index=frame_index,
                video_path=args.video,
                rows_written=rows_written,
                extract_match_features=extract_match_features,
                vehicle_class_ids={BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID},
                person_class_id=PERSON_CLASS_ID,
                target_classes=TARGET_CLASSES,
                match_people_to_vehicles=match_people_to_vehicles,
                result_to_detections=result_to_detections,
                sync_cuda_if_available=sync_cuda_if_available,
            )

            if args.progress_every and frame_index % args.progress_every == 0:
                print(f"Processed {frame_index} frames, wrote {rows_written} rows", end="\r")

    cap.release()
    print(f"\nWrote {rows_written} candidate pairs to {output_path}")
    return 0


def _update_tracker_on_frame(
    model,
    tracker,
    frame,
    args,
    export_pairs: bool,
    writer,
    fps: float,
    frame_index: int,
    video_path: str,
    rows_written: int,
    extract_match_features=None,
    vehicle_class_ids=None,
    person_class_id=None,
    target_classes=None,
    match_people_to_vehicles=None,
    result_to_detections=None,
    sync_cuda_if_available=None,
) -> int:
    if target_classes is None:
        from yolov8n_overload_video_ljt import (
            BICYCLE_CLASS_ID,
            MOTORCYCLE_CLASS_ID,
            PERSON_CLASS_ID,
            TARGET_CLASSES,
            match_people_to_vehicles,
            result_to_detections,
            sync_cuda_if_available,
        )

        vehicle_class_ids = {BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID}
        person_class_id = PERSON_CLASS_ID
        target_classes = TARGET_CLASSES

    sync_cuda_if_available()
    result = model.predict(
        frame,
        classes=target_classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )[0]
    sync_cuda_if_available()

    detections = result_to_detections(result, min_area=args.min_area)
    people = [det for det in detections if det.class_id == person_class_id]
    vehicles = [det for det in detections if det.class_id in vehicle_class_ids]
    grouped_people, _ = match_people_to_vehicles(people, vehicles, args.heuristic_threshold)
    rider_counts = {idx: len(grouped_people[idx]) for idx in range(len(vehicles))}
    track_matches = tracker.update(vehicles, rider_counts)

    if not export_pairs or writer is None:
        return rows_written

    for vehicle_index, vehicle in enumerate(vehicles):
        for person_index, person in enumerate(people):
            features = extract_match_features(
                person,
                vehicle,
                frame_width=frame.shape[1],
            )
            row = {
                "video": video_path,
                "frame_index": frame_index,
                "timestamp": f"{frame_index / fps:.3f}",
                "vehicle_track_id": track_matches.get(vehicle_index, ""),
                "person_track_id": -(person_index + 1),
                "vehicle_bbox": json.dumps(_round_bbox(vehicle.bbox), ensure_ascii=False),
                "person_bbox": json.dumps(_round_bbox(person.bbox), ensure_ascii=False),
                "vehicle_conf": f"{vehicle.confidence:.6f}",
                "person_conf": f"{person.confidence:.6f}",
                "heuristic_match": 1 if features.values["legacy_match_score"] >= args.heuristic_threshold else 0,
                "correct_vehicle_id": "",
                "match_label": "",
                "error_type": "",
                "notes": "",
            }
            row.update({name: f"{features.values[name]:.8f}" for name in FEATURE_NAMES})
            writer.writerow(row)
            rows_written += 1
    return rows_written


def _round_bbox(bbox: List[float]) -> List[float]:
    return [round(float(value), 2) for value in bbox]


def _load_extract_match_features():
    module_path = Path(__file__).resolve().parents[1] / "app" / "rules" / "match_features.py"
    spec = importlib.util.spec_from_file_location("match_features_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load match feature module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.extract_match_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Export person-vehicle match samples for manual labeling")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--model", default="models/bifpn_best.pt", help="YOLO model path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--task", choices=["passenger", "all"], default="passenger", help="Kept for CLI compatibility")
    parser.add_argument("--sample-every", type=int, default=5, help="Export one frame every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, 0 means all frames")
    parser.add_argument("--heuristic-threshold", type=float, default=0.16, help="Legacy heuristic label hint threshold")
    parser.add_argument("--progress-every", type=int, default=60, help="Print progress every N frames, 0 disables")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    parser.add_argument("--track-iou", type=float, default=0.18, help="Vehicle tracker IoU threshold")
    parser.add_argument("--max-missed", type=int, default=6, help="Vehicle tracker max missed frames")
    parser.add_argument("--confirm-frames", type=int, default=2, help="Tracker overload confirmation frames")
    parser.add_argument("--min-area", type=float, default=20.0, help="Minimum detection area")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video does not exist: {args.video}")
        return 1
    return export_samples(args)


if __name__ == "__main__":
    raise SystemExit(main())
