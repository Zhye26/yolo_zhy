#!/usr/bin/env python3
"""Benchmark trackers against manually corrected stable-region data."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/private/tmp/Ultralytics")

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.mot_trackers_ljt import AVAILABLE_TRACKERS  # noqa: E402
from core.pipeline.pipeline_ljt import (  # noqa: E402
    ASSOCIATION_MOT_TRACKERS,
    MOTORCYCLE_CLASS_ID,
    TARGET_CLASSES,
    create_vehicle_tracker,
    iou,
    match_people_to_vehicles,
    result_to_detections,
    sync_cuda_if_available,
    uses_association_scene,
)


@dataclass(frozen=True)
class Region:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class RowObject:
    frame: int
    track_id: str
    bbox: Tuple[float, float, float, float]
    raw_overload: bool
    suspected_overload: bool
    confirmed_overload: bool
    overload_status: str = "NORMAL"


@dataclass(frozen=True)
class Match:
    frame: int
    gt_id: str
    pred_id: str
    gt_overload: bool
    pred_overload: bool
    pred_suspected_overload: bool
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MOT trackers inside manually selected stable regions.")
    parser.add_argument("--standard-dir", type=Path, default=ROOT / "standardData")
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "yolov8n.pt")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["iou", "association", *ASSOCIATION_MOT_TRACKERS, *sorted(AVAILABLE_TRACKERS)],
        help="Trackers to run. Unavailable trackers are recorded as skipped.",
    )
    parser.add_argument("--out-detail", type=Path, default=ROOT / "standardData" / "benchmark_region_results_ljt.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "standardData" / "benchmark_region_summary_ljt.csv")
    parser.add_argument(
        "--out-track-stats",
        type=Path,
        default=ROOT / "standardData" / "benchmark_region_track_stats_ljt.csv",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--match-thresh", type=float, default=1.05)
    parser.add_argument("--bbox-iou-thresh", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit per video/tracker.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--track-iou", type=float, default=0.18)
    parser.add_argument("--max-missed", type=int, default=6)
    parser.add_argument("--byte-track-thresh", type=float, default=0.25)
    parser.add_argument("--byte-match-thresh", type=float, default=0.8)
    parser.add_argument("--byte-track-buffer", type=int, default=30)
    parser.add_argument("--mot-min-hits", type=int, default=3)
    parser.add_argument("--reid-weights", type=Path, default=ROOT / "weights" / "osnet_x0_25_msmt17.pt")
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--reid-fp16", action="store_true")
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--association-min-hits", type=int, default=4)
    parser.add_argument("--association-lock-frames", type=int, default=20)
    parser.add_argument("--association-unbind-frames", type=int, default=15)
    parser.add_argument("--association-switch-margin", type=float, default=0.35)
    return parser.parse_args()


def read_region(video_dir: Path) -> Region:
    candidates = sorted(video_dir.glob("*region_stats.csv"))
    if not candidates:
        raise FileNotFoundError(f"No region stats CSV found in {video_dir}")
    with candidates[0].open(newline="", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    vals = [float(item) for item in row["region_bbox"].split()]
    if len(vals) != 4:
        raise ValueError(f"Invalid region_bbox in {candidates[0]}: {row['region_bbox']}")
    return Region(*vals)


def read_standard_rows(video_dir: Path, region: Region) -> List[RowObject]:
    candidates = sorted(video_dir.glob("*_frames_id_fixed.csv"))
    if not candidates:
        raise FileNotFoundError(f"No manually fixed CSV found in {video_dir}")
    rows: List[RowObject] = []
    with candidates[0].open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            bbox = parse_bbox(row["vehicle_bbox"])
            if not center_in_region(bbox, region):
                continue
            final_id = (row.get("final_vehicle_id") or row.get("vehicle_track_id") or "").strip()
            if not final_id:
                continue
            rows.append(
                RowObject(
                    frame=int(row["frame"]),
                    track_id=final_id,
                    bbox=bbox,
                    raw_overload=as_bool(row.get("raw_overload")),
                    suspected_overload=as_bool(row.get("raw_overload")) or as_bool(row.get("confirmed_overload")),
                    confirmed_overload=as_bool(row.get("confirmed_overload")),
                    overload_status="CONFIRMED" if as_bool(row.get("confirmed_overload")) else "NORMAL",
                )
            )
    return rows


def parse_bbox(value: str) -> Tuple[float, float, float, float]:
    vals = [float(item) for item in value.split()]
    if len(vals) != 4:
        raise ValueError(f"Invalid bbox: {value}")
    return vals[0], vals[1], vals[2], vals[3]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def center_in_region(bbox: Sequence[float], region: Region) -> bool:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2


def discover_video_dirs(standard_dir: Path) -> List[Path]:
    return sorted(path for path in standard_dir.iterdir() if path.is_dir() and list(path.glob("*.mp4")))


def tracker_args(base: argparse.Namespace, tracker_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        tracker=tracker_name,
        track_iou=base.track_iou,
        max_missed=base.max_missed,
        byte_track_thresh=base.byte_track_thresh,
        byte_match_thresh=base.byte_match_thresh,
        byte_track_buffer=base.byte_track_buffer,
        mot_min_hits=base.mot_min_hits,
        reid_weights=base.reid_weights,
        reid_device=base.reid_device,
        reid_fp16=base.reid_fp16,
        confirm_frames=base.confirm_frames,
        association_min_hits=base.association_min_hits,
        association_lock_frames=base.association_lock_frames,
        association_unbind_frames=base.association_unbind_frames,
        association_switch_margin=base.association_switch_margin,
    )


def run_tracker_on_video(
    model: YOLO,
    video_path: Path,
    region: Region,
    tracker_name: str,
    args: argparse.Namespace,
) -> Tuple[str, List[RowObject], Dict[str, Dict[str, object]], float, int]:
    tracker, resolved_tracker = create_vehicle_tracker(tracker_args(args, tracker_name))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    for _ in range(args.warmup):
        model.predict(first_frame, classes=TARGET_CLASSES, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
    sync_cuda_if_available()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows: List[RowObject] = []
    processed = 0
    frame_index = -1
    start_wall = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % max(1, args.frame_stride) != 0:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        result = model.predict(
            frame,
            classes=TARGET_CLASSES,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        sync_cuda_if_available()
        detections = result_to_detections(result, min_area=args.min_area)
        people = [det for det in detections if det.class_id != MOTORCYCLE_CLASS_ID]
        vehicles = [det for det in detections if det.class_id == MOTORCYCLE_CLASS_ID]

        if uses_association_scene(resolved_tracker):
            track_matches, grouped_people, _ = tracker.update_scene(people, vehicles, frame, args.match_thresh)
        else:
            grouped_people, match_scores = match_people_to_vehicles(people, vehicles, args.match_thresh)
            rider_counts = {idx: len(grouped_people[idx]) for idx in range(len(vehicles))}
            if resolved_tracker in AVAILABLE_TRACKERS:
                track_matches = tracker.update(vehicles, rider_counts, frame, grouped_people, match_scores)
            else:
                track_matches = tracker.update(vehicles, rider_counts, grouped_people, match_scores)

        for vehicle_idx, vehicle in enumerate(vehicles):
            bbox = tuple(float(v) for v in vehicle.bbox)
            if not center_in_region(bbox, region):
                continue
            track_id = str(track_matches[vehicle_idx])
            matched_people = grouped_people[vehicle_idx]
            raw_overload = len(matched_people) >= 2
            if hasattr(tracker, "get_overload_status"):
                overload_status = str(tracker.get_overload_status(track_matches[vehicle_idx]))
                suspected_overload = overload_status in {"SUSPECTED", "CONFIRMED"}
                confirmed_overload = overload_status == "CONFIRMED"
            else:
                confirmed_overload = bool(tracker.is_confirmed(track_matches[vehicle_idx]))
                suspected_overload = raw_overload or confirmed_overload
                overload_status = "CONFIRMED" if confirmed_overload else "SUSPECTED" if raw_overload else "NORMAL"
            if track_id.startswith("U"):
                raw_overload = False
                suspected_overload = False
                confirmed_overload = False
                overload_status = "UNCERTAIN"
            rows.append(
                RowObject(
                    frame=frame_index,
                    track_id=track_id,
                    bbox=bbox,
                    raw_overload=raw_overload,
                    suspected_overload=suspected_overload,
                    confirmed_overload=confirmed_overload,
                    overload_status=overload_status,
                )
            )
        processed += 1

    cap.release()
    elapsed = max(1e-6, time.perf_counter() - start_wall)
    track_stats = tracker.get_track_stats() if hasattr(tracker, "get_track_stats") else {}
    return resolved_tracker, rows, track_stats, processed / elapsed, processed


def match_rows(gt_rows: Sequence[RowObject], pred_rows: Sequence[RowObject], iou_thresh: float) -> List[Match]:
    gt_by_frame = group_by_frame(gt_rows)
    pred_by_frame = group_by_frame(pred_rows)
    matches: List[Match] = []
    for frame in sorted(set(gt_by_frame) & set(pred_by_frame)):
        candidates: List[Tuple[float, int, int]] = []
        for gt_idx, gt in enumerate(gt_by_frame[frame]):
            for pred_idx, pred in enumerate(pred_by_frame[frame]):
                score = iou(list(gt.bbox), list(pred.bbox))
                if score >= iou_thresh:
                    candidates.append((score, gt_idx, pred_idx))
        candidates.sort(reverse=True)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for score, gt_idx, pred_idx in candidates:
            if gt_idx in used_gt or pred_idx in used_pred:
                continue
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
            gt = gt_by_frame[frame][gt_idx]
            pred = pred_by_frame[frame][pred_idx]
            matches.append(
                Match(
                    frame=frame,
                    gt_id=gt.track_id,
                    pred_id=pred.track_id,
                    gt_overload=gt.confirmed_overload,
                    pred_overload=pred.confirmed_overload,
                    pred_suspected_overload=pred.suspected_overload,
                    iou=score,
                )
            )
    return matches


def group_by_frame(rows: Sequence[RowObject]) -> Dict[int, List[RowObject]]:
    grouped: Dict[int, List[RowObject]] = defaultdict(list)
    for row in rows:
        grouped[row.frame].append(row)
    return grouped


def metric_row(
    video_name: str,
    tracker: str,
    status: str,
    gt_rows: Sequence[RowObject],
    pred_rows: Sequence[RowObject],
    matches: Sequence[Match],
    wall_fps: float = 0.0,
    processed_frames: int = 0,
    error: str = "",
) -> Dict[str, object]:
    gt_ids = {row.track_id for row in gt_rows}
    pred_ids = {row.track_id for row in pred_rows}
    gt_overload_ids = {row.track_id for row in gt_rows if row.confirmed_overload}
    pred_overload_ids = {row.track_id for row in pred_rows if row.confirmed_overload and not row.track_id.startswith("U")}
    matched_gt_ids = {match.gt_id for match in matches}
    track_count_ratio = safe_div(len(pred_ids), len(gt_ids))
    track_overcount = len(pred_ids) - len(gt_ids)
    overload_count_ratio = safe_div(len(pred_overload_ids), len(gt_overload_ids))
    overload_track_overcount = len(pred_overload_ids) - len(gt_overload_ids)
    detection_precision = safe_div(len(matches), len(pred_rows))
    detection_recall = safe_div(len(matches), len(gt_rows))
    vehicle_coverage = safe_div(len(matched_gt_ids), len(gt_ids))
    id_consistency = compute_id_consistency(matches)
    id_switches, avg_fragments = compute_fragmentation(matches)
    overload_precision, overload_recall, overload_f1 = compute_overload_metrics(matches)
    suspected_precision, suspected_recall, suspected_f1 = compute_overload_metrics(matches, use_suspected=True)
    avg_iou = mean([match.iou for match in matches]) if matches else 0.0
    return {
        "video": video_name,
        "tracker": tracker,
        "status": status,
        "gt_rows": len(gt_rows),
        "pred_rows": len(pred_rows),
        "matched_rows": len(matches),
        "detection_precision": fmt(detection_precision),
        "detection_recall": fmt(detection_recall),
        "avg_match_iou": fmt(avg_iou),
        "gt_vehicle_count": len(gt_ids),
        "pred_track_count": len(pred_ids),
        "track_count_ratio": fmt(track_count_ratio),
        "track_overcount": track_overcount,
        "matched_gt_vehicle_count": len(matched_gt_ids),
        "vehicle_coverage": fmt(vehicle_coverage),
        "gt_overload_vehicle_count": len(gt_overload_ids),
        "pred_overload_track_count": len(pred_overload_ids),
        "overload_count_ratio": fmt(overload_count_ratio),
        "overload_track_overcount": overload_track_overcount,
        "id_consistency": fmt(id_consistency),
        "id_switches": id_switches,
        "avg_fragments_per_gt": fmt(avg_fragments),
        "overload_precision": fmt(overload_precision),
        "overload_recall": fmt(overload_recall),
        "overload_f1": fmt(overload_f1),
        "suspected_overload_precision": fmt(suspected_precision),
        "suspected_overload_recall": fmt(suspected_recall),
        "suspected_overload_f1": fmt(suspected_f1),
        "processed_frames": processed_frames,
        "wall_fps": fmt(wall_fps),
        "error": error,
    }


def compute_id_consistency(matches: Sequence[Match]) -> float:
    if not matches:
        return 0.0
    pred_to_gt_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for match in matches:
        pred_to_gt_counts[match.pred_id][match.gt_id] += 1
    pred_major_gt = {pred_id: counts.most_common(1)[0][0] for pred_id, counts in pred_to_gt_counts.items()}
    consistent = sum(1 for match in matches if pred_major_gt.get(match.pred_id) == match.gt_id)
    return consistent / len(matches)


def compute_fragmentation(matches: Sequence[Match]) -> Tuple[int, float]:
    if not matches:
        return 0, 0.0
    by_gt: Dict[str, List[Match]] = defaultdict(list)
    for match in matches:
        by_gt[match.gt_id].append(match)
    switches = 0
    fragments: List[int] = []
    for gt_matches in by_gt.values():
        gt_matches.sort(key=lambda item: item.frame)
        pred_ids = [item.pred_id for item in gt_matches]
        fragments.append(len(set(pred_ids)))
        prev_id = pred_ids[0]
        for pred_id in pred_ids[1:]:
            if pred_id != prev_id:
                switches += 1
                prev_id = pred_id
    return switches, mean(fragments) if fragments else 0.0


def compute_overload_metrics(matches: Sequence[Match], use_suspected: bool = False) -> Tuple[float, float, float]:
    def pred(match: Match) -> bool:
        return match.pred_suspected_overload if use_suspected else match.pred_overload

    tp = sum(1 for match in matches if pred(match) and match.gt_overload)
    fp = sum(1 for match in matches if pred(match) and not match.gt_overload)
    fn = sum(1 for match in matches if not pred(match) and match.gt_overload)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def build_track_stat_rows(
    video_name: str,
    tracker: str,
    pred_rows: Sequence[RowObject],
    matches: Sequence[Match],
    tracker_stats: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    pred_rows_by_id: Dict[str, List[RowObject]] = defaultdict(list)
    for row in pred_rows:
        pred_rows_by_id[row.track_id].append(row)

    matches_by_pred: Dict[str, List[Match]] = defaultdict(list)
    for match in matches:
        matches_by_pred[match.pred_id].append(match)

    rows: List[Dict[str, object]] = []
    for pred_id in sorted(pred_rows_by_id):
        pred_track_rows = pred_rows_by_id[pred_id]
        pred_matches = matches_by_pred.get(pred_id, [])
        gt_counts = Counter(match.gt_id for match in pred_matches)
        matched_gt_id = gt_counts.most_common(1)[0][0] if gt_counts else ""
        gt_overload = any(match.gt_overload for match in pred_matches if not matched_gt_id or match.gt_id == matched_gt_id)
        observed = len(pred_track_rows)
        raw_frames = sum(1 for row in pred_track_rows if row.raw_overload)
        suspected_frames = sum(1 for row in pred_track_rows if row.suspected_overload)
        confirmed_frames = sum(1 for row in pred_track_rows if row.confirmed_overload)
        stats = tracker_stats.get(pred_id, {})
        stat_observed = int(stats.get("observed_frames", observed) or observed)
        final_suspected = not pred_id.startswith("U") and any(row.suspected_overload for row in pred_track_rows)
        final_confirmed = not pred_id.startswith("U") and any(row.confirmed_overload for row in pred_track_rows)
        if pred_id.startswith("U"):
            status = "UNCERTAIN"
        elif final_confirmed:
            status = "CONFIRMED"
        elif final_suspected:
            status = "SUSPECTED"
        else:
            status = str(stats.get("overload_status") or pred_track_rows[-1].overload_status)
        row = {
            "video": video_name,
            "tracker": tracker,
            "track_id": pred_id,
            "is_uncertain_id": int(pred_id.startswith("U")),
            "matched_gt_id": matched_gt_id,
            "gt_overload": int(gt_overload),
            "matched_rows": len(pred_matches),
            "observed_frames": stat_observed,
            "raw_overload_frames": int(stats.get("raw_overload_frames", raw_frames) or raw_frames),
            "suspected_overload_frames": suspected_frames,
            "confirmed_overload_frames": confirmed_frames,
            "raw_overload_ratio": fmt(float(stats.get("raw_overload_ratio", safe_div(raw_frames, observed)) or 0.0)),
            "high_conf_observed_frames": int(stats.get("high_conf_observed_frames", 0) or 0),
            "high_conf_overload_frames": int(stats.get("high_conf_overload_frames", 0) or 0),
            "high_conf_overload_ratio": fmt(float(stats.get("high_conf_overload_ratio", 0.0) or 0.0)),
            "moving_observed_frames": int(stats.get("moving_observed_frames", 0) or 0),
            "moving_overload_frames": int(stats.get("moving_overload_frames", 0) or 0),
            "moving_overload_ratio": fmt(float(stats.get("moving_overload_ratio", 0.0) or 0.0)),
            "max_rider_count": int(stats.get("max_rider_count", 0) or 0),
            "avg_vehicle_conf": fmt(float(stats.get("avg_vehicle_conf", 0.0) or 0.0)),
            "avg_match_score": fmt(float(stats.get("avg_match_score", 0.0) or 0.0)),
            "avg_speed_px": fmt(float(stats.get("avg_speed_px", 0.0) or 0.0)),
            "final_overload_status": status,
            "final_suspected_overload": int(final_suspected),
            "final_confirmed_overload": int(final_confirmed),
        }
        rows.append(row)
    return rows


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            grouped[str(row["tracker"])].append(row)
    summary: List[Dict[str, object]] = []
    for tracker, tracker_rows in sorted(grouped.items()):
        totals = {
            "gt_rows": sum(int(row["gt_rows"]) for row in tracker_rows),
            "pred_rows": sum(int(row["pred_rows"]) for row in tracker_rows),
            "matched_rows": sum(int(row["matched_rows"]) for row in tracker_rows),
            "gt_vehicle_count": sum(int(row["gt_vehicle_count"]) for row in tracker_rows),
            "pred_track_count": sum(int(row["pred_track_count"]) for row in tracker_rows),
            "matched_gt_vehicle_count": sum(int(row["matched_gt_vehicle_count"]) for row in tracker_rows),
            "gt_overload_vehicle_count": sum(int(row["gt_overload_vehicle_count"]) for row in tracker_rows),
            "pred_overload_track_count": sum(int(row["pred_overload_track_count"]) for row in tracker_rows),
            "id_switches": sum(int(row["id_switches"]) for row in tracker_rows),
        }
        summary.append(
            {
                "tracker": tracker,
                "videos": len(tracker_rows),
                "gt_rows": totals["gt_rows"],
                "pred_rows": totals["pred_rows"],
                "matched_rows": totals["matched_rows"],
                "detection_precision": fmt(safe_div(totals["matched_rows"], totals["pred_rows"])),
                "detection_recall": fmt(safe_div(totals["matched_rows"], totals["gt_rows"])),
                "gt_vehicle_count": totals["gt_vehicle_count"],
                "pred_track_count": totals["pred_track_count"],
                "track_count_ratio": fmt(safe_div(totals["pred_track_count"], totals["gt_vehicle_count"])),
                "track_overcount": totals["pred_track_count"] - totals["gt_vehicle_count"],
                "vehicle_coverage": fmt(safe_div(totals["matched_gt_vehicle_count"], totals["gt_vehicle_count"])),
                "gt_overload_vehicle_count": totals["gt_overload_vehicle_count"],
                "pred_overload_track_count": totals["pred_overload_track_count"],
                "overload_count_ratio": fmt(
                    safe_div(totals["pred_overload_track_count"], totals["gt_overload_vehicle_count"])
                ),
                "overload_track_overcount": totals["pred_overload_track_count"] - totals["gt_overload_vehicle_count"],
                "mean_id_consistency": fmt(mean_float(tracker_rows, "id_consistency")),
                "id_switches": totals["id_switches"],
                "mean_avg_fragments_per_gt": fmt(mean_float(tracker_rows, "avg_fragments_per_gt")),
                "mean_overload_f1": fmt(mean_float(tracker_rows, "overload_f1")),
                "mean_suspected_overload_f1": fmt(mean_float(tracker_rows, "suspected_overload_f1")),
                "mean_wall_fps": fmt(mean_float(tracker_rows, "wall_fps")),
            }
        )
    return summary


def mean_float(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if str(row.get(key, "")) not in {"", "nan"}]
    return mean(values) if values else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def fmt(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "0.000000"
    return f"{value:.6f}"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    video_dirs = discover_video_dirs(args.standard_dir)
    if not video_dirs:
        raise SystemExit(f"No video directories found under {args.standard_dir}")

    print(f"[load] model={args.model}")
    model = YOLO(str(args.model))

    detail_rows: List[Dict[str, object]] = []
    track_stat_rows: List[Dict[str, object]] = []
    for video_dir in video_dirs:
        video_path = next(iter(sorted(video_dir.glob("*.mp4"))))
        region = read_region(video_dir)
        gt_rows = read_standard_rows(video_dir, region)
        print(f"[video] {video_dir.name} gt_rows={len(gt_rows)} region={region}")
        for tracker_name in args.trackers:
            print(f"[run] video={video_dir.name} tracker={tracker_name}")
            try:
                resolved_tracker, pred_rows, tracker_stats, wall_fps, processed = run_tracker_on_video(
                    model, video_path, region, tracker_name, args
                )
                matches = match_rows(gt_rows, pred_rows, args.bbox_iou_thresh)
                track_stat_rows.extend(
                    build_track_stat_rows(video_dir.name, resolved_tracker, pred_rows, matches, tracker_stats)
                )
                row = metric_row(
                    video_dir.name,
                    resolved_tracker,
                    "ok",
                    gt_rows,
                    pred_rows,
                    matches,
                    wall_fps=wall_fps,
                    processed_frames=processed,
                )
            except Exception as exc:  # Keep one tracker failure from stopping the whole benchmark.
                row = metric_row(video_dir.name, tracker_name, "skipped", gt_rows, [], [], error=str(exc))
                print(f"[skip] video={video_dir.name} tracker={tracker_name}: {exc}")
            detail_rows.append(row)
            write_csv(args.out_detail, detail_rows)
            write_csv(args.out_track_stats, track_stat_rows)

    write_csv(args.out_summary, aggregate_rows(detail_rows))
    write_csv(args.out_track_stats, track_stat_rows)
    print(f"[done] detail={args.out_detail}")
    print(f"[done] track_stats={args.out_track_stats}")
    print(f"[done] summary={args.out_summary}")


if __name__ == "__main__":
    main()
