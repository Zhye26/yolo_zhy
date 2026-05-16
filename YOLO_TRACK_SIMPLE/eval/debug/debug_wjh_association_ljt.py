#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.gui.gui_wjh_ljt import (  # noqa: E402
    HEAD_CLASS_ID,
    HELMET_CLASS_ID,
    RIDER_CLASS_ID,
    TARGET_CLASSES,
    WjhAssociationGuiLjt,
    _center,
    build_head_helmet_evidence,
    occupant_rider_match_score,
)
from core.pipeline.pipeline_ljt import iou, overlap_ratio, point_in_box, result_to_detections  # noqa: E402


def _fmt_box(box: List[float]) -> str:
    return " ".join(f"{v:.1f}" for v in box)


def _class_name(class_id: int) -> str:
    if class_id == HEAD_CLASS_ID:
        return "head"
    if class_id == HELMET_CLASS_ID:
        return "helmet"
    if class_id == RIDER_CLASS_ID:
        return "rider"
    return str(class_id)


def _rounded_box_key(box: List[float]) -> Tuple[int, int, int, int]:
    return tuple(round(v) for v in box)


def _new_tracker(args):
    dummy = SimpleNamespace(
        byte_track_thresh=SimpleNamespace(get=lambda: args.byte_track_thresh),
        byte_match_thresh=SimpleNamespace(get=lambda: args.byte_match_thresh),
        byte_track_buffer=SimpleNamespace(get=lambda: args.byte_track_buffer),
        confirm_frames=SimpleNamespace(get=lambda: args.confirm_frames),
        max_missed=SimpleNamespace(get=lambda: args.max_missed),
        association_min_hits=SimpleNamespace(get=lambda: args.association_min_hits),
        association_lock_frames=SimpleNamespace(get=lambda: args.association_lock_frames),
        association_unbind_frames=SimpleNamespace(get=lambda: args.association_unbind_frames),
        association_switch_margin=SimpleNamespace(get=lambda: args.association_switch_margin),
    )
    return WjhAssociationGuiLjt._new_tracker(dummy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug wjh.pt rider/head/helmet association geometry.")
    parser.add_argument("--video", default="test.mp4")
    parser.add_argument("--model", default="weights/wjh.pt")
    parser.add_argument("--start", type=int, default=1488)
    parser.add_argument("--end", type=int, default=1501)
    parser.add_argument("--track-id", default="")
    parser.add_argument("--out", default="runs/wjh_debug/wjh_association_iou_debug.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--head-conf", type=float, default=0.25)
    parser.add_argument("--helmet-conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-area", type=float, default=12.0)
    parser.add_argument("--match-thresh", type=float, default=0.55)
    parser.add_argument("--byte-track-thresh", type=float, default=0.25)
    parser.add_argument("--byte-match-thresh", type=float, default=0.8)
    parser.add_argument("--byte-track-buffer", type=int, default=30)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--max-missed", type=int, default=6)
    parser.add_argument("--association-min-hits", type=int, default=2)
    parser.add_argument("--association-lock-frames", type=int, default=20)
    parser.add_argument("--association-unbind-frames", type=int, default=15)
    parser.add_argument("--association-switch-margin", type=float, default=0.25)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    model = YOLO(args.model)
    tracker = _new_tracker(args)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    rows: List[Dict[str, object]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    for frame_index in range(args.start, args.end + 1):
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            frame,
            classes=TARGET_CLASSES,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        detections = result_to_detections(result, min_area=args.min_area)
        riders = [det for det in detections if det.class_id == RIDER_CLASS_ID]
        heads = [
            det for det in detections
            if det.class_id == HEAD_CLASS_ID and det.confidence >= args.head_conf
        ]
        helmets = [
            det for det in detections
            if det.class_id == HELMET_CLASS_ID and det.confidence >= args.helmet_conf
        ]
        evidence = build_head_helmet_evidence(heads, helmets)

        track_matches, grouped_evidence, match_scores = tracker.update_scene(
            evidence,
            riders,
            frame,
            args.match_thresh,
        )

        matched_lookup: Dict[Tuple[int, Tuple[int, int, int, int]], Tuple[str, float]] = {}
        for rider_idx, matched_items in grouped_evidence.items():
            for evidence_idx, item in enumerate(matched_items):
                stable_id = getattr(item, "stable_id", "")
                score = match_scores[rider_idx][evidence_idx] if evidence_idx < len(match_scores[rider_idx]) else 0.0
                matched_lookup[(rider_idx, _rounded_box_key(item.bbox))] = (stable_id, score)

        for rider_idx, rider in enumerate(riders):
            track_id = str(track_matches[rider_idx])
            for evidence_idx, item in enumerate(evidence):
                cx, cy = _center(item.bbox)
                matched = matched_lookup.get((rider_idx, _rounded_box_key(item.bbox)))
                custom_score = occupant_rider_match_score(item, rider)
                row = {
                    "frame": frame_index,
                    "track_id": track_id,
                    "rider_idx": rider_idx,
                    "rider_conf": f"{rider.confidence:.3f}",
                    "rider_bbox": _fmt_box(rider.bbox),
                    "evidence_idx": evidence_idx,
                    "evidence_class": _class_name(item.class_id),
                    "evidence_conf": f"{item.confidence:.3f}",
                    "evidence_bbox": _fmt_box(item.bbox),
                    "bbox_iou": f"{iou(item.bbox, rider.bbox):.4f}",
                    "overlap_ratio": f"{overlap_ratio(item.bbox, rider.bbox):.4f}",
                    "center_in_rider": int(point_in_box(cx, cy, rider.bbox)),
                    "match_score": f"{custom_score:.4f}",
                    "selected": int(matched is not None),
                    "selected_stable_id": matched[0] if matched else "",
                    "selected_score": f"{matched[1]:.4f}" if matched else "",
                }
                rows.append(row)
                if matched and (not args.track_id or args.track_id == track_id):
                    print(
                        f"frame={frame_index} track={track_id} rider_idx={rider_idx} "
                        f"{row['evidence_class']} conf={row['evidence_conf']} "
                        f"bbox_iou={row['bbox_iou']} overlap={row['overlap_ratio']} "
                        f"score={row['match_score']} stable={row['selected_stable_id']}"
                    )

    cap.release()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "track_id",
        "rider_idx",
        "rider_conf",
        "rider_bbox",
        "evidence_idx",
        "evidence_class",
        "evidence_conf",
        "evidence_bbox",
        "bbox_iou",
        "overlap_ratio",
        "center_in_rider",
        "match_score",
        "selected",
        "selected_stable_id",
        "selected_score",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
