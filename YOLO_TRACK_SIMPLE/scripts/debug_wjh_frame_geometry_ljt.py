#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui_wjh_ljt import HEAD_CLASS_ID, HELMET_CLASS_ID, RIDER_CLASS_ID, TARGET_CLASSES, _center  # noqa: E402
from pipeline_ljt import Detection, iou, overlap_ratio, point_in_box, result_to_detections  # noqa: E402


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


def evidence_inside_rider(evidence: Detection, rider: Detection, min_overlap: float) -> bool:
    cx, cy = _center(evidence.bbox)
    return point_in_box(cx, cy, rider.bbox) and overlap_ratio(evidence.bbox, rider.bbox) >= min_overlap


def best_head_helmet_pairs(heads: List[Detection], helmets: List[Detection], min_pair_overlap: float) -> List[Tuple[int, int, float, float]]:
    candidates: List[Tuple[float, int, int, float]] = []
    for head_idx, head in enumerate(heads):
        for helmet_idx, helmet in enumerate(helmets):
            ov = overlap_ratio(head.bbox, helmet.bbox)
            pair_iou = iou(head.bbox, helmet.bbox)
            if ov >= min_pair_overlap:
                candidates.append((ov, head_idx, helmet_idx, pair_iou))
    candidates.sort(reverse=True)

    used_heads: set[int] = set()
    used_helmets: set[int] = set()
    pairs: List[Tuple[int, int, float, float]] = []
    for ov, head_idx, helmet_idx, pair_iou in candidates:
        if head_idx in used_heads or helmet_idx in used_helmets:
            continue
        used_heads.add(head_idx)
        used_helmets.add(helmet_idx)
        pairs.append((head_idx, helmet_idx, ov, pair_iou))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug wjh.pt using per-frame rider-box geometry only.")
    parser.add_argument("--video", default="test.mp4")
    parser.add_argument("--model", default="weights/wjh.pt")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--out-prefix", default="runs/wjh_debug/wjh_frame_geometry")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--head-conf", type=float, default=0.25)
    parser.add_argument("--helmet-conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-area", type=float, default=12.0)
    parser.add_argument("--min-rider-overlap", type=float, default=0.80)
    parser.add_argument("--min-pair-overlap", type=float, default=0.30)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames - 1 if args.end < 0 else min(args.end, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    rider_rows: List[Dict[str, object]] = []
    evidence_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for frame_index in range(args.start, end_frame + 1):
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
        pairs = best_head_helmet_pairs(heads, helmets, args.min_pair_overlap)
        paired_heads = {head_idx for head_idx, _, _, _ in pairs}
        paired_helmets = {helmet_idx for _, helmet_idx, _, _ in pairs}

        for head_idx, helmet_idx, ov, pair_iou in pairs:
            pair_rows.append(
                {
                    "frame": frame_index,
                    "head_idx": head_idx,
                    "helmet_idx": helmet_idx,
                    "head_conf": f"{heads[head_idx].confidence:.3f}",
                    "helmet_conf": f"{helmets[helmet_idx].confidence:.3f}",
                    "head_bbox": _fmt_box(heads[head_idx].bbox),
                    "helmet_bbox": _fmt_box(helmets[helmet_idx].bbox),
                    "head_helmet_overlap": f"{ov:.4f}",
                    "head_helmet_iou": f"{pair_iou:.4f}",
                }
            )

        for rider_idx, rider in enumerate(riders):
            inside_heads = [
                (idx, item) for idx, item in enumerate(heads)
                if evidence_inside_rider(item, rider, args.min_rider_overlap)
            ]
            inside_helmets = [
                (idx, item) for idx, item in enumerate(helmets)
                if evidence_inside_rider(item, rider, args.min_rider_overlap)
            ]
            paired_inside_heads = sum(1 for idx, _ in inside_heads if idx in paired_heads)
            paired_inside_helmets = sum(1 for idx, _ in inside_helmets if idx in paired_helmets)
            unpaired_inside_heads = len(inside_heads) - paired_inside_heads
            unpaired_inside_helmets = len(inside_helmets) - paired_inside_helmets
            evidence_count = len(inside_heads) + len(inside_helmets)

            rider_rows.append(
                {
                    "frame": frame_index,
                    "rider_idx": rider_idx,
                    "rider_conf": f"{rider.confidence:.3f}",
                    "rider_bbox": _fmt_box(rider.bbox),
                    "head_count": len(inside_heads),
                    "helmet_count": len(inside_helmets),
                    "evidence_count": evidence_count,
                    "paired_head_count": paired_inside_heads,
                    "paired_helmet_count": paired_inside_helmets,
                    "unpaired_head_count": unpaired_inside_heads,
                    "unpaired_helmet_count": unpaired_inside_helmets,
                    "raw_overload_by_evidence": int(evidence_count >= 2),
                    "raw_overload_by_people_estimate": int((len(pairs) + unpaired_inside_heads + unpaired_inside_helmets) >= 2),
                    "no_helmet_by_head_only": int(unpaired_inside_heads > 0),
                }
            )

            for class_id, items in ((HEAD_CLASS_ID, inside_heads), (HELMET_CLASS_ID, inside_helmets)):
                for evidence_idx, item in items:
                    cx, cy = _center(item.bbox)
                    evidence_rows.append(
                        {
                            "frame": frame_index,
                            "rider_idx": rider_idx,
                            "evidence_idx": evidence_idx,
                            "evidence_class": _class_name(class_id),
                            "evidence_conf": f"{item.confidence:.3f}",
                            "evidence_bbox": _fmt_box(item.bbox),
                            "bbox_iou": f"{iou(item.bbox, rider.bbox):.4f}",
                            "overlap_ratio": f"{overlap_ratio(item.bbox, rider.bbox):.4f}",
                            "center_in_rider": int(point_in_box(cx, cy, rider.bbox)),
                        }
                    )

        if frame_index % 200 == 0:
            print(f"processed frame {frame_index}/{end_frame}")

    cap.release()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            out_prefix.with_name(out_prefix.name + "_riders.csv"),
            rider_rows,
            [
                "frame",
                "rider_idx",
                "rider_conf",
                "rider_bbox",
                "head_count",
                "helmet_count",
                "evidence_count",
                "paired_head_count",
                "paired_helmet_count",
                "unpaired_head_count",
                "unpaired_helmet_count",
                "raw_overload_by_evidence",
                "raw_overload_by_people_estimate",
                "no_helmet_by_head_only",
            ],
        ),
        (
            out_prefix.with_name(out_prefix.name + "_evidence.csv"),
            evidence_rows,
            [
                "frame",
                "rider_idx",
                "evidence_idx",
                "evidence_class",
                "evidence_conf",
                "evidence_bbox",
                "bbox_iou",
                "overlap_ratio",
                "center_in_rider",
            ],
        ),
        (
            out_prefix.with_name(out_prefix.name + "_head_helmet_pairs.csv"),
            pair_rows,
            [
                "frame",
                "head_idx",
                "helmet_idx",
                "head_conf",
                "helmet_conf",
                "head_bbox",
                "helmet_bbox",
                "head_helmet_overlap",
                "head_helmet_iou",
            ],
        ),
    ]
    for path, rows, fieldnames in outputs:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} rows={len(rows)}")


if __name__ == "__main__":
    main()
