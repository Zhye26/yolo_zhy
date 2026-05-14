#!/usr/bin/env python3
"""
Compute vehicle and overload counts inside a manually defined region.

Example:
    python tools/region_stats.py \
      --csv "/path/to/10-yolov8n-overload-ljt_frames_id_fixed.csv" \
      --region 200,300,900,700 \
      --output "/path/to/10_region_stats.csv"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_bbox(raw: str) -> List[float]:
    raw = raw.strip()
    if raw.startswith("["):
        import json

        return [float(item) for item in json.loads(raw)]
    return [float(item) for item in raw.replace(",", " ").split()]


def parse_region(raw: str) -> List[float]:
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError("region must contain four values: x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("region must satisfy x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]


def intersection_area(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_touches_region(box: List[float], region: List[float], mode: str) -> bool:
    if mode == "center":
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
    if mode == "intersect":
        return intersection_area(box, region) > 0
    return box_touches_region(box, region, "center") or box_touches_region(box, region, "intersect")


def row_vehicle_identity(row: Dict[str, str]) -> str:
    for key in ("final_vehicle_id", "correct_vehicle_id", "vehicle_track_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def row_is_overload(row: Dict[str, str]) -> bool:
    for key in ("confirmed_overload", "raw_overload"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "true", "yes"}:
            return True
    return False


def compute_stats(csv_path: Path, region: List[float], mode: str) -> Dict[str, object]:
    # Author: You Pinzhen - compute region-level vehicle and overload counts from corrected frame CSVs.
    vehicle_ids: set[str] = set()
    overload_vehicle_ids: set[str] = set()
    frame_hits: set[int] = set()
    row_hits = 0

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "vehicle_bbox" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} does not contain vehicle_bbox")

        for row in reader:
            try:
                bbox = parse_bbox(row["vehicle_bbox"])
            except Exception:
                continue
            if not box_touches_region(bbox, region, mode):
                continue

            row_hits += 1
            frame_value = row.get("frame") or row.get("frame_index") or ""
            if frame_value != "":
                frame_hits.add(int(float(frame_value)))
            vehicle_id = row_vehicle_identity(row)
            vehicle_ids.add(vehicle_id)
            if row_is_overload(row):
                overload_vehicle_ids.add(vehicle_id)

    return {
        "vehicle_count": len(vehicle_ids),
        "overload_vehicle_count": len(overload_vehicle_ids),
        "row_hits": row_hits,
        "frame_hits": len(frame_hits),
        "vehicle_ids": sorted(vehicle_ids),
        "overload_vehicle_ids": sorted(overload_vehicle_ids),
    }


def write_stats(output_path: Path, csv_path: Path, region: List[float], mode: str, stats: Dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "source_csv",
            "region_bbox",
            "region_mode",
            "vehicle_count",
            "overload_vehicle_count",
            "row_hits",
            "frame_hits",
            "vehicle_ids",
            "overload_vehicle_ids",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "source_csv": str(csv_path),
                "region_bbox": " ".join(f"{value:.1f}" for value in region),
                "region_mode": mode,
                "vehicle_count": stats["vehicle_count"],
                "overload_vehicle_count": stats["overload_vehicle_count"],
                "row_hits": stats["row_hits"],
                "frame_hits": stats["frame_hits"],
                "vehicle_ids": " ".join(stats["vehicle_ids"]),
                "overload_vehicle_ids": " ".join(stats["overload_vehicle_ids"]),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute region-level vehicle statistics from frame CSV")
    parser.add_argument("--csv", required=True, help="*_frames.csv or *_frames_id_fixed.csv")
    parser.add_argument("--region", required=True, help="x1,y1,x2,y2")
    parser.add_argument("--output", "-o", help="Output stats CSV")
    parser.add_argument(
        "--mode",
        choices=["center", "intersect", "either"],
        default="either",
        help="How a vehicle box enters the region",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV does not exist: {csv_path}")
        return 1

    try:
        region = parse_region(args.region)
    except ValueError as exc:
        print(f"Invalid region: {exc}")
        return 1

    stats = compute_stats(csv_path, region, args.mode)
    output_path = Path(args.output) if args.output else csv_path.with_name(f"{csv_path.stem}_region_stats.csv")
    write_stats(output_path, csv_path, region, args.mode, stats)

    print(f"source_csv: {csv_path}")
    print(f"region: {' '.join(f'{value:.1f}' for value in region)}")
    print(f"mode: {args.mode}")
    print(f"vehicle_count: {stats['vehicle_count']}")
    print(f"overload_vehicle_count: {stats['overload_vehicle_count']}")
    print(f"row_hits: {stats['row_hits']}")
    print(f"frame_hits: {stats['frame_hits']}")
    print(f"vehicle_ids: {' '.join(stats['vehicle_ids']) or '-'}")
    print(f"overload_vehicle_ids: {' '.join(stats['overload_vehicle_ids']) or '-'}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
