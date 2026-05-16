#!/usr/bin/env python3
"""Compare gui_cross_validate_ljt vs association outputs on one standard video."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


@dataclass
class FrameMetrics:
    total_frames: int
    positive_frames: int
    predicted_positive_frames: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class Region:
    x1: float
    y1: float
    x2: float
    y2: float


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_bbox(value: str) -> Tuple[float, float, float, float]:
    vals = [float(item) for item in str(value).split()]
    if len(vals) != 4:
        raise ValueError(f"Invalid bbox: {value}")
    return vals[0], vals[1], vals[2], vals[3]


def center_in_region(bbox: Sequence[float], region: Region) -> bool:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2


def read_region(region_csv: Path) -> Region:
    rows = read_csv(region_csv)
    if not rows:
        raise ValueError(f"Empty region CSV: {region_csv}")
    bbox = parse_bbox(rows[0]["region_bbox"])
    return Region(*bbox)


def load_gt_frame_overload(gt_csv: Path) -> Dict[int, bool]:
    gt_rows = read_csv(gt_csv)
    frame_map: Dict[int, bool] = {}
    for row in gt_rows:
        frame = int(row["frame"])
        overload = _to_bool(row.get("confirmed_overload"))
        frame_map[frame] = frame_map.get(frame, False) or overload
    return frame_map


def load_cross_frame_overload(cross_frames_csv: Path) -> Dict[int, bool]:
    rows = read_csv(cross_frames_csv)
    return {int(row["frame"]): _to_bool(row.get("final_overload")) for row in rows}


def load_association_frame_overload(association_frames_csv: Path) -> Dict[int, bool]:
    rows = read_csv(association_frames_csv)
    frame_map: Dict[int, bool] = {}
    for row in rows:
        frame = int(row["frame"])
        overload = _to_bool(row.get("confirmed_overload"))
        frame_map[frame] = frame_map.get(frame, False) or overload
    return frame_map


def load_association_frame_overload_in_region(association_frames_csv: Path, region: Region) -> Dict[int, bool]:
    rows = read_csv(association_frames_csv)
    frame_map: Dict[int, bool] = {}
    for row in rows:
        bbox = parse_bbox(row["vehicle_bbox"])
        if not center_in_region(bbox, region):
            continue
        frame = int(row["frame"])
        overload = _to_bool(row.get("confirmed_overload"))
        frame_map[frame] = frame_map.get(frame, False) or overload
    return frame_map


def evaluate_frame_metrics(gt: Dict[int, bool], pred: Dict[int, bool], universe_frames: Iterable[int]) -> FrameMetrics:
    tp = fp = fn = tn = 0
    frames = list(universe_frames)

    for frame in frames:
        g = bool(gt.get(frame, False))
        p = bool(pred.get(frame, False))
        if g and p:
            tp += 1
        elif (not g) and p:
            fp += 1
        elif g and (not p):
            fn += 1
        else:
            tn += 1

    total = len(frames)
    positive = sum(1 for f in frames if gt.get(f, False))
    pred_positive = sum(1 for f in frames if pred.get(f, False))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, total)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    return FrameMetrics(
        total_frames=total,
        positive_frames=positive,
        predicted_positive_frames=pred_positive,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
    )


def parse_assoc_perf(summary_csv: Path, frame_csv: Path) -> Tuple[Optional[float], Optional[float], bool]:
    avg_fps = None
    avg_elapsed_ms = None

    if summary_csv.exists():
        rows = read_csv(summary_csv)
        if rows:
            row = rows[0]
            try:
                avg_fps = float(row.get("avg_fps", ""))
            except ValueError:
                avg_fps = None
            try:
                avg_elapsed_ms = float(row.get("avg_elapsed_ms", ""))
            except ValueError:
                avg_elapsed_ms = None

    if (avg_fps is None or avg_elapsed_ms is None) and frame_csv.exists():
        rows = read_csv(frame_csv)
        elapsed_vals: List[float] = []
        fps_vals: List[float] = []
        for row in rows:
            try:
                elapsed_vals.append(float(row.get("elapsed_ms", "")))
            except ValueError:
                pass
            try:
                fps_vals.append(float(row.get("fps", "")))
            except ValueError:
                pass
        if avg_elapsed_ms is None and elapsed_vals:
            avg_elapsed_ms = sum(elapsed_vals) / len(elapsed_vals)
        if avg_fps is None:
            if fps_vals:
                avg_fps = sum(fps_vals) / len(fps_vals)
            elif avg_elapsed_ms and avg_elapsed_ms > 0:
                avg_fps = 1000.0 / avg_elapsed_ms

    return avg_fps, avg_elapsed_ms, False


def parse_assoc_region_perf(frame_csv: Path, region: Region) -> Tuple[Optional[float], Optional[float]]:
    rows = read_csv(frame_csv)
    fps_vals: List[float] = []
    elapsed_vals: List[float] = []
    for row in rows:
        bbox = parse_bbox(row["vehicle_bbox"])
        if not center_in_region(bbox, region):
            continue
        try:
            fps_vals.append(float(row.get("fps", "")))
        except ValueError:
            pass
        try:
            elapsed_vals.append(float(row.get("elapsed_ms", "")))
        except ValueError:
            pass
    avg_fps = (sum(fps_vals) / len(fps_vals)) if fps_vals else None
    avg_elapsed = (sum(elapsed_vals) / len(elapsed_vals)) if elapsed_vals else None
    if avg_fps is None and avg_elapsed and avg_elapsed > 0:
        avg_fps = 1000.0 / avg_elapsed
    return avg_fps, avg_elapsed


def parse_cross_perf(cross_frames_csv: Path, video_path: Path) -> Tuple[Optional[float], Optional[float], bool]:
    rows = read_csv(cross_frames_csv)
    if not rows:
        return None, None, False

    # cross frame csv currently has no elapsed/fps columns in this project branch.
    if "fps" not in rows[0] and "elapsed_ms" not in rows[0]:
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        return (source_fps if source_fps > 0 else None), None, True

    elapsed_vals: List[float] = []
    fps_vals: List[float] = []
    for row in rows:
        try:
            elapsed_vals.append(float(row.get("elapsed_ms", "")))
        except ValueError:
            pass
        try:
            fps_vals.append(float(row.get("fps", "")))
        except ValueError:
            pass

    avg_elapsed = sum(elapsed_vals) / len(elapsed_vals) if elapsed_vals else None
    avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else (1000.0 / avg_elapsed if avg_elapsed and avg_elapsed > 0 else None)
    return avg_fps, avg_elapsed, False


def compute_association_id_consistency(gt_csv: Path, association_frames_csv: Path, region: Region, iou_thresh: float = 0.5) -> Tuple[float, int]:
    gt_rows = read_csv(gt_csv)
    pred_rows = read_csv(association_frames_csv)
    gt_by_frame: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
    pred_by_frame: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}

    for row in gt_rows:
        bbox = parse_bbox(row["vehicle_bbox"])
        if not center_in_region(bbox, region):
            continue
        frame = int(row["frame"])
        gt_id = (row.get("final_vehicle_id") or row.get("vehicle_track_id") or "").strip()
        if not gt_id:
            continue
        gt_by_frame.setdefault(frame, []).append((gt_id, bbox))

    for row in pred_rows:
        bbox = parse_bbox(row["vehicle_bbox"])
        if not center_in_region(bbox, region):
            continue
        frame = int(row["frame"])
        pred_id = str(row.get("vehicle_track_id", "")).strip()
        if not pred_id:
            continue
        pred_by_frame.setdefault(frame, []).append((pred_id, bbox))

    mapping: Dict[str, Dict[str, int]] = {}
    total_matches = 0
    for frame, gt_items in gt_by_frame.items():
        pred_items = pred_by_frame.get(frame, [])
        if not pred_items:
            continue
        used_pred: set[int] = set()
        for gt_id, gt_bbox in gt_items:
            best_idx = -1
            best_iou = 0.0
            for idx, (_pred_id, pred_bbox) in enumerate(pred_items):
                if idx in used_pred:
                    continue
                score = _bbox_iou(gt_bbox, pred_bbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_thresh:
                used_pred.add(best_idx)
                pred_id = pred_items[best_idx][0]
                mapping.setdefault(gt_id, {})
                mapping[gt_id][pred_id] = mapping[gt_id].get(pred_id, 0) + 1
                total_matches += 1

    if total_matches == 0:
        return 0.0, 0
    dominant = sum(max(counter.values()) for counter in mapping.values() if counter)
    id_switches = sum(max(0, len(counter) - 1) for counter in mapping.values() if counter)
    return dominant / total_matches, id_switches


def compute_cross_id_stability_proxy(cross_targets_csv: Path) -> Tuple[float, int]:
    rows = read_csv(cross_targets_csv)
    frame_to_ids: Dict[int, set[str]] = {}
    for row in rows:
        if not _to_bool(row.get("final_overload")):
            continue
        frame = int(row["frame"])
        target_id = str(row.get("target_id", "")).strip()
        if not target_id:
            continue
        frame_to_ids.setdefault(frame, set()).add(target_id)

    frames = sorted(frame_to_ids)
    if len(frames) <= 1:
        return 1.0 if frames else 0.0, 0

    kept = 0
    total = 0
    switches = 0
    prev_ids = frame_to_ids[frames[0]]
    for frame in frames[1:]:
        cur_ids = frame_to_ids[frame]
        overlap = len(prev_ids & cur_ids)
        total += max(1, len(prev_ids))
        kept += overlap
        if overlap == 0 and prev_ids and cur_ids:
            switches += 1
        prev_ids = cur_ids
    return _safe_div(kept, total), switches


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fmt_float(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def make_rows(
    name: str,
    m: FrameMetrics,
    avg_fps: Optional[float],
    avg_elapsed_ms: Optional[float],
    fps_is_source_fallback: bool,
) -> List[Tuple[str, str]]:
    return [
        (f"{name}.total_frames", str(m.total_frames)),
        (f"{name}.gt_positive_frames", str(m.positive_frames)),
        (f"{name}.pred_positive_frames", str(m.predicted_positive_frames)),
        (f"{name}.tp", str(m.tp)),
        (f"{name}.fp", str(m.fp)),
        (f"{name}.fn", str(m.fn)),
        (f"{name}.tn", str(m.tn)),
        (f"{name}.precision", fmt_float(m.precision)),
        (f"{name}.recall", fmt_float(m.recall)),
        (f"{name}.f1", fmt_float(m.f1)),
        (f"{name}.accuracy", fmt_float(m.accuracy)),
        (f"{name}.false_positive_rate", fmt_float(m.false_positive_rate)),
        (f"{name}.false_negative_rate", fmt_float(m.false_negative_rate)),
        (f"{name}.avg_fps", fmt_float(avg_fps, 3)),
        (f"{name}.avg_elapsed_ms", fmt_float(avg_elapsed_ms, 3)),
        (f"{name}.fps_is_source_fallback", str(int(fps_is_source_fallback))),
    ]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_video_dir = root / "standardData" / "13"

    parser = argparse.ArgumentParser(description="Compare cross-validate method vs association method on one standard video.")
    parser.add_argument("--video-dir", type=Path, default=default_video_dir)
    parser.add_argument("--out-csv", type=Path, default=default_video_dir / "13-cross-vs-association-compare-ljt.csv")
    parser.add_argument("--target-label", type=str, default="association")
    parser.add_argument("--baseline-label", type=str, default="cross")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir

    stem = video_dir.name
    gt_csv = video_dir / f"{stem}-yolov8n-overload-ypz_frames_id_fixed.csv"
    cross_frames_csv = video_dir / f"{stem}-cross-validate-ljt_frames.csv"
    cross_targets_csv = video_dir / f"{stem}-cross-validate-ljt_targets.csv"
    association_frames_csv = video_dir / f"{stem}-yolov8n-overload-association-ljt_frames.csv"
    association_summary_csv = video_dir / f"{stem}-yolov8n-overload-association-ljt_summary.csv"
    video_path = video_dir / f"{stem}.mp4"
    region_csv = video_dir / f"{stem}-yolov8n-overload-ypz_frames_region_stats.csv"

    required = [gt_csv, cross_frames_csv, cross_targets_csv, association_frames_csv, region_csv]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    gt = load_gt_frame_overload(gt_csv)
    region = read_region(region_csv)
    cross_pred = load_cross_frame_overload(cross_frames_csv)
    assoc_pred = load_association_frame_overload(association_frames_csv)
    assoc_pred_region = load_association_frame_overload_in_region(association_frames_csv, region)

    all_frames = sorted(set(gt) | set(cross_pred) | set(assoc_pred))

    cross_metrics = evaluate_frame_metrics(gt, cross_pred, all_frames)
    assoc_metrics = evaluate_frame_metrics(gt, assoc_pred, all_frames)

    cross_fps, cross_elapsed, cross_fallback = parse_cross_perf(cross_frames_csv, video_path)
    assoc_fps, assoc_elapsed, assoc_fallback = parse_assoc_perf(association_summary_csv, association_frames_csv)
    assoc_region_fps, assoc_region_elapsed = parse_assoc_region_perf(association_frames_csv, region)
    assoc_id_consistency, assoc_id_switches = compute_association_id_consistency(gt_csv, association_frames_csv, region)
    cross_id_stability_proxy, cross_id_switch_proxy = compute_cross_id_stability_proxy(cross_targets_csv)

    rows: List[Tuple[str, str]] = []
    rows.extend(make_rows("cross", cross_metrics, cross_fps, cross_elapsed, cross_fallback))
    rows.extend(make_rows("association", assoc_metrics, assoc_fps, assoc_elapsed, assoc_fallback))
    rows.extend(
        [
            ("region.region_bbox", f"{region.x1:.1f} {region.y1:.1f} {region.x2:.1f} {region.y2:.1f}"),
            ("region.cross_speed_fps", fmt_float(cross_fps, 3)),
            ("region.cross_speed_fps_is_source_fallback", str(int(cross_fallback))),
            ("region.association_speed_fps", fmt_float(assoc_region_fps, 3)),
            ("region.association_speed_elapsed_ms", fmt_float(assoc_region_elapsed, 3)),
            ("region.cross_overload_frames", str(sum(1 for v in cross_pred.values() if v))),
            ("region.association_overload_frames", str(sum(1 for v in assoc_pred_region.values() if v))),
            ("region.gt_overload_frames", str(sum(1 for v in gt.values() if v))),
            ("region.association_id_consistency_gt", fmt_float(assoc_id_consistency)),
            ("region.association_id_switches_gt", str(assoc_id_switches)),
            ("region.cross_id_stability_proxy", fmt_float(cross_id_stability_proxy)),
            ("region.cross_id_switch_proxy", str(cross_id_switch_proxy)),
        ]
    )

    baseline_label = args.baseline_label
    target_label = args.target_label
    baseline = cross_metrics if baseline_label == "cross" else assoc_metrics
    target = assoc_metrics if target_label == "association" else cross_metrics
    recall_impr = _safe_div(target.recall - baseline.recall, baseline.recall)
    f1_impr = _safe_div(target.f1 - baseline.f1, baseline.f1)
    fn_reduction = _safe_div(baseline.fn - target.fn, baseline.fn)
    fpr_reduction = _safe_div(baseline.false_positive_rate - target.false_positive_rate, baseline.false_positive_rate)
    rows.extend(
        [
            ("derived.baseline_label", baseline_label),
            ("derived.target_label", target_label),
            ("derived.tracking_improvement_rate_recall", fmt_float(recall_impr)),
            ("derived.tracking_improvement_rate_f1", fmt_float(f1_impr)),
            ("derived.miss_reduction_rate", fmt_float(fn_reduction)),
            ("derived.false_positive_rate_reduction", fmt_float(fpr_reduction)),
        ]
    )

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    print(f"[done] wrote comparison: {out_csv}")
    print(f"[cross] f1={cross_metrics.f1:.4f} precision={cross_metrics.precision:.4f} recall={cross_metrics.recall:.4f} avg_fps={fmt_float(cross_fps, 3)}")
    print(f"[association] f1={assoc_metrics.f1:.4f} precision={assoc_metrics.precision:.4f} recall={assoc_metrics.recall:.4f} avg_fps={fmt_float(assoc_fps, 3)}")
    print(
        f"[derived] target={target_label} baseline={baseline_label} "
        f"tracking_improvement_rate_recall={recall_impr:.4f} "
        f"tracking_improvement_rate_f1={f1_impr:.4f} "
        f"miss_reduction_rate={fn_reduction:.4f}"
    )


if __name__ == "__main__":
    main()
