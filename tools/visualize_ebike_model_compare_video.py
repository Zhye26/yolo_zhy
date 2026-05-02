#!/usr/bin/env python3
"""
Visualize ebike and person detection comparison on a video.

Output layout (2x2):
- Top-left: original frame
- Top-right: baseline_best.pt (classes 0/1/2)
- Bottom-left: bifpn_best.pt (classes 0/1/2)
- Bottom-right: yolov8n.pt (classes 1/3 as ebike proxy, class 0 as person)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_CLASS_SPECS = {
    "baseline": [(0, "ebike"), (1, "driver"), (2, "passenger")],
    "bifpn": [(0, "ebike"), (1, "driver"), (2, "passenger")],
    "yolov8n": [(1, "bicycle"), (3, "motorcycle"), (0, "person")],
}


def draw_boxes(
    image,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int],
    labels: List[str],
    title: str,
) -> any:
    vis = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if i < len(labels):
            cv2.putText(
                vis,
                labels[i],
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def sync_cuda_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_filtered(model: YOLO, source, classes: List[int], conf: float, imgsz: int):
    r = model.predict(
        source=source,
        classes=classes,
        conf=conf,
        iou=0.7,
        imgsz=imgsz,
        verbose=False,
    )[0]
    return r


def predict_timed_filtered(
    model: YOLO,
    source,
    cls_id: int,
    conf: float,
    imgsz: int,
):
    sync_cuda_if_available()
    start = time.perf_counter()
    result = predict_filtered(model, source, classes=[cls_id], conf=conf, imgsz=imgsz)
    sync_cuda_if_available()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def result_to_boxes_and_labels(result, alias: Dict[int, str]) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[str] = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes, labels
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for b, c, k in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        boxes.append((x1, y1, x2, y2))
        cls_name = alias.get(int(k), str(int(k)))
        labels.append(f"{cls_name} {float(c):.2f}")
    return boxes, labels


def count_summary(counts: Dict[str, int]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in counts.items())


def predict_model_classes(
    model_name: str,
    model: YOLO,
    source,
    frame_id: str,
    conf: float,
    imgsz: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[Dict[str, object]], Dict[str, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[str] = []
    rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}

    for cls_id, cls_name in MODEL_CLASS_SPECS[model_name]:
        result, elapsed_ms = predict_timed_filtered(model, source, cls_id, conf, imgsz)
        cls_boxes, cls_labels = result_to_boxes_and_labels(result, {cls_id: cls_name})
        boxes.extend(cls_boxes)
        labels.extend(cls_labels)
        counts[cls_name] = len(cls_boxes)
        rows.append(
            {
                "frame": frame_id,
                "model": model_name,
                "class_id": cls_id,
                "class_name": cls_name,
                "elapsed_ms": f"{elapsed_ms:.3f}",
                "detections": len(cls_boxes),
            }
        )

    return boxes, labels, rows, counts


def predict_model_combined_timed(
    model_name: str,
    model: YOLO,
    source,
    frame_id: str,
    timestamp_sec: float,
    conf: float,
    imgsz: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], Dict[str, object], Dict[str, int]]:
    class_specs = MODEL_CLASS_SPECS[model_name]
    class_ids = [cls_id for cls_id, _ in class_specs]
    alias = {cls_id: cls_name for cls_id, cls_name in class_specs}

    sync_cuda_if_available()
    start = time.perf_counter()
    result = predict_filtered(model, source, classes=class_ids, conf=conf, imgsz=imgsz)
    sync_cuda_if_available()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    boxes, labels = result_to_boxes_and_labels(result, alias)
    counts = {cls_name: 0 for _, cls_name in class_specs}
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id in result.boxes.cls.cpu().numpy().astype(int):
            cls_name = alias.get(int(cls_id), str(int(cls_id)))
            counts[cls_name] = counts.get(cls_name, 0) + 1

    row = {
        "frame": frame_id,
        "timestamp_sec": f"{timestamp_sec:.3f}",
        "model": model_name,
        "class_ids": "+".join(str(cls_id) for cls_id in class_ids),
        "class_names": "+".join(cls_name for _, cls_name in class_specs),
        "elapsed_ms": f"{elapsed_ms:.3f}",
        "fps": f"{1000.0 / elapsed_ms:.3f}" if elapsed_ms > 0 else "",
        "detections": len(boxes),
        "detection_summary": count_summary(counts),
    }
    return boxes, labels, row, counts


def warmup_model(model_name: str, model: YOLO, source, conf: float, imgsz: int, repeats: int) -> None:
    class_ids = [cls_id for cls_id, _ in MODEL_CLASS_SPECS[model_name]]
    for _ in range(repeats):
        predict_filtered(model, source, classes=class_ids, conf=conf, imgsz=imgsz)
    sync_cuda_if_available()


def main():
    parser = argparse.ArgumentParser(description="Visualize ebike model comparison on video")
    parser.add_argument("--video", default="data/test_video/docker-compose.cpu.mp4")
    parser.add_argument("--out", default="data/model_compare_video_vis/docker-compose.cpu_compare.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up predictions per model before timing")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max processed frames")
    parser.add_argument(
        "--record-class-timing",
        action="store_true",
        help="Also run and record per-model, per-class timings. This adds extra inference passes.",
    )
    parser.add_argument(
        "--timing-csv",
        default=None,
        help="CSV path for per-frame, per-model, per-class inference timings",
    )
    parser.add_argument(
        "--combined-timing-csv",
        default=None,
        help="CSV path for per-frame, per-model combined multi-class inference timings and FPS",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timing_csv = (
        Path(args.timing_csv)
        if args.timing_csv
        else out_path.with_name(f"{out_path.stem}_timing_by_frame_model_class.csv")
    )
    combined_timing_csv = (
        Path(args.combined_timing_csv)
        if args.combined_timing_csv
        else out_path.with_name(f"{out_path.stem}_timing_by_frame_model_combined.csv")
    )
    timing_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_timing_csv.parent.mkdir(parents=True, exist_ok=True)

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
        (frame_w * 2, frame_h * 2),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open output video writer: {out_path}")

    baseline = YOLO("models/baseline_best.pt")
    bifpn = YOLO("models/bifpn_best.pt")
    y8 = YOLO("yolov8n.pt")

    ok, first_frame = cap.read()
    if not ok:
        writer.release()
        cap.release()
        raise SystemExit(f"Video has no readable frames: {video_path}")

    if args.warmup > 0:
        warmup_model("baseline", baseline, first_frame, args.conf, args.imgsz, args.warmup)
        warmup_model("bifpn", bifpn, first_frame, args.conf, args.imgsz, args.warmup)
        warmup_model("yolov8n", y8, first_frame, args.conf, args.imgsz, args.warmup)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    timing_rows: List[Dict[str, object]] = []
    combined_timing_rows: List[Dict[str, object]] = []
    processed = 0
    frame_index = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % max(1, args.frame_stride) != 0:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        frame_id = str(frame_index)
        timestamp_sec = frame_index / source_fps if source_fps > 0 else 0.0

        base_boxes, base_labels, base_combined_row, base_counts = predict_model_combined_timed(
            "baseline", baseline, frame, frame_id, timestamp_sec, args.conf, args.imgsz
        )
        bifpn_boxes, bifpn_labels, bifpn_combined_row, bifpn_counts = predict_model_combined_timed(
            "bifpn", bifpn, frame, frame_id, timestamp_sec, args.conf, args.imgsz
        )
        y8_boxes, y8_labels, y8_combined_row, y8_counts = predict_model_combined_timed(
            "yolov8n", y8, frame, frame_id, timestamp_sec, args.conf, args.imgsz
        )
        combined_timing_rows.extend([base_combined_row, bifpn_combined_row, y8_combined_row])

        if args.record_class_timing:
            _, _, base_rows, _ = predict_model_classes("baseline", baseline, frame, frame_id, args.conf, args.imgsz)
            _, _, bifpn_rows, _ = predict_model_classes("bifpn", bifpn, frame, frame_id, args.conf, args.imgsz)
            _, _, y8_rows, _ = predict_model_classes("yolov8n", y8, frame, frame_id, args.conf, args.imgsz)
            timing_rows.extend(base_rows)
            timing_rows.extend(bifpn_rows)
            timing_rows.extend(y8_rows)

        p_original = draw_boxes(frame, [], (255, 255, 255), [], "original")
        p_base = draw_boxes(frame, base_boxes, (0, 255, 0), base_labels, f"baseline: {count_summary(base_counts)}")
        p_bifpn = draw_boxes(frame, bifpn_boxes, (255, 0, 0), bifpn_labels, f"bifpn: {count_summary(bifpn_counts)}")
        p_y8 = draw_boxes(frame, y8_boxes, (0, 128, 255), y8_labels, f"yolov8n: {count_summary(y8_counts)}")

        top = cv2.hconcat([p_original, p_base])
        bottom = cv2.hconcat([p_bifpn, p_y8])
        canvas = cv2.vconcat([top, bottom])

        cv2.putText(
            canvas,
            f"{video_path.name} frame={frame_index} t={timestamp_sec:.2f}s",
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(canvas)
        processed += 1
        if processed % 50 == 0:
            print(f"[processed] {processed} frames")

    writer.release()
    cap.release()

    if timing_rows:
        with timing_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "model", "class_id", "class_name", "elapsed_ms", "detections"],
            )
            writer.writeheader()
            writer.writerows(timing_rows)
        print(f"[timing] {timing_csv}")

    if combined_timing_rows:
        with combined_timing_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "timestamp_sec",
                    "model",
                    "class_ids",
                    "class_names",
                    "elapsed_ms",
                    "fps",
                    "detections",
                    "detection_summary",
                ],
            )
            writer.writeheader()
            writer.writerows(combined_timing_rows)
        print(f"[combined timing] {combined_timing_csv}")

    print(f"[video] {out_path}")
    print(f"[summary] source_frames={total_frames} processed_frames={processed} source_fps={source_fps:.3f} output_fps={output_fps:.3f}")


if __name__ == "__main__":
    main()
