#!/usr/bin/env python3
"""
Validate trained ebike models on a real video using the same core predict style
as the web demo: direct Ultralytics model.predict on each frame.

This script intentionally does not use yolov8n bicycle/motorcycle proxy classes.
It only evaluates the trained models' own class 0: ebike.
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


MODELS = {
    "baseline": "models/baseline_best.pt",
    "bifpn": "models/bifpn_best.pt",
}
EBIKE_CLASS_ID = 0


def sync_cuda_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_thresholds(raw: str) -> List[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one confidence threshold is required")
    return values


def predict_web_style(
    model: YOLO,
    frame,
    conf: float,
    iou: float,
    imgsz: int,
):
    sync_cuda_if_available()
    start = time.perf_counter()
    result = model.predict(
        frame,
        classes=[EBIKE_CLASS_ID],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )[0]
    sync_cuda_if_available()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def result_to_predictions(result) -> List[Tuple[Tuple[int, int, int, int], float]]:
    predictions: List[Tuple[Tuple[int, int, int, int], float]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        predictions.append(((x1, y1, x2, y2), float(conf)))
    return predictions


def filter_predictions(
    predictions: List[Tuple[Tuple[int, int, int, int], float]],
    conf: float,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    return [(box, score) for box, score in predictions if score >= conf]


def draw_predictions(
    frame,
    predictions: List[Tuple[Tuple[int, int, int, int], float]],
    color: Tuple[int, int, int],
    title: str,
):
    vis = frame.copy()
    for box, score in predictions:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"ebike {score:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def draw_original(frame, title: str):
    vis = frame.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def draw_overlay(frame, baseline_preds, bifpn_preds, title: str):
    vis = frame.copy()
    for box, score in baseline_preds:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"B {score:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for box, score in bifpn_preds:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, f"F {score:.2f}", (x1, min(vis.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Validate trained ebike models on video without COCO proxy classes")
    parser.add_argument("--video", default="data/test_video/docker-compose.cpu.mp4")
    parser.add_argument("--out", default="data/model_compare_video_vis/trained_ebike_web_style.mp4")
    parser.add_argument("--csv", default=None, help="Per-frame prediction CSV path")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path")
    parser.add_argument("--imgsz", type=int, default=640, help="Web default is 640")
    parser.add_argument("--iou", type=float, default=0.45, help="Web default is 0.45")
    parser.add_argument("--conf", type=float, default=0.5, help="Visualization confidence; Web default is 0.5")
    parser.add_argument(
        "--report-confs",
        default="0.05,0.25,0.5",
        help="Comma-separated confidence thresholds to report from one low-conf prediction pass",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_path.with_name(f"{out_path.stem}_per_frame.csv")
    summary_path = Path(args.summary_csv) if args.summary_csv else out_path.with_name(f"{out_path.stem}_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    report_confs = parse_thresholds(args.report_confs)
    predict_conf = min(report_confs + [args.conf])

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

    models = {name: YOLO(path) for name, path in MODELS.items()}

    ok, first_frame = cap.read()
    if not ok:
        writer.release()
        cap.release()
        raise SystemExit(f"Video has no readable frames: {video_path}")

    for _ in range(args.warmup):
        for model in models.values():
            model.predict(first_frame, classes=[EBIKE_CLASS_ID], conf=predict_conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
    sync_cuda_if_available()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows: List[Dict[str, object]] = []
    summary: Dict[Tuple[str, float], Dict[str, float]] = {
        (model_name, conf): {"frames_with_detection": 0, "detections": 0, "confidence_sum": 0.0}
        for model_name in models
        for conf in report_confs
    }
    timings: Dict[str, List[float]] = {model_name: [] for model_name in models}

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

        timestamp_sec = frame_index / source_fps if source_fps > 0 else 0.0
        frame_predictions = {}

        for model_name, model in models.items():
            result, elapsed_ms = predict_web_style(model, frame, predict_conf, args.iou, args.imgsz)
            timings[model_name].append(elapsed_ms)
            predictions = result_to_predictions(result)
            frame_predictions[model_name] = predictions

            for conf in report_confs:
                filtered = filter_predictions(predictions, conf)
                if filtered:
                    summary[(model_name, conf)]["frames_with_detection"] += 1
                summary[(model_name, conf)]["detections"] += len(filtered)
                summary[(model_name, conf)]["confidence_sum"] += sum(score for _, score in filtered)
                rows.append(
                    {
                        "frame": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.3f}",
                        "model": model_name,
                        "model_path": MODELS[model_name],
                        "class_id": EBIKE_CLASS_ID,
                        "class_name": "ebike",
                        "report_conf": f"{conf:.3f}",
                        "predict_conf": f"{predict_conf:.3f}",
                        "imgsz": args.imgsz,
                        "iou": args.iou,
                        "elapsed_ms": f"{elapsed_ms:.3f}",
                        "fps": f"{1000.0 / elapsed_ms:.3f}" if elapsed_ms > 0 else "",
                        "detections": len(filtered),
                        "max_conf": f"{max((score for _, score in filtered), default=0.0):.3f}",
                    }
                )

        baseline_draw = filter_predictions(frame_predictions["baseline"], args.conf)
        bifpn_draw = filter_predictions(frame_predictions["bifpn"], args.conf)
        p_original = draw_original(frame, f"original frame={frame_index} t={timestamp_sec:.2f}s")
        p_baseline = draw_predictions(frame, baseline_draw, (0, 255, 0), f"baseline ebike@{args.conf}: {len(baseline_draw)}")
        p_bifpn = draw_predictions(frame, bifpn_draw, (255, 0, 0), f"bifpn ebike@{args.conf}: {len(bifpn_draw)}")
        p_overlay = draw_overlay(frame, baseline_draw, bifpn_draw, "overlay: baseline green, bifpn blue")

        writer.write(cv2.vconcat([cv2.hconcat([p_original, p_baseline]), cv2.hconcat([p_bifpn, p_overlay])]))

        processed += 1
        if processed % 50 == 0:
            print(f"[processed] {processed} frames")

    writer.release()
    cap.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "timestamp_sec",
            "model",
            "model_path",
            "class_id",
            "class_name",
            "report_conf",
            "predict_conf",
            "imgsz",
            "iou",
            "elapsed_ms",
            "fps",
            "detections",
            "max_conf",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "model_path",
            "report_conf",
            "processed_frames",
            "frames_with_detection",
            "frame_hit_rate",
            "detections",
            "avg_confidence",
            "avg_elapsed_ms",
            "avg_fps",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for model_name in models:
            avg_ms = sum(timings[model_name]) / len(timings[model_name]) if timings[model_name] else 0.0
            for conf in report_confs:
                item = summary[(model_name, conf)]
                detections = int(item["detections"])
                frames_with_detection = int(item["frames_with_detection"])
                writer_csv.writerow(
                    {
                        "model": model_name,
                        "model_path": MODELS[model_name],
                        "report_conf": f"{conf:.3f}",
                        "processed_frames": processed,
                        "frames_with_detection": frames_with_detection,
                        "frame_hit_rate": f"{frames_with_detection / processed:.4f}" if processed else "0.0000",
                        "detections": detections,
                        "avg_confidence": f"{item['confidence_sum'] / detections:.4f}" if detections else "0.0000",
                        "avg_elapsed_ms": f"{avg_ms:.3f}",
                        "avg_fps": f"{1000.0 / avg_ms:.3f}" if avg_ms > 0 else "",
                    }
                )

    print(f"[video] {out_path}")
    print(f"[per-frame csv] {csv_path}")
    print(f"[summary csv] {summary_path}")
    print(f"[summary] source_frames={total_frames} processed_frames={processed} source_fps={source_fps:.3f} output_fps={output_fps:.3f}")


if __name__ == "__main__":
    main()
