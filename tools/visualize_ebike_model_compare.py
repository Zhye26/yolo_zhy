#!/usr/bin/env python3
"""
Visualize ebike and person detection comparison on the same images.

Output layout (2x2):
- Top-left: GT boxes (ebike/driver/passenger)
- Top-right: baseline_best.pt (classes 0/1/2)
- Bottom-left: bifpn_best.pt (classes 0/1/2)
- Bottom-right: yolov8n.pt (classes 1/3 as ebike proxy, class 0 as person)
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import torch
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_CLASSES = {0: "ebike_gt", 1: "driver_gt", 2: "passenger_gt"}
MODEL_CLASS_SPECS = {
    "baseline": [(0, "ebike"), (1, "driver"), (2, "passenger")],
    "bifpn": [(0, "ebike"), (1, "driver"), (2, "passenger")],
    "yolov8n": [(1, "bicycle"), (3, "motorcycle"), (0, "person")],
}


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def load_gt_xyxy_norm(label_path: Path, class_names: Dict[int, str]) -> List[Tuple[int, List[float]]]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    out: List[Tuple[int, List[float]]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in class_names:
            continue
        x, y, w, h = map(float, parts[1:5])
        out.append((cls, [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]))
    return out


def norm_to_pixel(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(box[0] * w))))
    y1 = max(0, min(h - 1, int(round(box[1] * h))))
    x2 = max(0, min(w - 1, int(round(box[2] * w))))
    y2 = max(0, min(h - 1, int(round(box[3] * h))))
    return x1, y1, x2, y2


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


def predict_filtered(model: YOLO, image_path: Path, classes: List[int], conf: float, imgsz: int):
    r = model.predict(
        source=str(image_path),
        classes=classes,
        conf=conf,
        iou=0.7,
        imgsz=imgsz,
        verbose=False,
    )[0]
    return r


def predict_timed_filtered(
    model: YOLO,
    image_path: Path,
    cls_id: int,
    conf: float,
    imgsz: int,
):
    sync_cuda_if_available()
    start = time.perf_counter()
    result = predict_filtered(model, image_path, classes=[cls_id], conf=conf, imgsz=imgsz)
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
    image_path: Path,
    conf: float,
    imgsz: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[Dict[str, object]], Dict[str, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[str] = []
    rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}

    for cls_id, cls_name in MODEL_CLASS_SPECS[model_name]:
        result, elapsed_ms = predict_timed_filtered(model, image_path, cls_id, conf, imgsz)
        cls_boxes, cls_labels = result_to_boxes_and_labels(result, {cls_id: cls_name})
        boxes.extend(cls_boxes)
        labels.extend(cls_labels)
        counts[cls_name] = len(cls_boxes)
        rows.append(
            {
                "image": image_path.name,
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
    image_path: Path,
    conf: float,
    imgsz: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], Dict[str, object], Dict[str, int]]:
    class_specs = MODEL_CLASS_SPECS[model_name]
    class_ids = [cls_id for cls_id, _ in class_specs]
    alias = {cls_id: cls_name for cls_id, cls_name in class_specs}

    sync_cuda_if_available()
    start = time.perf_counter()
    result = predict_filtered(model, image_path, classes=class_ids, conf=conf, imgsz=imgsz)
    sync_cuda_if_available()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    boxes, labels = result_to_boxes_and_labels(result, alias)
    counts = {cls_name: 0 for _, cls_name in class_specs}
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id in result.boxes.cls.cpu().numpy().astype(int):
            cls_name = alias.get(int(cls_id), str(int(cls_id)))
            counts[cls_name] = counts.get(cls_name, 0) + 1

    row = {
        "image": image_path.name,
        "model": model_name,
        "class_ids": "+".join(str(cls_id) for cls_id in class_ids),
        "class_names": "+".join(cls_name for _, cls_name in class_specs),
        "elapsed_ms": f"{elapsed_ms:.3f}",
        "fps": f"{1000.0 / elapsed_ms:.3f}" if elapsed_ms > 0 else "",
        "detections": len(boxes),
        "detection_summary": count_summary(counts),
    }
    return boxes, labels, row, counts


def warmup_model(model_name: str, model: YOLO, image_path: Path, conf: float, imgsz: int, repeats: int) -> None:
    class_ids = [cls_id for cls_id, _ in MODEL_CLASS_SPECS[model_name]]
    for _ in range(repeats):
        predict_filtered(model, image_path, classes=class_ids, conf=conf, imgsz=imgsz)
    sync_cuda_if_available()


def main():
    parser = argparse.ArgumentParser(description="Visualize ebike model comparison")
    parser.add_argument("--images", default="data/merged_dataset/images/test")
    parser.add_argument("--labels", default="data/merged_dataset/labels/test")
    parser.add_argument("--out", default="data/model_compare_vis")
    parser.add_argument("--sample", type=int, default=50, help="Number of random images to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image", default=None, help="Optional single image path")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up predictions per model before timing")
    parser.add_argument(
        "--timing-csv",
        default=None,
        help="CSV path for per-image, per-model, per-class inference timings",
    )
    parser.add_argument(
        "--combined-timing-csv",
        default=None,
        help="CSV path for per-image, per-model combined multi-class inference timings and FPS",
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = Path(args.timing_csv) if args.timing_csv else out_dir / "timing_by_image_model_class.csv"
    combined_timing_csv = (
        Path(args.combined_timing_csv)
        if args.combined_timing_csv
        else out_dir / "timing_by_image_model_combined.csv"
    )
    timing_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_timing_csv.parent.mkdir(parents=True, exist_ok=True)

    baseline = YOLO("models/baseline_best.pt")
    bifpn = YOLO("models/bifpn_best.pt")
    y8 = YOLO("yolov8n.pt")

    if args.image:
        image_list = [Path(args.image)]
    else:
        all_images = list(iter_images(images_dir))
        if args.sample is not None and args.sample < len(all_images):
            rng = random.Random(args.seed)
            image_list = rng.sample(all_images, args.sample)
        else:
            image_list = all_images

    if image_list and args.warmup > 0:
        warmup_image = sorted(image_list)[0]
        warmup_model("baseline", baseline, warmup_image, args.conf, args.imgsz, args.warmup)
        warmup_model("bifpn", bifpn, warmup_image, args.conf, args.imgsz, args.warmup)
        warmup_model("yolov8n", y8, warmup_image, args.conf, args.imgsz, args.warmup)

    timing_rows: List[Dict[str, object]] = []
    combined_timing_rows: List[Dict[str, object]] = []

    for idx, image_path in enumerate(sorted(image_list)):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_norm = load_gt_xyxy_norm(label_path, GT_CLASSES)
        gt_boxes = [norm_to_pixel(b, w, h) for _, b in gt_norm]
        gt_labels = [GT_CLASSES[cls] for cls, _ in gt_norm]

        base_boxes, base_labels, base_rows, base_counts = predict_model_classes(
            "baseline", baseline, image_path, args.conf, args.imgsz
        )
        bifpn_boxes, bifpn_labels, bifpn_rows, bifpn_counts = predict_model_classes(
            "bifpn", bifpn, image_path, args.conf, args.imgsz
        )
        y8_boxes, y8_labels, y8_rows, y8_counts = predict_model_classes(
            "yolov8n", y8, image_path, args.conf, args.imgsz
        )
        _, _, base_combined_row, _ = predict_model_combined_timed(
            "baseline", baseline, image_path, args.conf, args.imgsz
        )
        _, _, bifpn_combined_row, _ = predict_model_combined_timed(
            "bifpn", bifpn, image_path, args.conf, args.imgsz
        )
        _, _, y8_combined_row, _ = predict_model_combined_timed(
            "yolov8n", y8, image_path, args.conf, args.imgsz
        )
        timing_rows.extend(base_rows)
        timing_rows.extend(bifpn_rows)
        timing_rows.extend(y8_rows)
        combined_timing_rows.extend([base_combined_row, bifpn_combined_row, y8_combined_row])

        gt_counts = {name: 0 for name in GT_CLASSES.values()}
        for cls, _ in gt_norm:
            gt_counts[GT_CLASSES[cls]] += 1

        p_gt = draw_boxes(image, gt_boxes, (0, 255, 255), gt_labels, f"GT: {count_summary(gt_counts)}")
        p_base = draw_boxes(image, base_boxes, (0, 255, 0), base_labels, f"baseline: {count_summary(base_counts)}")
        p_bifpn = draw_boxes(image, bifpn_boxes, (255, 0, 0), bifpn_labels, f"bifpn: {count_summary(bifpn_counts)}")
        p_y8 = draw_boxes(image, y8_boxes, (0, 128, 255), y8_labels, f"yolov8n: {count_summary(y8_counts)}")

        top = cv2.hconcat([p_gt, p_base])
        bottom = cv2.hconcat([p_bifpn, p_y8])
        canvas = cv2.vconcat([top, bottom])

        cv2.putText(
            canvas,
            image_path.name,
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        out_path = out_dir / f"{idx:04d}_{image_path.stem}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"[saved] {out_path}")

    if timing_rows:
        with timing_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image", "model", "class_id", "class_name", "elapsed_ms", "detections"],
            )
            writer.writeheader()
            writer.writerows(timing_rows)
        print(f"[timing] {timing_csv}")

    if combined_timing_rows:
        with combined_timing_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
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


if __name__ == "__main__":
    main()
