#!/usr/bin/env python3
"""
Compare ebike detection ability across models on a YOLO-format test split.

Rules:
- baseline_best.pt / bifpn_best.pt: use class 0 (ebike)
- yolov8n.pt: use class 1 + 3 (bicycle + motorcycle) as ebike proxy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EvalResult:
    model_name: str
    tp: int
    fp: int
    fn: int
    total_gt: int
    total_pred: int
    images: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def load_gt_boxes(label_path: Path, target_class: int = 0) -> List[List[float]]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    boxes: List[List[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls != target_class:
            continue
        x, y, w, h = map(float, parts[1:5])
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])  # normalized xyxy
    return boxes


def iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(pred_boxes: List[List[float]], gt_boxes: List[List[float]], iou_thr: float) -> Tuple[int, int, int]:
    matched_gt = set()
    matched_pred = set()
    candidates = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            score = iou(pb, gb)
            if score >= iou_thr:
                candidates.append((score, pi, gi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    tp = 0
    for _, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def evaluate_model(
    model_path: Path,
    images_dir: Path,
    labels_dir: Path,
    inference_classes: List[int],
    conf_thr: float,
    iou_thr: float,
    imgsz: int,
) -> EvalResult:
    model = YOLO(str(model_path))
    tp = fp = fn = 0
    total_gt = total_pred = 0
    image_count = 0

    for img_path in iter_images(images_dir):
        image_count += 1
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_gt_boxes(lbl_path, target_class=0)

        results = model.predict(
            source=str(img_path),
            classes=inference_classes,
            conf=conf_thr,
            iou=0.7,
            imgsz=imgsz,
            verbose=False,
        )
        pred_boxes: List[List[float]] = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            h, w = r.orig_shape
            for x1, y1, x2, y2 in xyxy:
                pred_boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])

        im_tp, im_fp, im_fn = greedy_match(pred_boxes, gt_boxes, iou_thr=iou_thr)
        tp += im_tp
        fp += im_fp
        fn += im_fn
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

    return EvalResult(
        model_name=model_path.name,
        tp=tp,
        fp=fp,
        fn=fn,
        total_gt=total_gt,
        total_pred=total_pred,
        images=image_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ebike detection across models")
    parser.add_argument("--images", default="data/merged_dataset/images/test")
    parser.add_argument("--labels", default="data/merged_dataset/labels/test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit("images/labels directory not found")

    eval_plan = [
        (Path("models/baseline_best.pt"), [0]),
        (Path("models/bifpn_best.pt"), [0]),
        (Path("yolov8n.pt"), [1, 3]),
    ]

    results: List[EvalResult] = []
    for model_path, infer_classes in eval_plan:
        if not model_path.exists():
            print(f"[skip] model not found: {model_path}")
            continue
        print(f"[run] {model_path.name} classes={infer_classes}")
        res = evaluate_model(
            model_path=model_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            inference_classes=infer_classes,
            conf_thr=args.conf,
            iou_thr=args.iou,
            imgsz=args.imgsz,
        )
        results.append(res)

    print("\n=== EBIKE COMPARISON ===")
    print(f"dataset images={results[0].images if results else 0}, gt_boxes(class0)={results[0].total_gt if results else 0}")
    print(f"conf={args.conf}, match_iou={args.iou}, imgsz={args.imgsz}")
    print("model,tp,fp,fn,precision,recall,f1,total_pred")
    for r in results:
        print(
            f"{r.model_name},{r.tp},{r.fp},{r.fn},"
            f"{r.precision:.4f},{r.recall:.4f},{r.f1:.4f},{r.total_pred}"
        )


if __name__ == "__main__":
    main()

