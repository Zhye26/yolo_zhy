#!/usr/bin/env python3
"""
Manual review tool for YOLO labels.

Features:
- Verify image/label pairing (missing/orphan files).
- Visualize bounding boxes for manual inspection.
- Mark each image as: ok / suspect / skip.
- Save review records to CSV and summary JSON.
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LabelObject:
    class_id: int
    x: float
    y: float
    w: float
    h: float


def parse_classes(classes_file: Optional[Path]) -> Dict[int, str]:
    if not classes_file or not classes_file.exists():
        return {}
    names: Dict[int, str] = {}
    with classes_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                names[idx] = name
    return names


def find_pairs(images_dir: Path, labels_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    labels = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    image_by_stem = {p.stem: p for p in images}
    label_by_stem = {p.stem: p for p in labels}

    pairs: List[Tuple[Path, Path]] = []
    missing_label_images: List[Path] = []
    orphan_labels: List[Path] = []

    for stem, img_path in image_by_stem.items():
        lbl = label_by_stem.get(stem)
        if lbl is None:
            missing_label_images.append(img_path)
        else:
            pairs.append((img_path, lbl))

    for stem, lbl_path in label_by_stem.items():
        if stem not in image_by_stem:
            orphan_labels.append(lbl_path)

    return pairs, missing_label_images, orphan_labels


def parse_yolo_label(label_path: Path) -> Tuple[List[LabelObject], List[str]]:
    objects: List[LabelObject] = []
    bad_lines: List[str] = []
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return objects, bad_lines

    for i, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            bad_lines.append(f"line {i}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            bad_lines.append(f"line {i}: parse error")
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            bad_lines.append(f"line {i}: coords out of range [0,1]")
        objects.append(LabelObject(cls, x, y, w, h))

    return objects, bad_lines


def yolo_to_xyxy(obj: LabelObject, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1 = int((obj.x - obj.w / 2.0) * img_w)
    y1 = int((obj.y - obj.h / 2.0) * img_h)
    x2 = int((obj.x + obj.w / 2.0) * img_w)
    y2 = int((obj.y + obj.h / 2.0) * img_h)
    return x1, y1, x2, y2


def draw_overlay(
    image,
    image_path: Path,
    label_path: Path,
    objects: List[LabelObject],
    class_names: Dict[int, str],
    index: int,
    total: int,
):
    vis = image.copy()
    h, w = vis.shape[:2]

    for obj in objects:
        x1, y1, x2, y2 = yolo_to_xyxy(obj, w, h)
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cls_name = class_names.get(obj.class_id, str(obj.class_id))
        label = f"{obj.class_id}:{cls_name}"
        cv2.putText(vis, label, (max(0, x1), max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    tip_lines = [
        f"[{index + 1}/{total}] {image_path.name}",
        f"label: {label_path.name}  boxes: {len(objects)}",
        "keys: a=ok  s=suspect  d=skip  q=quit",
    ]
    y = 25
    for line in tip_lines:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        y += 28
    return vis


def review_pairs(
    pairs: List[Tuple[Path, Path]],
    class_names: Dict[int, str],
    out_dir: Path,
    progress_path: Path,
    start_index: int = 0,
    sample: Optional[int] = None,
    random_seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "review_records.csv"
    summary_path = out_dir / "review_summary.json"

    if sample is not None and sample < len(pairs):
        rng = random.Random(random_seed)
        pairs = rng.sample(pairs, sample)
        start_index = 0

    decisions: List[Dict[str, str]] = []
    stats = {"ok": 0, "suspect": 0, "skip": 0}
    last_shown_index = start_index
    last_shown_image = ""

    def write_progress(next_start: int, finished: bool, last_decision_index: Optional[int] = None, last_decision_image: str = ""):
        payload = {
            "generated_at": datetime.now().isoformat(),
            "finished": finished,
            "total_pairs": len(pairs),
            "start_index": start_index,
            "next_start": next_start,
            "reviewed_count": len(decisions),
            "stats": stats,
            "last_shown_index": last_shown_index,
            "last_shown_image": last_shown_image,
            "last_decision_index": last_decision_index,
            "last_decision_image": last_decision_image,
            "records_csv": str(csv_path),
        }
        progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    write_progress(next_start=start_index, finished=False)

    for idx in range(start_index, len(pairs)):
        image_path, label_path = pairs[idx]
        last_shown_index = idx
        last_shown_image = str(image_path)
        write_progress(next_start=idx, finished=False)
        image = cv2.imread(str(image_path))
        if image is None:
            decisions.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "status": "suspect",
                    "reason": "cannot_read_image",
                }
            )
            stats["suspect"] += 1
            write_progress(next_start=idx + 1, finished=False, last_decision_index=idx, last_decision_image=str(image_path))
            continue

        objects, bad_lines = parse_yolo_label(label_path)
        vis = draw_overlay(
            image=image,
            image_path=image_path,
            label_path=label_path,
            objects=objects,
            class_names=class_names,
            index=idx,
            total=len(pairs),
        )
        if bad_lines:
            cv2.putText(
                vis,
                f"bad label lines: {len(bad_lines)} (auto mark suspect if you press 'a')",
                (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        cv2.imshow("review_labels", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            write_progress(next_start=idx, finished=False)
            break
        if key == ord("a"):
            status = "ok" if not bad_lines else "suspect"
        elif key == ord("s"):
            status = "suspect"
        else:
            status = "skip"

        stats[status] += 1
        decisions.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "status": status,
                "boxes": str(len(objects)),
                "bad_lines": "|".join(bad_lines),
            }
        )
        write_progress(next_start=idx + 1, finished=False, last_decision_index=idx, last_decision_image=str(image_path))

    cv2.destroyAllWindows()

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["image", "label", "status", "boxes", "bad_lines", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in decisions:
            writer.writerow(row)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "reviewed": len(decisions),
        "stats": stats,
        "records_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(next_start=min(len(pairs), start_index + len(decisions)), finished=(start_index + len(decisions) >= len(pairs)))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Manual review for image-label pairing and bbox alignment")
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--labels", required=True, help="Labels directory")
    parser.add_argument("--classes", default="data/classes.txt", help="Optional classes.txt path")
    parser.add_argument("--out", default="data/review_results", help="Output directory for reports")
    parser.add_argument("--progress-file", default=None, help="Optional progress JSON path")
    parser.add_argument("--start", type=int, default=0, help="Start index for review")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size for quick inspection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out)
    progress_path = Path(args.progress_file) if args.progress_file else out_dir / "review_progress.json"
    class_names = parse_classes(Path(args.classes) if args.classes else None)

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit("images/labels directory does not exist")

    pairs, missing_label_images, orphan_labels = find_pairs(images_dir, labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_report = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "paired_count": len(pairs),
        "missing_label_images": [str(p) for p in missing_label_images],
        "orphan_labels": [str(p) for p in orphan_labels],
    }
    (out_dir / "pairing_report.json").write_text(json.dumps(pair_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"paired: {len(pairs)}")
    print(f"missing label for image: {len(missing_label_images)}")
    print(f"orphan label (no image): {len(orphan_labels)}")
    print(f"pairing report: {out_dir / 'pairing_report.json'}")

    if not pairs:
        print("No image-label pairs to review.")
        return

    review_pairs(
        pairs=pairs,
        class_names=class_names,
        out_dir=out_dir,
        progress_path=progress_path,
        start_index=max(0, args.start),
        sample=args.sample,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
