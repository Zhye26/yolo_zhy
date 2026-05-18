#!/usr/bin/env python3
"""
Train a lightweight person-to-vehicle association classifier from corrected CSVs.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.rules.match_features import FEATURE_NAMES
except ModuleNotFoundError:
    # Keep this script usable in a lightweight labeling/training environment
    # where Flask/OpenCV are not installed.
    FEATURE_NAMES = [
        "person_center_relative_x",
        "person_center_relative_y",
        "person_bottom_relative_y",
        "bbox_iou",
        "lower_half_iou",
        "horizontal_overlap",
        "vertical_overlap",
        "overlap_ratio",
        "area_ratio",
        "height_ratio",
        "width_ratio",
        "bottom_distance_ratio",
        "center_distance_ratio",
        "person_conf",
        "vehicle_conf",
        "is_edge_vehicle",
        "legacy_match_score",
    ]


def load_dataset(csv_paths: List[str]) -> Tuple[List[List[float]], List[int]]:
    rows: List[List[float]] = []
    labels: List[int] = []

    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = [name for name in FEATURE_NAMES if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{csv_path} is missing feature columns: {missing}")

            for row in reader:
                raw_label = str(row.get("match_label", "")).strip()
                if raw_label == "":
                    continue
                label = _parse_label(raw_label)
                if label is None:
                    continue
                rows.append([float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES])
                labels.append(label)

    return rows, labels


def train(args) -> int:
    x, y = load_dataset(args.csv)
    if not x:
        print("No labeled rows found. Fill match_label with 0/1 before training.")
        return 1
    if len(set(y)) < 2:
        print("Training needs both positive and negative labels in match_label.")
        return 1

    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    stratify = y if min(y.count(0), y.count(1)) >= 2 else None
    if len(y) >= 10 and stratify is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
    else:
        x_train, x_test, y_train, y_test = x, [], y, []

    if args.model_type == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
        )
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or None,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
        )

    model.fit(x_train, y_train)

    print(f"Training rows: {len(x_train)}")
    print(f"Positive labels: {sum(y)}")
    print(f"Negative labels: {len(y) - sum(y)}")

    if x_test:
        predictions = model.predict(x_test)
        print("\nValidation report:")
        print(classification_report(y_test, predictions, digits=3))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
    else:
        print("\nValidation skipped: not enough labeled rows for a stratified split.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "threshold": args.threshold,
            "model_type": args.model_type,
        },
        output_path,
    )
    print(f"\nSaved classifier to {output_path}")
    return 0


def _parse_label(raw_label: str):
    normalized = raw_label.lower()
    if normalized in {"1", "true", "yes", "y", "match"}:
        return 1
    if normalized in {"0", "false", "no", "n", "not_match"}:
        return 0
    raise ValueError(f"Unsupported match_label value: {raw_label}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train person-vehicle match classifier")
    parser.add_argument("csv", nargs="+", help="Corrected CSV file(s) from export_match_samples.py")
    parser.add_argument("--output", "-o", default="models/person_vehicle_match.joblib", help="Output joblib path")
    parser.add_argument("--model-type", choices=["random_forest", "logistic"], default="random_forest")
    parser.add_argument("--threshold", type=float, default=0.55, help="Runtime match probability threshold")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
