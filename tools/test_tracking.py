#!/usr/bin/env python3
"""
Tracking and violation regression runner for local videos.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import EbikeDetector
from app.services.video_processor import VideoProcessor


def build_detector(args):
    """Build detector based on CLI args."""
    detector = EbikeDetector(
        model_path=args.model or "models/bifpn_best.pt",
        task_mode=args.task,
    )
    detector.load_model()
    return detector


def test_tracking(video_path, args, output_path=None):
    """Run video regression with and without tracking."""
    print(f"Video: {video_path}")
    print(f"Model: {args.model or 'models/bifpn_best.pt'}")
    print("Pipeline: YOLO")
    print(f"Task: {args.task}")
    print("-" * 50)

    detector = build_detector(args)
    processor = VideoProcessor(detector)

    print("\n[1] Processing with tracking...")
    result_tracking = processor.process_video(
        video_path,
        output_path=output_path,
        use_tracking=True,
        callback=lambda progress: print(f"  Progress: {progress:.1f}%", end="\r"),
    )
    print()

    print("\n[2] Processing without tracking...")
    detector.reset_tracker()
    result_no_tracking = processor.process_video(
        video_path,
        output_path=output_path.replace(".mp4", "_no_track.mp4") if output_path else None,
        use_tracking=False,
        callback=lambda progress: print(f"  Progress: {progress:.1f}%", end="\r"),
    )
    print()

    print("\n" + "=" * 50)
    print("Comparison:")
    print("=" * 50)
    print(f"Total frames: {result_tracking['total_frames']}")
    print(f"FPS: {result_tracking['fps']}")
    print()
    print(f"{'Metric':<20} {'No track':<15} {'Track':<15} {'Delta'}")
    print("-" * 60)

    no_track_count = len(result_no_tracking["violations"])
    track_all_count = len(result_tracking["violations"])
    track_unique_count = len(result_tracking["unique_violations"])

    print(f"{'All violations':<20} {no_track_count:<15} {track_all_count:<15} -")
    print(f"{'Unique violations':<20} {no_track_count:<15} {track_unique_count:<15} {no_track_count - track_unique_count}")

    if no_track_count > 0:
        reduction = (1 - track_unique_count / no_track_count) * 100
        print(f"\nDedup reduction: {reduction:.1f}%")

    print("\nBy type:")
    print("-" * 40)

    types_no_track = {}
    for violation in result_no_tracking["violations"]:
        violation_type = violation["type"]
        types_no_track[violation_type] = types_no_track.get(violation_type, 0) + 1

    types_track = {}
    for violation in result_tracking["unique_violations"]:
        violation_type = violation["type"]
        types_track[violation_type] = types_track.get(violation_type, 0) + 1

    all_types = set(types_no_track.keys()) | set(types_track.keys())
    for violation_type in sorted(all_types):
        no_track_total = types_no_track.get(violation_type, 0)
        track_total = types_track.get(violation_type, 0)
        print(f"  {violation_type}: {no_track_total} -> {track_total} ({no_track_total - track_total})")

    print(f"\nOutput video: {result_tracking['output_path']}")
    return result_tracking, result_no_tracking


def main():
    parser = argparse.ArgumentParser(description="Tracking regression runner")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--model", "-m", default="models/bifpn_best.pt", help="Model path")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--task", choices=["ebike", "all", "helmet", "passenger"], default="ebike", help="Violation task mode")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video does not exist: {args.video}")
        sys.exit(1)

    test_tracking(args.video, args, args.output)


if __name__ == "__main__":
    main()
