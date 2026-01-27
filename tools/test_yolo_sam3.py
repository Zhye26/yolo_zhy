#!/usr/bin/env python3
"""
Test script for YOLO + SAM3 cascade detection.
Demonstrates segmentation visualization with colored masks.
"""
import sys
import os
import cv2
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import YoloSam3Detector


def test_image(detector: YoloSam3Detector, image_path: str, output_path: str):
    """Test on a single image."""
    print(f"Processing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    # Run cascade detection
    start = time.time()
    result = detector.detect(image)
    detect_time = time.time() - start

    # Draw results
    start = time.time()
    output = detector.draw_results(image, result)
    render_time = time.time() - start

    # Save output
    cv2.imwrite(output_path, output)

    # Print stats
    n_det = len(result.frame_result.detections)
    n_seg = len(result.segmentations)
    print(f"  Detections: {n_det}, Segmentations: {n_seg}")
    print(f"  Detection time: {detect_time*1000:.1f}ms")
    print(f"  Render time: {render_time*1000:.1f}ms")
    print(f"  Output saved: {output_path}")


def test_video(detector: YoloSam3Detector, video_path: str, output_path: str):
    """Test on a video file."""
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0

    detector.reset_tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        result = detector.detect(frame)
        output = detector.draw_results(frame, result)
        elapsed = time.time() - start
        total_time += elapsed

        out.write(output)
        frame_count += 1

        if frame_count % 30 == 0:
            avg_fps = frame_count / total_time
            print(f"  Frame {frame_count}/{total_frames}, FPS: {avg_fps:.1f}")

    cap.release()
    out.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test YOLO+SAM3 cascade")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--model", default="models/bifpn_best.pt",
                        help="YOLO model path")
    parser.add_argument("--sam3", default="/home/ubuntu/SAM3/sam3.pt",
                        help="SAM3 checkpoint path")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Mask transparency (0-1)")
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_sam3{ext}"

    # Initialize detector
    print("Loading YOLO + SAM3 models...")
    detector = YoloSam3Detector(
        model_path=args.model,
        sam3_checkpoint=args.sam3,
        mask_alpha=args.alpha,
    )

    if not detector.load_model():
        print("Failed to load models")
        sys.exit(1)

    print("Models loaded successfully")

    # Process input
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        test_video(detector, args.input, output_path)
    else:
        test_image(detector, args.input, output_path)


if __name__ == "__main__":
    main()
