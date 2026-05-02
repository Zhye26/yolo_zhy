"""
Video stream processor.
Handles video file and stream processing using the detection pipeline.
"""
import cv2
import os
from typing import Optional, Callable, Dict, Generator
from app.services.detector import EbikeDetector
from app.config import settings


class VideoProcessor:
    """Video stream processor."""

    def __init__(self, detector: EbikeDetector):
        self.detector = detector
        self.output_folder = settings.storage.output_folder

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        callback: Optional[Callable[[float], None]] = None,
        use_tracking: bool = True
    ) -> Dict:
        """
        Process video file.

        Args:
            video_path: Input video path
            output_path: Output video path
            callback: Progress callback function
            use_tracking: Whether to use tracking (always True with new pipeline)

        Returns:
            Processing result dict
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            basename = os.path.basename(video_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(self.output_folder, f"{name}_result{ext}")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.detector.reset_tracker()

        all_violations = []
        unique_violations = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps
            detections = self.detector.detect(frame)
            violations, new_violations = self.detector.detect_violations(
                detections, use_tracking=use_tracking
            )

            for v in violations:
                v['frame_number'] = frame_number
                v['timestamp'] = timestamp
                all_violations.append(v)

            for v in new_violations:
                v['frame_number'] = frame_number
                v['timestamp'] = timestamp
                unique_violations.append(v)

            result_frame = self.detector.draw_results(frame, detections, violations)
            out.write(result_frame)

            frame_number += 1

            if callback and frame_number % 30 == 0:
                progress = frame_number / total_frames * 100
                callback(progress)

        cap.release()
        out.release()

        return {
            'output_path': output_path,
            'violations': all_violations,
            'unique_violations': unique_violations,
            'total_frames': total_frames,
            'fps': fps,
            'tracking_enabled': use_tracking,
            'stats': self.detector.get_stats(),
        }

    def process_stream(
        self,
        stream_url: str,
        use_tracking: bool = True
    ) -> Generator[bytes, None, None]:
        """Process live video stream (generator)."""
        cap = cv2.VideoCapture(stream_url)
        self.detector.reset_tracker()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            violations, _ = self.detector.detect_violations(
                detections, use_tracking=use_tracking
            )
            result_frame = self.detector.draw_results(frame, detections, violations)

            _, buffer = cv2.imencode('.jpg', result_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
