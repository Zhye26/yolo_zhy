"""
Detection result renderer.
Handles drawing detections, tracks, and violations on frames.
"""
from typing import List, Dict, Optional
import cv2
import numpy as np
from app.core.types import Detection, Track, ViolationEvent, FrameResult
from app.config import settings


class DetectionRenderer:
    """Renders detection results on frames."""

    CLASS_COLORS = {
        0: (0, 255, 0),    # ebike - green
        1: (255, 0, 0),    # driver - blue
        2: (0, 165, 255),  # passenger - orange
        3: (255, 255, 0),  # helmet - cyan
    }
    VIOLATION_COLOR = (0, 0, 255)  # red

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or settings.detection.class_names

    def render(
        self,
        frame: np.ndarray,
        result: FrameResult,
        show_tracks: bool = True,
        show_violations: bool = True,
    ) -> np.ndarray:
        """
        Render detection results on frame.

        Args:
            frame: BGR image
            result: FrameResult from pipeline
            show_tracks: Whether to show track IDs
            show_violations: Whether to highlight violations

        Returns:
            Rendered frame
        """
        output = frame.copy()

        if show_tracks and result.tracks:
            self._draw_tracks(output, result.tracks)
        elif result.detections:
            self._draw_detections(output, result.detections)

        if show_violations and result.active_violations:
            self._draw_violations(output, result.active_violations)

        return output

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> None:
        """Draw detection boxes."""
        for det in detections:
            self._draw_box(
                frame,
                det.bbox,
                det.class_id,
                det.confidence,
                track_id=None,
            )

    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> None:
        """Draw tracked detection boxes."""
        for track in tracks:
            det = track.detection
            self._draw_box(
                frame,
                det.bbox,
                det.class_id,
                det.confidence,
                track_id=track.track_id if track.track_id >= 0 else None,
            )

    def _draw_box(
        self,
        frame: np.ndarray,
        bbox: List[float],
        class_id: int,
        confidence: float,
        track_id: Optional[int] = None,
    ) -> None:
        """Draw a single detection box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = self.CLASS_COLORS.get(class_id, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"cls{class_id}"
        label = f"{class_name} {confidence:.2f}"
        if track_id is not None:
            label = f"ID:{track_id} {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _draw_violations(self, frame: np.ndarray, violations: List[ViolationEvent]) -> None:
        """Draw violation highlights."""
        for v in violations:
            x1, y1, x2, y2 = [int(val) for val in v.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.VIOLATION_COLOR, 3)

            label = v.description
            if v.track_id is not None:
                label = f"ID:{v.track_id} {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw, y2 + th + 8), self.VIOLATION_COLOR, -1)
            cv2.putText(frame, label, (x1, y2 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
