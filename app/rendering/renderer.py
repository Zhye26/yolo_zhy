"""
Detection result renderer.
Handles drawing detections, tracks, and violations on frames.
"""
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    FONT_CANDIDATES = [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
        Path('/usr/share/fonts/truetype/arphic/ukai.ttc'),
    ]

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or settings.detection.class_names
        self._font_path = self._resolve_font_path()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

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
            if det.class_id != settings.detection.ebike_class_id:
                continue
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
            if det.class_id != settings.detection.ebike_class_id:
                continue
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
            if v.track_id is not None and v.track_id >= 0:
                label = f"ID:{v.track_id} {label}"

            self._draw_label(
                frame,
                label=label,
                origin=(x1, y2),
                bg_color=self.VIOLATION_COLOR,
                text_color=(255, 255, 255),
                font_scale=0.6,
                thickness=2,
                prefer_unicode=True,
            )

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        origin: tuple[int, int],
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        font_scale: float,
        thickness: int,
        prefer_unicode: bool = False,
    ) -> None:
        if prefer_unicode and any(ord(char) > 127 for char in label) and self._font_path is not None:
            self._draw_unicode_label(frame, label, origin, bg_color, text_color, font_scale)
            return

        x, y = origin
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x, y), (x + tw, y + th + 8), bg_color, -1)
        cv2.putText(frame, label, (x, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    def _draw_unicode_label(
        self,
        frame: np.ndarray,
        label: str,
        origin: tuple[int, int],
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        font_scale: float,
    ) -> None:
        font_size = max(16, int(26 * font_scale))
        font = self._get_font(font_size)
        if font is None:
            safe_label = label.encode('ascii', errors='ignore').decode('ascii') or 'violation'
            self._draw_label(frame, safe_label, origin, bg_color, text_color, font_scale, 2, prefer_unicode=False)
            return

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        x, y = origin
        left, top, right, bottom = draw.textbbox((x, y), label, font=font)
        padding_x = 6
        padding_y = 4
        draw.rectangle(
            (left - padding_x, top - padding_y, right + padding_x, bottom + padding_y),
            fill=(bg_color[2], bg_color[1], bg_color[0]),
        )
        draw.text((x, y), label, font=font, fill=(text_color[2], text_color[1], text_color[0]))
        frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _resolve_font_path(self) -> Optional[Path]:
        for path in self.FONT_CANDIDATES:
            if path.exists():
                return path
        return None

    def _get_font(self, font_size: int) -> Optional[ImageFont.FreeTypeFont]:
        if self._font_path is None:
            return None
        if font_size not in self._font_cache:
            self._font_cache[font_size] = ImageFont.truetype(str(self._font_path), font_size)
        return self._font_cache[font_size]
