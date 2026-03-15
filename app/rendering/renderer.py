"""
Detection result renderer.
Handles drawing detections, tracks, and violations on frames.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
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
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/msyhbd.ttc'),
        Path('C:/Windows/Fonts/simhei.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
        Path('/usr/share/fonts/truetype/arphic/ukai.ttc'),
    ]

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or settings.detection.class_names
        self._font_path = self._resolve_font_path()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._track_to_display_id: Dict[int, int] = {}
        self._display_entities: Dict[int, Dict[str, Any]] = {}
        self._frame_index = 0
        self._frame_claimed_display_ids: set[int] = set()
        self._next_display_id = 1
        self._display_match_max_gap = 90

    def reset(self) -> None:
        """Reset per-video display state."""
        self._track_to_display_id = {}
        self._display_entities = {}
        self._frame_index = 0
        self._frame_claimed_display_ids = set()
        self._next_display_id = 1

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
        self._frame_index += 1
        self._frame_claimed_display_ids = set()
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
        ebike_tracks = [
            track for track in tracks
            if track.detection.class_id == settings.detection.ebike_class_id
        ]
        ebike_tracks.sort(key=lambda item: item.detection.area, reverse=True)

        for track in ebike_tracks:
            det = track.detection
            display_id = None
            if self._should_show_track_id(track):
                display_id = self._display_track_id(track.track_id, det.bbox)
            self._draw_box(
                frame,
                det.bbox,
                det.class_id,
                det.confidence,
                track_id=display_id,
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
                label = f"ID:{self._display_track_id(v.track_id, v.bbox, force=True)} {label}"

            label_y = max(0, y1 - 30) if y2 >= frame.shape[0] - 36 else y2
            self._draw_label(
                frame,
                label=label,
                origin=(x1, label_y),
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

        x, y = self._clamp_label_origin(frame, origin)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = max(0, min(x, frame.shape[1] - tw - 1))
        y = max(0, min(y, frame.shape[0] - th - 9))
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
        x, y = self._clamp_label_origin(frame, origin)
        left, top, right, bottom = draw.textbbox((x, y), label, font=font)
        padding_x = 6
        padding_y = 4
        box_w = (right - left) + padding_x * 2
        box_h = (bottom - top) + padding_y * 2
        x = max(0, min(x, frame.shape[1] - box_w - 1))
        y = max(0, min(y, frame.shape[0] - box_h - 1))
        left, top, right, bottom = draw.textbbox((x, y), label, font=font)
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

    def _display_track_id(
        self,
        track_id: int,
        bbox: List[float],
        force: bool = False,
    ) -> int:
        if track_id in self._track_to_display_id:
            display_id = self._track_to_display_id[track_id]
            self._touch_display_entity(display_id, bbox)
            self._frame_claimed_display_ids.add(display_id)
            return display_id

        display_id = self._match_display_entity(bbox, force=force)
        if display_id is None:
            display_id = self._next_display_id
            self._next_display_id += 1

        self._track_to_display_id[track_id] = display_id
        self._touch_display_entity(display_id, bbox)
        self._frame_claimed_display_ids.add(display_id)
        return display_id

    def _match_display_entity(self, bbox: List[float], force: bool) -> Optional[int]:
        best_display_id: Optional[int] = None
        best_score = 0.0

        for display_id, entity in self._display_entities.items():
            if display_id in self._frame_claimed_display_ids:
                continue
            if self._frame_index - int(entity["last_seen"]) > self._display_match_max_gap:
                continue

            prev_bbox = entity["bbox"]
            iou = self._iou(bbox, prev_bbox)
            overlap = self._overlap_ratio(bbox, prev_bbox)
            center_score = self._center_proximity_score(bbox, prev_bbox)
            score = iou * 0.65 + overlap * 0.20 + center_score * 0.15
            min_score = 0.18 if force else 0.24
            if score >= min_score and score > best_score:
                best_score = score
                best_display_id = display_id

        return best_display_id

    def _touch_display_entity(self, display_id: int, bbox: List[float]) -> None:
        self._display_entities[display_id] = {
            "bbox": list(bbox),
            "last_seen": self._frame_index,
        }

    def _should_show_track_id(self, track: Track) -> bool:
        bbox = track.detection.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        box_bottom = bbox[3]
        min_hits = 6
        if box_bottom >= 440.0 and width >= 120.0 and area >= 18000.0:
            min_hits = 3
        elif box_bottom >= 420.0 and width >= 100.0 and area >= 14000.0:
            min_hits = 4
        if track.hits < min_hits:
            return False
        return (box_bottom >= 430.0 and (width >= 90 or height >= 90 or area >= 12000.0)) or area >= 45000.0

    def _center_proximity_score(self, box_a: List[float], box_b: List[float]) -> float:
        ax = (box_a[0] + box_a[2]) / 2
        ay = (box_a[1] + box_a[3]) / 2
        bx = (box_b[0] + box_b[2]) / 2
        by = (box_b[1] + box_b[3]) / 2
        aw = max(1.0, box_a[2] - box_a[0])
        ah = max(1.0, box_a[3] - box_a[1])
        bw = max(1.0, box_b[2] - box_b[0])
        bh = max(1.0, box_b[3] - box_b[1])
        norm = max((aw + bw) * 0.5, (ah + bh) * 0.5, 1.0)
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return max(0.0, 1.0 - distance / (norm * 1.5))

    def _iou(self, box_a: List[float], box_b: List[float]) -> float:
        inter_area = self._intersection_area(box_a, box_b)
        if inter_area <= 0:
            return 0.0
        area_a = self._area(box_a)
        area_b = self._area(box_b)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def _overlap_ratio(self, box_a: List[float], box_b: List[float]) -> float:
        inter_area = self._intersection_area(box_a, box_b)
        if inter_area <= 0:
            return 0.0
        return inter_area / max(1.0, min(self._area(box_a), self._area(box_b)))

    def _intersection_area(self, box_a: List[float], box_b: List[float]) -> float:
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])
        return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    def _area(self, bbox: List[float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _clamp_label_origin(self, frame: np.ndarray, origin: tuple[int, int]) -> tuple[int, int]:
        x, y = origin
        return max(0, x), max(0, y)
