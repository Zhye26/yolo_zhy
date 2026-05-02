"""
E-bike violation detector service.
Facade for the detection pipeline, maintaining backward compatibility.
"""
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.core import FramePipeline, FrameResult, PipelineConfig
from app.rendering import DetectionRenderer


class EbikeDetector:
    """E-bike violation detector with pipeline backend."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_tensorrt: bool = False,
        task_mode: str = "ebike",
    ):
        self.model_path = model_path or str(settings.model.model_path)
        self.use_tensorrt = use_tensorrt
        self.task_mode = task_mode
        self.class_names = settings.detection.class_names

        self._pipeline: Optional[FramePipeline] = None
        self._renderer: Optional[DetectionRenderer] = None
        self._last_result: Optional[FrameResult] = None

    def load_model(self) -> bool:
        """Load detection model."""
        try:
            with self._task_settings():
                config = PipelineConfig(
                    use_tensorrt=self.use_tensorrt,
                    enable_tracking=True,
                    enable_rules=True,
                    enable_dedup=True,
                )
                self._pipeline = FramePipeline(config)
                self._pipeline.initialize(self.model_path)
                self._renderer = DetectionRenderer(self._pipeline.class_names)
            return True
        except Exception as e:
            print(f"妯″瀷鍔犺浇澶辫触: {e}")
            return False

    def detect(self, image: np.ndarray) -> "DetectionResult":
        """
        Detect objects in a single image.
        Returns a wrapper for backward compatibility.
        """
        if self._pipeline is None:
            if not self.load_model():
                raise RuntimeError("Failed to initialize detection pipeline.")

        with self._task_settings():
            result = self._pipeline.process(image)
        self._last_result = result
        return DetectionResult(result, self.class_names)

    def track(self, image: np.ndarray, persist: bool = True) -> "DetectionResult":
        """
        Detect and track objects.
        Same as detect() since pipeline always tracks.
        """
        return self.detect(image)

    def reset_tracker(self) -> None:
        """Reset tracker state for new video."""
        if self._pipeline:
            with self._task_settings():
                self._pipeline.reset()
        if self._renderer:
            self._renderer.reset()

    def detect_violations(
        self,
        detections: "DetectionResult",
        use_tracking: bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get violations from detection result.
        Returns (all_violations, new_violations) for backward compatibility.
        """
        result = detections._frame_result

        all_violations = [
            self._violation_to_dict(v)
            for v in result.active_violations
        ]
        new_violations = [
            self._violation_to_dict(v)
            for v in result.new_violations
        ]

        return all_violations, new_violations

    def _violation_to_dict(self, v) -> Dict:
        """Convert ViolationEvent to legacy dict format."""
        violation_type = "overload" if v.violation_type == v.violation_type.PASSENGER else v.violation_type.name.lower()
        return {
            "type": violation_type,
            "bbox": v.bbox,
            "confidence": v.confidence,
            "track_id": v.track_id,
            "description": v.description,
        }

    def draw_results(
        self,
        image: np.ndarray,
        detections: "DetectionResult",
        violations: List[Dict],
    ) -> np.ndarray:
        """Draw detection results on image."""
        if self._renderer is None:
            self._renderer = DetectionRenderer(self.class_names)

        return self._renderer.render(
            image,
            detections._frame_result,
            show_tracks=True,
            show_violations=True,
        )

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        if self._pipeline:
            return self._pipeline.get_stats()
        return {}

    @contextmanager
    def _task_settings(self) -> Iterator[None]:
        original_passenger_rule = settings.rules.passenger_rule_enabled
        original_helmet_rule = settings.rules.helmet_rule_enabled
        original_helmet_detection = settings.detection.helmet_detection_enabled
        try:
            if self.task_mode == "ebike":
                settings.rules.passenger_rule_enabled = False
                settings.rules.helmet_rule_enabled = False
                settings.detection.helmet_detection_enabled = False
            elif self.task_mode == "helmet":
                settings.rules.passenger_rule_enabled = False
                settings.rules.helmet_rule_enabled = True
                settings.detection.helmet_detection_enabled = True
            elif self.task_mode == "passenger":
                settings.rules.passenger_rule_enabled = True
                settings.rules.helmet_rule_enabled = False
                settings.detection.helmet_detection_enabled = False
            else:
                settings.rules.passenger_rule_enabled = True
                settings.rules.helmet_rule_enabled = True
                settings.detection.helmet_detection_enabled = True
            yield
        finally:
            settings.rules.passenger_rule_enabled = original_passenger_rule
            settings.rules.helmet_rule_enabled = original_helmet_rule
            settings.detection.helmet_detection_enabled = original_helmet_detection


class DetectionResult:
    """Wrapper for backward compatibility with YOLO result interface."""

    def __init__(self, frame_result: FrameResult, class_names: List[str]):
        self._frame_result = frame_result
        self._class_names = class_names

    @property
    def boxes(self) -> Optional["BoxesWrapper"]:
        """Return boxes wrapper for backward compatibility."""
        if not self._frame_result.tracks and not self._frame_result.detections:
            return None
        return BoxesWrapper(self._frame_result)


class BoxesWrapper:
    """Wrapper to mimic YOLO boxes interface."""

    def __init__(self, frame_result: FrameResult):
        self._result = frame_result

    def __len__(self) -> int:
        if self._result.tracks:
            return len(self._result.tracks)
        return len(self._result.detections)

    def __iter__(self):
        if self._result.tracks:
            for track in self._result.tracks:
                yield BoxItem(track.detection, track.track_id)
        else:
            for det in self._result.detections:
                yield BoxItem(det, None)


class BoxItem:
    """Single box item wrapper."""

    def __init__(self, detection, track_id: Optional[int]):
        self._det = detection
        self._track_id = track_id

    @property
    def cls(self):
        return [self._det.class_id]

    @property
    def conf(self):
        return [self._det.confidence]

    @property
    def xyxy(self):
        import torch

        return [torch.tensor(self._det.bbox)]

    @property
    def id(self):
        if self._track_id is not None and self._track_id >= 0:
            return [self._track_id]
        return None
