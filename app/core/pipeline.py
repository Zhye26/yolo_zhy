"""
Single frame processing pipeline.
Orchestrates detection, tracking, rule evaluation, and violation deduplication.
"""
from typing import Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

from app.core.types import (
    FrameMeta,
    FrameContext,
    FrameResult,
    Detection,
    Track,
    ViolationEvent,
)
from app.inference import DetectorBackend, UltralyticsBackend, TensorRTBackend
from app.tracking import TrackManager
from app.rules import RuleEngine
from app.violations import ViolationDeduper
from app.config import settings


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    use_tensorrt: bool = False
    enable_tracking: bool = True
    enable_rules: bool = True
    enable_dedup: bool = True


class FramePipeline:
    """
    Single frame processing pipeline.

    Pipeline stages:
    1. Detection: Run YOLO inference
    2. Tracking: Assign track IDs via ByteTrack
    3. Rule evaluation: Check for violations
    4. Deduplication: Filter duplicate violations via FSM
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._detector: Optional[DetectorBackend] = None
        self._tracker: Optional[TrackManager] = None
        self._rule_engine: Optional[RuleEngine] = None
        self._deduper: Optional[ViolationDeduper] = None
        self._frame_count = 0
        self._initialized = False

    def initialize(self, model_path: str) -> None:
        """Initialize all pipeline components."""
        if self.config.use_tensorrt:
            self._detector = TensorRTBackend()
        else:
            self._detector = UltralyticsBackend()

        self._detector.load(model_path)
        self._detector.warmup()

        if self.config.enable_tracking:
            self._tracker = TrackManager()

        if self.config.enable_rules:
            self._rule_engine = RuleEngine()

        if self.config.enable_dedup:
            self._deduper = ViolationDeduper()

        self._initialized = True

    def reset(self) -> None:
        """Reset pipeline state for new video."""
        self._frame_count = 0
        if self._tracker:
            self._tracker.reset()
        if self._deduper:
            self._deduper.reset()
        if self._rule_engine:
            self._rule_engine.reset()
        detector_reset = getattr(self._detector, "reset", None)
        if callable(detector_reset):
            detector_reset()

    def process(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> FrameResult:
        """
        Process a single frame through the pipeline.

        Args:
            frame: BGR image as numpy array
            timestamp: Frame timestamp in seconds

        Returns:
            FrameResult with detections, tracks, and violations
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self._frame_count += 1
        meta = FrameMeta(
            frame_index=self._frame_count,
            timestamp=timestamp,
            width=frame.shape[1],
            height=frame.shape[0],
        )

        detections = self._detect(frame)
        tracks = self._track(detections, frame)
        context = FrameContext(
            frame=frame,
            meta=meta,
            detections=detections,
            tracks=tracks,
        )
        active_violations, new_violations = self._evaluate_violations(context)

        return FrameResult(
            meta=meta,
            detections=detections,
            tracks=tracks,
            active_violations=active_violations,
            new_violations=new_violations,
        )

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame."""
        return self._detector.predict(
            frame,
            conf_thresh=settings.detection.conf_thresh,
            iou_thresh=settings.detection.iou_thresh,
        )

    def _track(
        self,
        detections: List[Detection],
        frame: np.ndarray
    ) -> List[Track]:
        """Run tracking on detections."""
        if not self._tracker:
            return [Track(track_id=-1, detection=d) for d in detections]
        return self._tracker.update(detections, frame)

    def _evaluate_violations(
        self,
        context: FrameContext
    ) -> Tuple[List[ViolationEvent], List[ViolationEvent]]:
        """Evaluate rules and deduplicate violations."""
        if not self._rule_engine:
            return [], []

        candidates = self._rule_engine.evaluate(context)

        if not self._deduper:
            return [], []

        return self._deduper.update(candidates, context.meta)

    @property
    def class_names(self) -> List[str]:
        """Get class names from detector."""
        if self._detector:
            return self._detector.class_names
        return []

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "frame_count": self._frame_count,
            "initialized": self._initialized,
        }
        if self._tracker:
            stats["tracker"] = self._tracker.get_stats()
        if self._deduper:
            stats["deduper"] = self._deduper.get_stats()
        if self._rule_engine:
            stats["rules"] = self._rule_engine.get_rules_info()
        return stats
