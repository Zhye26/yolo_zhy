"""
YOLO + SAM3 cascade detector.
Uses YOLO for detection and SAM3 for instance segmentation.
Supports memory-efficient sequential loading for limited GPU memory.
"""
import gc
from contextlib import contextmanager
import cv2
import numpy as np
import torch
from typing import Iterator, List, Dict, Optional
from dataclasses import dataclass

from app.core import FramePipeline, PipelineConfig, FrameResult
from app.inference.segmentation import SAM3Backend
from app.inference.segmentation.sam3_backend import SegmentationResult
from app.rendering import DetectionRenderer, MaskRenderer
from app.config import settings


@dataclass
class CascadeResult:
    """Result from YOLO+SAM3 cascade detection."""
    frame_result: FrameResult
    segmentations: List[SegmentationResult]


class YoloSam3Detector:
    """YOLO + SAM3 cascade detector with segmentation visualization."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sam3_checkpoint: str = "/home/ubuntu/SAM3/sam3.pt",
        use_tensorrt: bool = False,
        mask_alpha: float = 0.4,
        sequential_mode: bool = False,
        task_mode: str = "all",
    ):
        self.model_path = model_path or str(settings.model.model_path)
        self.sam3_checkpoint = sam3_checkpoint
        self.use_tensorrt = use_tensorrt
        self.mask_alpha = mask_alpha
        self.sequential_mode = sequential_mode
        self.task_mode = task_mode
        self.class_names = settings.detection.class_names

        self._pipeline: Optional[FramePipeline] = None
        self._sam3: Optional[SAM3Backend] = None
        self._det_renderer: Optional[DetectionRenderer] = None
        self._mask_renderer: Optional[MaskRenderer] = None
        self._last_result: Optional[CascadeResult] = None

    def load_model(self) -> bool:
        """Load YOLO and SAM3 models."""
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

                if not self.sequential_mode:
                    self._sam3 = SAM3Backend(
                        checkpoint_path=self.sam3_checkpoint,
                        device="cuda",
                    )
                    self._sam3.load()

            self._det_renderer = DetectionRenderer(self._pipeline.class_names)
            self._mask_renderer = MaskRenderer(alpha=self.mask_alpha)
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_sam3_on_demand(self) -> None:
        """Load SAM3 on demand for sequential mode."""
        if self._sam3 is None:
            self._sam3 = SAM3Backend(
                checkpoint_path=self.sam3_checkpoint,
                device="cuda",
            )
        if not self._sam3.is_loaded:
            self._sam3.load()

    def _unload_sam3(self) -> None:
        """Unload SAM3 to free GPU memory."""
        if self._sam3 is not None:
            del self._sam3.model
            del self._sam3.processor
            self._sam3.model = None
            self._sam3.processor = None
            self._sam3._loaded = False
            self._clear_gpu_memory()

    def detect(self, image: np.ndarray) -> CascadeResult:
        """
        Run YOLO detection + SAM3 segmentation cascade.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            CascadeResult with detections and segmentations
        """
        if self._pipeline is None:
            if not self.load_model():
                raise RuntimeError("Failed to initialize YOLO+SAM3 pipeline.")

        with self._task_settings():
            frame_result = self._pipeline.process(image)

        boxes = []
        class_ids = []
        class_names = []
        sources = frame_result.tracks or frame_result.detections
        segmentation_targets = self._select_segmentation_targets(sources)
        for det in segmentation_targets:
            boxes.append(det.bbox)
            class_ids.append(det.class_id)
            class_names.append(det.class_name or self.class_names[det.class_id])

        segmentations: List[SegmentationResult] = []
        if boxes:
            if self.sequential_mode:
                self._clear_gpu_memory()
                self._load_sam3_on_demand()

            if self._sam3:
                segmentations = self._sam3.segment_with_boxes(
                    image, boxes, class_ids, class_names
                )

            if self.sequential_mode:
                self._unload_sam3()

        result = CascadeResult(
            frame_result=frame_result,
            segmentations=segmentations,
        )
        self._last_result = result
        return result

    def _select_segmentation_targets(self, sources) -> List:
        """Pick only business-relevant, stable targets for SAM3."""
        allowed_class_ids = {
            settings.detection.ebike_class_id,
            settings.detection.driver_class_id,
            settings.detection.passenger_class_id,
        }
        selected = []
        for item in sources:
            det = item.detection if hasattr(item, 'detection') else item
            if det.class_id not in allowed_class_ids:
                continue
            if det.confidence < 0.10:
                continue
            selected.append(det)

        selected.sort(key=lambda det: (det.class_id, det.confidence, det.area), reverse=True)
        return selected[:8]

    def detect_violations(
        self,
        detections: CascadeResult,
        use_tracking: bool = False,
    ) -> tuple[List[Dict], List[Dict]]:
        """Return legacy violation dicts for VideoProcessor compatibility."""
        result = detections.frame_result
        all_violations = [self._violation_to_dict(v) for v in result.active_violations]
        new_violations = [self._violation_to_dict(v) for v in result.new_violations]
        return all_violations, new_violations

    def _violation_to_dict(self, violation) -> Dict:
        return {
            'type': violation.violation_type.name.lower(),
            'bbox': violation.bbox,
            'confidence': violation.confidence,
            'track_id': violation.track_id,
            'description': violation.description,
        }

    def draw_results(
        self,
        image: np.ndarray,
        result: CascadeResult,
        violations: Optional[List[Dict]] = None,
        show_masks: bool = True,
        show_boxes: bool = True,
        show_violations: bool = True,
    ) -> np.ndarray:
        """Draw detection and segmentation results."""
        output = image.copy()

        if show_masks and result.segmentations and self._mask_renderer:
            output = self._mask_renderer.render_masks(
                output, result.segmentations, show_labels=False
            )

        if show_boxes and self._det_renderer:
            output = self._det_renderer.render(
                output,
                result.frame_result,
                show_tracks=True,
                show_violations=show_violations,
            )

        return output

    def reset_tracker(self) -> None:
        """Reset tracker state for new video."""
        if self._pipeline:
            with self._task_settings():
                self._pipeline.reset()
        if self._det_renderer:
            self._det_renderer.reset()

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {}
        if self._pipeline:
            stats.update(self._pipeline.get_stats())
        stats['sam3_loaded'] = self._sam3 is not None and self._sam3.is_loaded
        stats['sam3_checkpoint'] = self.sam3_checkpoint
        stats['sam3_sequential_mode'] = self.sequential_mode
        return stats

    @contextmanager
    def _task_settings(self) -> Iterator[None]:
        original_passenger_rule = settings.rules.passenger_rule_enabled
        original_helmet_rule = settings.rules.helmet_rule_enabled
        original_helmet_detection = settings.detection.helmet_detection_enabled
        try:
            if self.task_mode == "helmet":
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
