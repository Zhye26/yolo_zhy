"""
YOLO + SAM3 cascade detector.
Uses YOLO for detection and SAM3 for instance segmentation.
Supports memory-efficient sequential loading for limited GPU memory.
"""
import gc
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
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
        sequential_mode: bool = False,  # Memory-efficient mode
    ):
        self.model_path = model_path or str(settings.model.model_path)
        self.sam3_checkpoint = sam3_checkpoint
        self.use_tensorrt = use_tensorrt
        self.mask_alpha = mask_alpha
        self.sequential_mode = sequential_mode
        self.class_names = settings.detection.class_names

        self._pipeline: Optional[FramePipeline] = None
        self._sam3: Optional[SAM3Backend] = None
        self._det_renderer: Optional[DetectionRenderer] = None
        self._mask_renderer: Optional[MaskRenderer] = None
        self._last_result: Optional[CascadeResult] = None

    def load_model(self) -> bool:
        """Load YOLO and SAM3 models."""
        try:
            # Load YOLO pipeline
            config = PipelineConfig(
                use_tensorrt=self.use_tensorrt,
                enable_tracking=True,
                enable_rules=True,
                enable_dedup=True,
            )
            self._pipeline = FramePipeline(config)
            self._pipeline.initialize(self.model_path)

            # Load SAM3 (skip in sequential mode - load on demand)
            if not self.sequential_mode:
                self._sam3 = SAM3Backend(
                    checkpoint_path=self.sam3_checkpoint,
                    device="cuda",
                )
                self._sam3.load()

            # Initialize renderers
            self._det_renderer = DetectionRenderer(self._pipeline.class_names)
            self._mask_renderer = MaskRenderer(alpha=self.mask_alpha)

            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _clear_gpu_memory(self):
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_sam3_on_demand(self):
        """Load SAM3 on demand for sequential mode."""
        if self._sam3 is None:
            self._sam3 = SAM3Backend(
                checkpoint_path=self.sam3_checkpoint,
                device="cuda",
            )
        if not self._sam3.is_loaded:
            self._sam3.load()

    def _unload_sam3(self):
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
            self.load_model()

        # Step 1: YOLO detection
        frame_result = self._pipeline.process(image)

        # Step 2: SAM3 segmentation using YOLO boxes as prompts
        boxes = []
        class_ids = []
        class_names = []

        # Use tracks if available, otherwise detections
        sources = frame_result.tracks or frame_result.detections
        for item in sources:
            det = item.detection if hasattr(item, 'detection') else item
            boxes.append(det.bbox)
            class_ids.append(det.class_id)
            class_names.append(det.class_name or self.class_names[det.class_id])

        segmentations = []
        if boxes:
            # In sequential mode, load SAM3 on demand
            if self.sequential_mode:
                self._clear_gpu_memory()
                self._load_sam3_on_demand()

            if self._sam3:
                segmentations = self._sam3.segment_with_boxes(
                    image, boxes, class_ids, class_names
                )

            # In sequential mode, unload SAM3 after use
            if self.sequential_mode:
                self._unload_sam3()

        result = CascadeResult(
            frame_result=frame_result,
            segmentations=segmentations,
        )
        self._last_result = result
        return result

    def draw_results(
        self,
        image: np.ndarray,
        result: CascadeResult,
        show_masks: bool = True,
        show_boxes: bool = True,
        show_violations: bool = True,
    ) -> np.ndarray:
        """
        Draw detection and segmentation results.

        Args:
            image: BGR image
            result: CascadeResult from detect()
            show_masks: Whether to show segmentation masks
            show_boxes: Whether to show bounding boxes
            show_violations: Whether to show violation indicators

        Returns:
            Rendered image
        """
        output = image.copy()

        # Draw masks first (as background layer)
        if show_masks and result.segmentations and self._mask_renderer:
            output = self._mask_renderer.render_masks(
                output, result.segmentations, show_labels=False
            )

        # Draw boxes and violations on top
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
            self._pipeline.reset()

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {}
        if self._pipeline:
            stats.update(self._pipeline.get_stats())
        stats["sam3_loaded"] = self._sam3 is not None and self._sam3.is_loaded
        return stats
