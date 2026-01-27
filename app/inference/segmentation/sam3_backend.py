"""
SAM3 segmentation backend for YOLO+SAM cascade.
Uses YOLO detections as box prompts for SAM3 mask generation.
"""
import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class SegmentationResult:
    """Result of segmentation for a single detection."""
    mask: np.ndarray  # H x W boolean mask
    score: float
    class_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]


class SAM3Backend:
    """SAM3 segmentation backend using box prompts from YOLO."""

    def __init__(
        self,
        checkpoint_path: str = "/home/ubuntu/SAM3/sam3.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load SAM3 model."""
        if self._loaded:
            return

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            enable_segmentation=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        self._loaded = True

    def warmup(self, imgsz: Tuple[int, int] = (640, 640)) -> None:
        """Warmup with dummy input."""
        if not self._loaded:
            self.load()
        dummy = np.zeros((imgsz[1], imgsz[0], 3), dtype=np.uint8)
        self.segment_with_boxes(dummy, [[100, 100, 200, 200]], [0], ["dummy"])

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _xyxy_to_cxcywh_normalized(
        self, bbox: List[float], img_w: int, img_h: int
    ) -> List[float]:
        """Convert [x1,y1,x2,y2] to normalized [cx,cy,w,h]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return [cx, cy, w, h]

    @torch.inference_mode()
    def segment_with_boxes(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        class_ids: List[int],
        class_names: List[str],
    ) -> List[SegmentationResult]:
        """
        Generate masks for given bounding boxes.

        Args:
            frame: BGR image (H, W, 3)
            boxes: List of [x1, y1, x2, y2] bounding boxes
            class_ids: Class ID for each box
            class_names: Class name for each box

        Returns:
            List of SegmentationResult with masks
        """
        if not self._loaded:
            self.load()

        if len(boxes) == 0:
            return []

        # Convert BGR to RGB PIL Image
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)
        img_h, img_w = frame.shape[:2]

        results = []
        # Process each box as a geometric prompt
        state = self.processor.set_image(pil_image)

        for i, (bbox, cls_id, cls_name) in enumerate(
            zip(boxes, class_ids, class_names)
        ):
            # Reset prompts for each detection
            self.processor.reset_all_prompts(state)
            state = self.processor.set_image(pil_image, state)

            # Convert bbox to normalized cxcywh format
            norm_box = self._xyxy_to_cxcywh_normalized(bbox, img_w, img_h)

            # Add box as positive prompt
            state = self.processor.add_geometric_prompt(
                box=norm_box, label=True, state=state
            )

            # Get mask if available
            if "masks" in state and len(state["masks"]) > 0:
                mask = state["masks"][0, 0].cpu().numpy()
                score = state["scores"][0].item() if "scores" in state else 1.0
            else:
                # Fallback: create mask from bbox
                mask = np.zeros((img_h, img_w), dtype=bool)
                x1, y1, x2, y2 = map(int, bbox)
                mask[y1:y2, x1:x2] = True
                score = 0.5

            results.append(SegmentationResult(
                mask=mask,
                score=score,
                class_id=cls_id,
                class_name=cls_name,
                bbox=bbox,
            ))

        return results
