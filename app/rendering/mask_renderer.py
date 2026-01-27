"""
Mask visualization renderer with semi-transparent colored overlays.
Each class gets a distinct color for clear visual differentiation.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from app.inference.segmentation.sam3_backend import SegmentationResult


# Class-specific colors (BGR format for OpenCV)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "ebike": (255, 165, 0),      # Orange
    "driver": (0, 255, 0),       # Green
    "passenger": (0, 0, 255),    # Red
    "helmet": (255, 255, 0),     # Cyan
    "default": (128, 128, 128),  # Gray
}

# Fallback colors by class_id
ID_COLORS: List[Tuple[int, int, int]] = [
    (255, 165, 0),   # 0: ebike - Orange
    (0, 255, 0),     # 1: driver - Green
    (0, 0, 255),     # 2: passenger - Red
    (255, 255, 0),   # 3: helmet - Cyan
    (255, 0, 255),   # 4: Magenta
    (0, 255, 255),   # 5: Yellow
    (128, 0, 255),   # 6: Purple
    (255, 128, 0),   # 7: Light Orange
]


class MaskRenderer:
    """Renders segmentation masks with semi-transparent colored overlays."""

    def __init__(self, alpha: float = 0.4, show_contours: bool = True):
        """
        Args:
            alpha: Transparency level (0=transparent, 1=opaque)
            show_contours: Whether to draw mask contours
        """
        self.alpha = alpha
        self.show_contours = show_contours

    def get_color(self, class_name: str, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class."""
        if class_name in CLASS_COLORS:
            return CLASS_COLORS[class_name]
        if 0 <= class_id < len(ID_COLORS):
            return ID_COLORS[class_id]
        return CLASS_COLORS["default"]

    def render_masks(
        self,
        frame: np.ndarray,
        results: List[SegmentationResult],
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Render segmentation masks on frame.

        Args:
            frame: BGR image (H, W, 3)
            results: List of SegmentationResult
            show_labels: Whether to show class labels

        Returns:
            Frame with rendered masks
        """
        if len(results) == 0:
            return frame

        output = frame.copy()
        overlay = frame.copy()

        for result in results:
            color = self.get_color(result.class_name, result.class_id)
            mask = result.mask.astype(np.uint8)

            # Apply colored overlay where mask is True
            overlay[mask > 0] = color

            # Draw contours
            if self.show_contours:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(output, contours, -1, color, 2)

            # Draw label
            if show_labels:
                x1, y1 = int(result.bbox[0]), int(result.bbox[1])
                label = f"{result.class_name} {result.score:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    output, (x1, y1 - th - 4), (x1 + tw, y1), color, -1
                )
                cv2.putText(
                    output, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

        # Blend overlay with output
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    def render_single_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        label: Optional[str] = None,
        bbox: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Render a single mask on frame."""
        output = frame.copy()
        overlay = frame.copy()
        mask_uint8 = mask.astype(np.uint8)

        overlay[mask_uint8 > 0] = color

        if self.show_contours:
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, color, 2)

        if label and bbox:
            x1, y1 = int(bbox[0]), int(bbox[1])
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(output, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                output, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output
