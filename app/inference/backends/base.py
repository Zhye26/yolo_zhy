"""
Abstract base class for detection backends.
Defines interface for YOLO model inference.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from app.core.types import Detection


class DetectorBackend(ABC):
    """Abstract base class for detection backends."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model from path."""
        pass

    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45
    ) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image as numpy array
            conf_thresh: Confidence threshold
            iou_thresh: IoU threshold for NMS

        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def warmup(self, imgsz: tuple = (640, 640)) -> None:
        """Warmup the model with dummy input."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Get class names."""
        pass
