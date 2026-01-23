"""
Ultralytics YOLO backend implementation.
"""
from typing import List, Optional
import numpy as np
from app.inference.backends.base import DetectorBackend
from app.core.types import Detection


class UltralyticsBackend(DetectorBackend):
    """Ultralytics YOLO backend for inference."""

    def __init__(self):
        self._model = None
        self._class_names: List[str] = []

    def load(self, model_path: str) -> None:
        """Load Ultralytics YOLO model."""
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._class_names = list(self._model.names.values())

    def predict(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45
    ) -> List[Detection]:
        """Run inference using Ultralytics."""
        if not self._model:
            return []

        results = self._model.predict(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confs, classes):
                detections.append(Detection(
                    bbox=bbox.tolist(),
                    confidence=float(conf),
                    class_id=int(cls_id),
                ))

        return detections

    def warmup(self, imgsz: tuple = (640, 640)) -> None:
        """Warmup model with dummy input."""
        if self._model:
            dummy = np.zeros((imgsz[1], imgsz[0], 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        return self._class_names
