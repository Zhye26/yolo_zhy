"""
TensorRT backend implementation for optimized inference.
"""
from typing import List, Optional
import numpy as np
from app.inference.backends.base import DetectorBackend
from app.core.types import Detection
from app.config import settings


class TensorRTBackend(DetectorBackend):
    """TensorRT backend for optimized inference."""

    def __init__(self):
        self._engine = None
        self._context = None
        self._class_names: List[str] = []
        self._input_shape: tuple = (640, 640)
        self._bindings = None

    def load(self, model_path: str) -> None:
        """
        Load TensorRT engine from file.
        Supports both .engine and .pt files (auto-export).
        """
        if model_path.endswith('.pt'):
            self._export_and_load(model_path)
        else:
            self._load_engine(model_path)

    def _export_and_load(self, pt_path: str) -> None:
        """Export PyTorch model to TensorRT and load."""
        from ultralytics import YOLO
        model = YOLO(pt_path)
        self._class_names = list(model.names.values())

        engine_path = pt_path.replace('.pt', '.engine')
        model.export(
            format='engine',
            imgsz=settings.model.imgsz,
            half=settings.model.half,
            device=0,
        )
        self._load_engine(engine_path)

    def _load_engine(self, engine_path: str) -> None:
        """Load pre-built TensorRT engine."""
        try:
            import tensorrt as trt
            from cuda import cudart

            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

            self._context = self._engine.create_execution_context()
            self._setup_bindings()

            if not self._class_names:
                self._class_names = settings.detection.class_names
        except ImportError:
            raise RuntimeError("TensorRT not available. Install tensorrt package.")

    def _setup_bindings(self) -> None:
        """Setup input/output bindings for TensorRT engine."""
        import tensorrt as trt
        from cuda import cudart
        import numpy as np

        self._bindings = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            shape = self._engine.get_tensor_shape(name)

            size = np.prod(shape) * np.dtype(dtype).itemsize
            _, device_mem = cudart.cudaMalloc(size)

            self._bindings[name] = {
                'dtype': dtype,
                'shape': shape,
                'device': device_mem,
                'size': size,
            }

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_shape = tuple(shape[2:4])

    def predict(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45
    ) -> List[Detection]:
        """Run inference using TensorRT engine."""
        if not self._engine:
            return []

        input_tensor = self._preprocess(frame)
        output = self._infer(input_tensor)
        return self._postprocess(output, frame.shape, conf_thresh, iou_thresh)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for TensorRT inference."""
        import cv2
        h, w = self._input_shape
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        return np.ascontiguousarray(batched)

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        from cuda import cudart
        import tensorrt as trt

        input_name = self._engine.get_tensor_name(0)
        output_name = self._engine.get_tensor_name(1)

        cudart.cudaMemcpy(
            self._bindings[input_name]['device'],
            input_tensor.ctypes.data,
            input_tensor.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )

        self._context.set_tensor_address(input_name, self._bindings[input_name]['device'])
        self._context.set_tensor_address(output_name, self._bindings[output_name]['device'])
        self._context.execute_async_v3(0)

        output_shape = self._bindings[output_name]['shape']
        output = np.empty(output_shape, dtype=self._bindings[output_name]['dtype'])
        cudart.cudaMemcpy(
            output.ctypes.data,
            self._bindings[output_name]['device'],
            output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )

        return output

    def _postprocess(
        self,
        output: np.ndarray,
        orig_shape: tuple,
        conf_thresh: float,
        iou_thresh: float
    ) -> List[Detection]:
        """Postprocess TensorRT output to detections."""
        predictions = output[0].T
        mask = predictions[:, 4:].max(axis=1) > conf_thresh
        predictions = predictions[mask]

        if len(predictions) == 0:
            return []

        boxes = predictions[:, :4]
        scores = predictions[:, 4:].max(axis=1)
        class_ids = predictions[:, 4:].argmax(axis=1)

        boxes = self._xywh_to_xyxy(boxes)
        boxes = self._scale_boxes(boxes, self._input_shape, orig_shape[:2])

        indices = self._nms(boxes, scores, iou_thresh)

        detections = []
        for i in indices:
            detections.append(Detection(
                bbox=boxes[i].tolist(),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
            ))

        return detections

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xywh to xyxy format."""
        result = np.zeros_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return result

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        from_shape: tuple,
        to_shape: tuple
    ) -> np.ndarray:
        """Scale boxes from input size to original image size."""
        scale_y = to_shape[0] / from_shape[0]
        scale_x = to_shape[1] / from_shape[1]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return boxes

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float
    ) -> List[int]:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def warmup(self, imgsz: tuple = (640, 640)) -> None:
        """Warmup TensorRT engine."""
        if self._engine:
            dummy = np.zeros((imgsz[1], imgsz[0], 3), dtype=np.uint8)
            self.predict(dummy)

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def __del__(self):
        """Cleanup CUDA memory."""
        if self._bindings:
            from cuda import cudart
            for binding in self._bindings.values():
                cudart.cudaFree(binding['device'])
