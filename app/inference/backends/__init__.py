"""Inference backends module."""
from .base import DetectorBackend
from .ultralytics_backend import UltralyticsBackend
from .tensorrt_backend import TensorRTBackend

__all__ = [
    "DetectorBackend",
    "UltralyticsBackend",
    "TensorRTBackend",
]
