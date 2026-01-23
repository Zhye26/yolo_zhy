"""Inference module for model backends."""
from .backends import DetectorBackend, UltralyticsBackend, TensorRTBackend

__all__ = [
    "DetectorBackend",
    "UltralyticsBackend",
    "TensorRTBackend",
]
