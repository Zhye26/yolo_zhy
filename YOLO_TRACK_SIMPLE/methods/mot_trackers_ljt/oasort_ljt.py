from __future__ import annotations

from .base_ljt import TrackerConfig


def create_tracker(config: TrackerConfig):
    raise RuntimeError(
        "OA-SORT is listed for comparison but is not available in the current boxmot installation. "
        "Add an OA-SORT implementation here, then register it in factory_ljt.py."
    )
