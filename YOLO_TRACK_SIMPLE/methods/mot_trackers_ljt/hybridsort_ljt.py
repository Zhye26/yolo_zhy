from __future__ import annotations

from .base_ljt import BoxmotTrackerAdapter, TrackerConfig


class HybridSortMethod(BoxmotTrackerAdapter):
    tracker_name = "hybridsort"
    boxmot_class_name = "HybridSORT"
    requires_reid = True

    def _constructor_kwargs(self):
        return {
            "reid_weights": self.config.reid_weights,
            "device": self.config.device,
            "half": self.config.fp16,
            "det_thresh": self.config.track_thresh,
            "max_age": self.config.max_age,
            "min_hits": self.config.min_hits,
            "iou_threshold": 0.3,
            "delta_t": 3,
            "asso_func": "iou",
            "inertia": 0.2,
        }


def create_tracker(config: TrackerConfig):
    return HybridSortMethod(config)
