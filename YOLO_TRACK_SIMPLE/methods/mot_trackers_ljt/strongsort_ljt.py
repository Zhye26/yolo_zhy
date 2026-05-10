from __future__ import annotations

from .base_ljt import BoxmotTrackerAdapter, TrackerConfig


class StrongSortMethod(BoxmotTrackerAdapter):
    tracker_name = "strongsort"
    boxmot_class_name = "StrongSORT"
    requires_reid = True

    def _constructor_kwargs(self):
        return {
            "model_weights": self.config.reid_weights,
            "device": self.config.device,
            "fp16": self.config.fp16,
            "max_dist": 0.2,
            "max_iou_dist": 0.7,
            "max_age": self.config.max_age,
            "n_init": 1,
            "nn_budget": 100,
        }


def create_tracker(config: TrackerConfig):
    return StrongSortMethod(config)
