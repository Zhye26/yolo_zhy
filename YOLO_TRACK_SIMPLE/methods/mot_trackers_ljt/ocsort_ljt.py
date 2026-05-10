from __future__ import annotations

from .base_ljt import BoxmotTrackerAdapter, TrackerConfig


class OCSortMethod(BoxmotTrackerAdapter):
    tracker_name = "ocsort"
    boxmot_class_name = "OCSORT"

    def _constructor_kwargs(self):
        return {
            "per_class": True,
            "det_thresh": self.config.track_thresh,
            "max_age": self.config.max_age,
            "min_hits": self.config.min_hits,
            "asso_threshold": self.config.match_thresh,
            "delta_t": 3,
            "asso_func": "iou",
            "inertia": 0.2,
            "use_byte": False,
        }


def create_tracker(config: TrackerConfig):
    return OCSortMethod(config)
