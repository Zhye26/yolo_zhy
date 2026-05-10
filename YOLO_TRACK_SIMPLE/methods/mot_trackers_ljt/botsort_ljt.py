from __future__ import annotations

from .base_ljt import BoxmotTrackerAdapter, TrackerConfig


class BoTSortMethod(BoxmotTrackerAdapter):
    tracker_name = "botsort"
    boxmot_class_name = "BoTSORT"
    requires_reid = True

    def _constructor_kwargs(self):
        return {
            "model_weights": self.config.reid_weights,
            "device": self.config.device,
            "fp16": self.config.fp16,
            "track_high_thresh": self.config.track_thresh,
            "track_low_thresh": 0.1,
            "new_track_thresh": max(self.config.track_thresh, 0.6),
            "track_buffer": self.config.track_buffer,
            "match_thresh": self.config.match_thresh,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "cmc_method": "sparseOptFlow",
            "frame_rate": self.config.frame_rate,
            "with_reid": True,
        }


def create_tracker(config: TrackerConfig):
    return BoTSortMethod(config)
