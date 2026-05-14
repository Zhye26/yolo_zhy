from __future__ import annotations

from .base_ljt import BoxmotTrackerAdapter, TrackerConfig


class ByteTrackMethod(BoxmotTrackerAdapter):
    tracker_name = "byte"
    boxmot_class_name = "BYTETracker"

    def _constructor_kwargs(self):
        return {
            "track_thresh": self.config.track_thresh,
            "match_thresh": self.config.match_thresh,
            "track_buffer": self.config.track_buffer,
            "frame_rate": self.config.frame_rate,
        }


def create_tracker(config: TrackerConfig):
    return ByteTrackMethod(config)
