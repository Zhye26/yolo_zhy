"""
ByteTrack state manager.
Encapsulates tracker lifecycle and provides clean interface for frame processing.
"""
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from app.core.types import Detection, Track
from app.config import settings


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack tracker."""
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30


class TrackManager:
    """
    Manages ByteTrack tracker state independently.
    Provides clean interface for tracking detections across frames.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig(
            track_thresh=settings.tracking.track_thresh,
            track_buffer=settings.tracking.track_buffer,
            match_thresh=settings.tracking.match_thresh,
            frame_rate=settings.tracking.frame_rate,
        )
        self._tracker = None
        self._frame_id = 0
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize ByteTrack tracker."""
        try:
            from boxmot import BYTETracker
            self._tracker = BYTETracker(
                track_thresh=self.config.track_thresh,
                track_buffer=self.config.track_buffer,
                match_thresh=self.config.match_thresh,
                frame_rate=self.config.frame_rate,
            )
        except ImportError:
            self._tracker = None

    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._frame_id = 0
        self._init_tracker()

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from current frame
            frame: Current frame image (for appearance features)

        Returns:
            List of tracks with assigned track IDs
        """
        if not self._tracker or not detections:
            return [
                Track(track_id=-1, detection=det)
                for det in detections
            ]

        self._frame_id += 1
        det_array = self._detections_to_array(detections)
        tracks_output = self._tracker.update(det_array, frame)

        return self._array_to_tracks(tracks_output, detections)

    def _detections_to_array(self, detections: List[Detection]) -> np.ndarray:
        """Convert detections to numpy array for ByteTrack."""
        if not detections:
            return np.empty((0, 6))

        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det_list.append([x1, y1, x2, y2, det.confidence, det.class_id])

        return np.array(det_list, dtype=np.float32)

    def _array_to_tracks(
        self,
        tracks_output: np.ndarray,
        original_detections: List[Detection]
    ) -> List[Track]:
        """Convert ByteTrack output to Track objects."""
        if tracks_output is None or len(tracks_output) == 0:
            return [
                Track(track_id=-1, detection=det)
                for det in original_detections
            ]

        tracks = []
        for row in tracks_output:
            if len(row) >= 6:
                x1, y1, x2, y2, track_id, conf = row[:6]
                class_id = int(row[6]) if len(row) > 6 else 0
            else:
                continue

            det = Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(conf),
                class_id=class_id,
            )
            tracks.append(Track(track_id=int(track_id), detection=det))

        return tracks

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks."""
        if not self._tracker:
            return 0
        return len(getattr(self._tracker, 'tracked_stracks', []))

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "frame_id": self._frame_id,
            "active_tracks": self.get_active_track_count(),
            "tracker_available": self._tracker is not None,
        }
