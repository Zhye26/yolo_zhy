"""
ByteTrack state manager.
Encapsulates tracker lifecycle and provides clean interface for frame processing.
"""
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from dataclasses import dataclass
from app.core.types import Detection, Track
from app.config import settings


@dataclass
class TrackerConfig:
    """Configuration for tracker backends."""
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30
    fallback_iou_thresh: float = 0.18
    fallback_max_missed: int = 6
    fallback_min_confirmed_hits: int = 2
    fallback_smooth_alpha: float = 0.65
    fallback_conf_decay: float = 0.92


@dataclass
class FallbackTrackState:
    """Internal lightweight track state for IoU fallback tracking."""
    track_id: int
    detection: Detection
    hits: int = 1
    missed: int = 0
    age: int = 1


class TrackManager:
    """
    Manages tracker state independently.
    Uses ByteTrack when available and falls back to a lightweight IoU tracker.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig(
            track_thresh=settings.tracking.track_thresh,
            track_buffer=settings.tracking.track_buffer,
            match_thresh=settings.tracking.match_thresh,
            frame_rate=settings.tracking.frame_rate,
            fallback_iou_thresh=settings.tracking.fallback_iou_thresh,
            fallback_max_missed=settings.tracking.fallback_max_missed,
            fallback_min_confirmed_hits=settings.tracking.fallback_min_confirmed_hits,
            fallback_smooth_alpha=settings.tracking.fallback_smooth_alpha,
            fallback_conf_decay=settings.tracking.fallback_conf_decay,
        )
        self._tracker = None
        self._tracker_mode = "fallback_iou"
        self._frame_id = 0
        self._next_track_id = 1
        self._fallback_tracks: Dict[int, FallbackTrackState] = {}
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize ByteTrack if available, otherwise keep fallback tracker."""
        try:
            from boxmot import BYTETracker
            self._tracker = BYTETracker(
                track_thresh=self.config.track_thresh,
                track_buffer=self.config.track_buffer,
                match_thresh=self.config.match_thresh,
                frame_rate=self.config.frame_rate,
            )
            self._tracker_mode = "bytetrack"
        except ImportError:
            self._tracker = None
            self._tracker_mode = "fallback_iou"

    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._frame_id = 0
        self._next_track_id = 1
        self._fallback_tracks = {}
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
        self._frame_id += 1

        if self._tracker is not None:
            if not detections:
                return []
            det_array = self._detections_to_array(detections)
            tracks_output = self._tracker.update(det_array, frame)
            tracks = self._array_to_tracks(tracks_output, detections)
            return self._append_unmatched_detections(tracks, detections)

        return self._update_fallback(detections)

    def _update_fallback(self, detections: List[Detection]) -> List[Track]:
        active_track_ids = [
            track_id
            for track_id, state in self._fallback_tracks.items()
            if state.missed <= self.config.fallback_max_missed
        ]

        matches, unmatched_track_ids, unmatched_detection_indices = self._match_fallback_tracks(
            active_track_ids,
            detections,
        )

        for track_id, detection_index in matches:
            self._update_fallback_track(track_id, detections[detection_index])

        for track_id in unmatched_track_ids:
            state = self._fallback_tracks.get(track_id)
            if state is None:
                continue
            state.missed += 1
            state.age += 1

        for detection_index in unmatched_detection_indices:
            self._create_fallback_track(detections[detection_index])

        self._prune_fallback_tracks()
        return self._build_fallback_outputs()

    def _match_fallback_tracks(
        self,
        track_ids: List[int],
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or not detections:
            return [], list(track_ids), list(range(len(detections)))

        candidates: List[Tuple[float, int, int]] = []
        for track_id in track_ids:
            state = self._fallback_tracks.get(track_id)
            if state is None:
                continue
            for detection_index, detection in enumerate(detections):
                if detection.class_id != state.detection.class_id:
                    continue
                score = self._association_score(state.detection.bbox, detection.bbox)
                if score < self._score_threshold_for_class(detection.class_id):
                    continue
                candidates.append((score, track_id, detection_index))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_track_ids = set()
        matched_detection_indices = set()
        matches: List[Tuple[int, int]] = []

        for _, track_id, detection_index in candidates:
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)
            matches.append((track_id, detection_index))

        unmatched_track_ids = [track_id for track_id in track_ids if track_id not in matched_track_ids]
        unmatched_detection_indices = [
            detection_index
            for detection_index in range(len(detections))
            if detection_index not in matched_detection_indices
        ]
        return matches, unmatched_track_ids, unmatched_detection_indices

    def _update_fallback_track(self, track_id: int, detection: Detection) -> None:
        state = self._fallback_tracks[track_id]
        smoothed_bbox = self._smooth_bbox(state.detection.bbox, detection.bbox)
        confidence = max(detection.confidence, state.detection.confidence * 0.85)
        state.detection = Detection(
            bbox=smoothed_bbox,
            confidence=confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        state.hits += 1
        state.missed = 0
        state.age += 1

    def _create_fallback_track(self, detection: Detection) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._fallback_tracks[track_id] = FallbackTrackState(
            track_id=track_id,
            detection=Detection(
                bbox=list(detection.bbox),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
            ),
        )

    def _prune_fallback_tracks(self) -> None:
        expired_track_ids = [
            track_id
            for track_id, state in self._fallback_tracks.items()
            if state.missed > self.config.fallback_max_missed
        ]
        for track_id in expired_track_ids:
            self._fallback_tracks.pop(track_id, None)

    def _build_fallback_outputs(self) -> List[Track]:
        tracks: List[Track] = []
        for track_id in sorted(self._fallback_tracks):
            state = self._fallback_tracks[track_id]
            if state.missed > 0:
                continue
            if state.hits < self.config.fallback_min_confirmed_hits:
                continue

            detection = Detection(
                bbox=list(state.detection.bbox),
                confidence=max(0.05, state.detection.confidence),
                class_id=state.detection.class_id,
                class_name=state.detection.class_name,
            )
            tracks.append(Track(
                track_id=track_id,
                detection=detection,
                age=state.age,
                hits=state.hits,
                state="tracked",
            ))
        return tracks

    def _smooth_bbox(self, previous_bbox: List[float], current_bbox: List[float]) -> List[float]:
        alpha = self.config.fallback_smooth_alpha
        return [
            previous * alpha + current * (1.0 - alpha)
            for previous, current in zip(previous_bbox, current_bbox)
        ]

    def _association_score(self, previous_bbox: List[float], current_bbox: List[float]) -> float:
        iou = self._iou(previous_bbox, current_bbox)
        center_similarity = self._center_similarity(previous_bbox, current_bbox)
        scale_similarity = self._scale_similarity(previous_bbox, current_bbox)
        return iou * 0.7 + center_similarity * 0.2 + scale_similarity * 0.1

    def _score_threshold_for_class(self, class_id: int) -> float:
        if class_id in {settings.detection.driver_class_id, settings.detection.passenger_class_id}:
            return max(0.08, self.config.fallback_iou_thresh * 0.7)
        return self.config.fallback_iou_thresh

    def _center_similarity(self, box_a: List[float], box_b: List[float]) -> float:
        center_ax = (box_a[0] + box_a[2]) / 2
        center_ay = (box_a[1] + box_a[3]) / 2
        center_bx = (box_b[0] + box_b[2]) / 2
        center_by = (box_b[1] + box_b[3]) / 2
        distance = math.hypot(center_ax - center_bx, center_ay - center_by)

        width_a = max(1.0, box_a[2] - box_a[0])
        height_a = max(1.0, box_a[3] - box_a[1])
        width_b = max(1.0, box_b[2] - box_b[0])
        height_b = max(1.0, box_b[3] - box_b[1])
        scale = max(math.hypot(width_a, height_a), math.hypot(width_b, height_b), 1.0)
        return max(0.0, 1.0 - distance / (scale * 1.8))

    def _scale_similarity(self, box_a: List[float], box_b: List[float]) -> float:
        area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return min(area_a, area_b) / max(area_a, area_b)

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
            return []

        tracks = []
        for row in tracks_output:
            if len(row) >= 6:
                x1, y1, x2, y2, track_id, conf = row[:6]
                class_id = int(row[6]) if len(row) > 6 else 0
            else:
                continue

            matched_detection = self._find_best_detection_match(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                class_id=class_id,
                detections=original_detections,
            )
            class_name = matched_detection.class_name if matched_detection else ""

            det = Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(conf),
                class_id=class_id,
                class_name=class_name,
            )
            tracks.append(Track(track_id=int(track_id), detection=det))

        return tracks

    def _append_unmatched_detections(
        self,
        tracks: List[Track],
        detections: List[Detection],
    ) -> List[Track]:
        """Keep detections that the tracker did not return."""
        combined = list(tracks)
        for det in detections:
            if any(self._track_matches_detection(track, det) for track in tracks):
                continue
            combined.append(Track(track_id=-1, detection=det))
        return combined

    def _track_matches_detection(self, track: Track, detection: Detection) -> bool:
        if track.detection.class_id != detection.class_id:
            return False
        return self._iou(track.detection.bbox, detection.bbox) > 0.4

    def _find_best_detection_match(
        self,
        bbox: List[float],
        class_id: int,
        detections: List[Detection],
    ) -> Optional[Detection]:
        best_match: Optional[Detection] = None
        best_iou = 0.0
        for det in detections:
            if det.class_id != class_id:
                continue
            iou = self._iou(bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = det
        return best_match

    def _iou(self, box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks."""
        if self._tracker is not None:
            return len(getattr(self._tracker, 'tracked_stracks', []))
        return sum(1 for state in self._fallback_tracks.values() if state.missed == 0)

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "frame_id": self._frame_id,
            "active_tracks": self.get_active_track_count(),
            "tracker_available": self._tracker is not None,
            "tracker_mode": self._tracker_mode,
            "fallback_tracks": len(self._fallback_tracks),
        }
