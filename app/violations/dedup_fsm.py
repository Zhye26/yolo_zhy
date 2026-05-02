"""
Violation deduplication finite state machine.
Manages violation lifecycle: IDLE -> CANDIDATE -> ACTIVE -> COOLDOWN -> IDLE
"""
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from app.core.types import (
    ViolationState,
    ViolationCandidate,
    ViolationEvent,
    ViolationType,
    FrameMeta,
)
from app.config import settings


@dataclass
class ViolationStateEntry:
    """State entry for a single tracked violation."""
    key: str
    state: ViolationState = ViolationState.IDLE
    violation_type: ViolationType = ViolationType.PASSENGER
    rule_id: str = ""
    track_id: Optional[int] = None
    bbox: List[float] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""
    first_frame: int = 0
    last_frame: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    hit_count: int = 0
    cooldown_start_frame: int = 0
    event_emitted: bool = False
    confirm_hits_required: int = 0


class ViolationDeduper:
    """
    Finite state machine for violation deduplication.

    State transitions:
    - IDLE -> CANDIDATE: First detection of a potential violation
    - CANDIDATE -> ACTIVE: Confirmed after min_frames_to_confirm
    - CANDIDATE -> IDLE: Timeout (gap > max_gap_frames)
    - ACTIVE -> COOLDOWN: Violation no longer detected
    - COOLDOWN -> IDLE: Cooldown period expired
    - COOLDOWN -> ACTIVE: Re-detected during cooldown
    """

    def __init__(self):
        self.state_store: Dict[str, ViolationStateEntry] = {}
        self.min_frames = settings.violations.min_frames_to_confirm
        self.cooldown_frames = settings.violations.cooldown_frames
        self.max_gap_frames = settings.violations.max_gap_frames
        self.render_grace_frames = 3
        self.render_grace_min_hits = 100

    def reset(self) -> None:
        """Reset all state."""
        self.state_store.clear()

    def update(
        self,
        candidates: List[ViolationCandidate],
        meta: FrameMeta
    ) -> Tuple[List[ViolationEvent], List[ViolationEvent]]:
        """
        Update state machine with new candidates.

        Args:
            candidates: List of violation candidates from rule engine
            meta: Frame metadata

        Returns:
            Tuple of (all_active_violations, newly_emitted_violations)
        """
        frame_idx = meta.frame_index
        timestamp = meta.timestamp

        detected_keys = set()
        new_events = []

        for candidate in candidates:
            key = self._make_key(candidate)
            detected_keys.add(key)

            if key not in self.state_store:
                self.state_store[key] = ViolationStateEntry(
                    key=key,
                    state=ViolationState.CANDIDATE,
                    violation_type=candidate.violation_type,
                    rule_id=candidate.rule_id,
                    track_id=candidate.entity_ids[0] if candidate.entity_ids else None,
                    bbox=candidate.bbox,
                    confidence=candidate.confidence,
                    description=candidate.description,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    first_timestamp=timestamp,
                    last_timestamp=timestamp,
                    hit_count=1,
                    confirm_hits_required=self._confirm_hits_required(candidate),
                )
            else:
                entry = self.state_store[key]
                entry.last_frame = frame_idx
                entry.last_timestamp = timestamp
                entry.hit_count += 1
                entry.bbox = candidate.bbox
                entry.confidence = max(entry.confidence, candidate.confidence)

                if entry.state == ViolationState.CANDIDATE:
                    if entry.hit_count >= max(1, entry.confirm_hits_required):
                        entry.state = ViolationState.ACTIVE
                        if not entry.event_emitted:
                            entry.event_emitted = True
                            new_events.append(self._create_event(entry))

                elif entry.state == ViolationState.COOLDOWN:
                    entry.state = ViolationState.ACTIVE

                elif entry.state == ViolationState.ACTIVE:
                    pass

        keys_to_remove = []
        for key, entry in self.state_store.items():
            if key not in detected_keys:
                gap = frame_idx - entry.last_frame

                if entry.state == ViolationState.CANDIDATE:
                    if gap > self.max_gap_frames:
                        keys_to_remove.append(key)

                elif entry.state == ViolationState.ACTIVE:
                    entry.state = ViolationState.COOLDOWN
                    entry.cooldown_start_frame = frame_idx

                elif entry.state == ViolationState.COOLDOWN:
                    cooldown_elapsed = frame_idx - entry.cooldown_start_frame
                    if cooldown_elapsed > self.cooldown_frames:
                        keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.state_store[key]

        active_violations = [
            self._create_event(entry)
            for entry in self.state_store.values()
            if self._should_render(entry, frame_idx)
        ]

        return active_violations, new_events

    def _should_render(self, entry: ViolationStateEntry, frame_idx: int) -> bool:
        if entry.state == ViolationState.ACTIVE:
            return True
        if entry.state != ViolationState.COOLDOWN:
            return False
        if entry.confirm_hits_required < self.min_frames and entry.hit_count >= max(1, entry.confirm_hits_required):
            cooldown_elapsed = frame_idx - entry.cooldown_start_frame
            return cooldown_elapsed <= 2
        if entry.hit_count < self.render_grace_min_hits:
            return False
        cooldown_elapsed = frame_idx - entry.cooldown_start_frame
        return cooldown_elapsed <= self.render_grace_frames

    def _make_key(self, candidate: ViolationCandidate) -> str:
        """Create a unique key for a violation candidate."""
        track_id = candidate.entity_ids[0] if candidate.entity_ids else "no_track"
        return f"{candidate.rule_id}_{track_id}"

    def _confirm_hits_required(self, candidate: ViolationCandidate) -> int:
        evidence_class = str(candidate.evidence.get("class", ""))
        if evidence_class == "overload_merged_side_view":
            return max(2, self.min_frames - 1)
        return self.min_frames

    def _create_event(self, entry: ViolationStateEntry) -> ViolationEvent:
        """Create a violation event from a state entry."""
        return ViolationEvent(
            event_id=str(uuid.uuid4())[:8],
            violation_type=entry.violation_type,
            rule_id=entry.rule_id,
            track_id=entry.track_id,
            bbox=entry.bbox,
            confidence=entry.confidence,
            start_frame=entry.first_frame,
            end_frame=entry.last_frame,
            start_timestamp=entry.first_timestamp,
            end_timestamp=entry.last_timestamp,
            description=entry.description,
        )

    def get_stats(self) -> Dict:
        """Get current state machine statistics."""
        states = {}
        for entry in self.state_store.values():
            state_name = entry.state.name
            states[state_name] = states.get(state_name, 0) + 1
        return {
            "total_entries": len(self.state_store),
            "by_state": states,
        }
