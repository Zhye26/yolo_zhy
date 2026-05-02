"""
Core data types for the detection pipeline.
These dataclasses define the contracts between pipeline stages.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
import numpy as np


class ViolationType(Enum):
    """Types of violations that can be detected."""
    PASSENGER = "passenger"
    NO_HELMET = "no_helmet"


class ViolationState(Enum):
    """States for the violation deduplication FSM."""
    IDLE = auto()
    CANDIDATE = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


@dataclass
class FrameMeta:
    """Metadata for a single frame."""
    frame_index: int
    timestamp: float  # seconds
    width: int
    height: int
    fps: float = 30.0
    source_id: str = "default"


@dataclass
class Detection:
    """A single detection from the model."""
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str = ""

    @property
    def center(self) -> tuple:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class Track:
    """A tracked object with ID."""
    track_id: int
    detection: Detection
    age: int = 0
    hits: int = 1
    state: str = "tracked"


@dataclass
class ViolationCandidate:
    """A potential violation detected by a rule."""
    rule_id: str
    violation_type: ViolationType
    entity_ids: List[int]  # track_ids or detection indices
    bbox: List[float]
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ViolationEvent:
    """A confirmed violation event after deduplication."""
    event_id: str
    violation_type: ViolationType
    rule_id: str
    track_id: Optional[int]
    bbox: List[float]
    confidence: float
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    description: str = ""
    screenshot_path: Optional[str] = None


@dataclass
class FrameContext:
    """Context passed through the pipeline for a single frame."""
    frame: np.ndarray
    meta: FrameMeta
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    candidates: List[ViolationCandidate] = field(default_factory=list)
    violations: List[ViolationEvent] = field(default_factory=list)


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    meta: FrameMeta
    detections: List[Detection]
    tracks: List[Track]
    active_violations: List[ViolationEvent] = field(default_factory=list)
    new_violations: List[ViolationEvent] = field(default_factory=list)
    rendered_frame: Optional[np.ndarray] = None
