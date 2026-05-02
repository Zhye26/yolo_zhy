"""Video processing session manager for real-time detection."""
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Generator
import cv2
import numpy as np


class SessionState(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SessionStats:
    total_frames: int = 0
    processed_frames: int = 0
    fps: float = 0.0
    passenger_violations: int = 0
    helmet_violations: int = 0
    start_time: float = 0.0
    last_update: float = 0.0


@dataclass
class VideoSession:
    session_id: str
    detection_id: int
    video_path: str
    state: SessionState = SessionState.CREATED
    stats: SessionStats = field(default_factory=SessionStats)
    error_message: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict:
        with self._lock:
            progress = 0.0
            if self.stats.total_frames > 0:
                progress = self.stats.processed_frames / self.stats.total_frames * 100
            return {
                "session_id": self.session_id,
                "detection_id": self.detection_id,
                "state": self.state.value,
                "progress": round(progress, 1),
                "stats": {
                    "total_frames": self.stats.total_frames,
                    "processed_frames": self.stats.processed_frames,
                    "fps": round(self.stats.fps, 1),
                    "passenger_violations": self.stats.passenger_violations,
                    "helmet_violations": self.stats.helmet_violations,
                },
                "error": self.error_message if self.state == SessionState.ERROR else None,
            }


class VideoSessionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: Dict[str, VideoSession] = {}
                    cls._instance._by_detection: Dict[int, str] = {}
        return cls._instance

    def create_session(self, detection_id: int, video_path: str) -> VideoSession:
        session_id = str(uuid.uuid4())[:8]
        session = VideoSession(
            session_id=session_id,
            detection_id=detection_id,
            video_path=video_path,
        )
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            session.stats.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        self._sessions[session_id] = session
        self._by_detection[detection_id] = session_id
        return session

    def get_session(self, session_id: str) -> Optional[VideoSession]:
        return self._sessions.get(session_id)

    def get_by_detection(self, detection_id: int) -> Optional[VideoSession]:
        session_id = self._by_detection.get(detection_id)
        return self._sessions.get(session_id) if session_id else None

    def update_stats(
        self,
        session_id: str,
        processed_frames: int = None,
        fps: float = None,
        passenger_violations: int = None,
        helmet_violations: int = None,
    ) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        with session._lock:
            if processed_frames is not None:
                session.stats.processed_frames = processed_frames
            if fps is not None:
                session.stats.fps = fps
            if passenger_violations is not None:
                session.stats.passenger_violations = passenger_violations
            if helmet_violations is not None:
                session.stats.helmet_violations = helmet_violations
            session.stats.last_update = time.time()

    def set_state(self, session_id: str, state: SessionState, error: str = "") -> None:
        session = self._sessions.get(session_id)
        if session:
            with session._lock:
                session.state = state
                if error:
                    session.error_message = error

    def cleanup_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            self._by_detection.pop(session.detection_id, None)


session_manager = VideoSessionManager()
