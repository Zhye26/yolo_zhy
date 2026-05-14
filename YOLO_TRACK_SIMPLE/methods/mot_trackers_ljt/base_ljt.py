from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


REID_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "osnet_x0_25_msmt17.pt"


@dataclass(frozen=True)
class TrackerConfig:
    track_thresh: float
    match_thresh: float
    track_buffer: int
    max_age: int
    min_hits: int
    reid_weights: Path
    device: str
    fp16: bool
    frame_rate: int = 30


class BoxmotTrackerAdapter:
    """Small adapter that keeps each boxmot method behind the same update API."""

    tracker_name = ""
    boxmot_class_name = ""
    requires_reid = False

    def __init__(self, config: TrackerConfig):
        self.config = config
        if self.requires_reid and not config.reid_weights.exists():
            raise RuntimeError(
                f"{self.tracker_name} requires ReID weights. "
                f"Expected: {config.reid_weights}. "
                "Set reid_weights in the GUI/CLI or add the weight file under weights/."
            )
        self.tracker = self._build_tracker()

    def _boxmot_class(self):
        try:
            import boxmot
        except ImportError as exc:
            raise RuntimeError("boxmot is required for this MOT method. Install project tracking dependencies first.") from exc
        return getattr(boxmot, self.boxmot_class_name)

    def _build_tracker(self):
        cls = self._boxmot_class()
        kwargs = self._constructor_kwargs()
        return cls(**kwargs)

    def _constructor_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError

    def update(self, det_array, frame):
        return self.tracker.update(det_array, frame)


def bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def tracker_config_from_args(args: Any) -> TrackerConfig:
    reid_weights = Path(getattr(args, "reid_weights", REID_DEFAULT_WEIGHTS) or REID_DEFAULT_WEIGHTS)
    return TrackerConfig(
        track_thresh=float(getattr(args, "byte_track_thresh", 0.25)),
        match_thresh=float(getattr(args, "byte_match_thresh", 0.8)),
        track_buffer=int(getattr(args, "byte_track_buffer", 30)),
        max_age=int(getattr(args, "max_missed", 30)),
        min_hits=int(getattr(args, "mot_min_hits", 3)),
        reid_weights=reid_weights,
        device=str(getattr(args, "reid_device", "cpu") or "cpu"),
        fp16=bool_arg(getattr(args, "reid_fp16", False)),
    )
