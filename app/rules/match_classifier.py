"""
Optional lightweight classifier for person-to-vehicle association.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.types import Detection
from app.rules.match_features import FEATURE_NAMES, extract_match_features


class PersonVehicleMatchClassifier:
    """Load and score a trained sklearn-style match classifier."""

    def __init__(self, model_path: Path, threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        self.feature_names = FEATURE_NAMES
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, person: Detection, vehicle: Detection, frame_width: int = 0) -> Optional[float]:
        if self.model is None:
            return None

        features = extract_match_features(person, vehicle, frame_width=frame_width)
        vector = [features.as_vector(self.feature_names)]
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(vector)[0][1])
        if hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(vector)[0])
            return 1.0 / (1.0 + pow(2.718281828, -decision))
        prediction = self.model.predict(vector)[0]
        return float(prediction)

    def is_match(self, person: Detection, vehicle: Detection, frame_width: int = 0) -> bool:
        score = self.score(person, vehicle, frame_width=frame_width)
        return score is not None and score >= self.threshold

    def _load(self) -> None:
        if not self.model_path.exists():
            return

        import joblib

        payload = joblib.load(self.model_path)
        if isinstance(payload, dict):
            self.model = payload.get("model")
            self.feature_names = payload.get("feature_names") or FEATURE_NAMES
            self.threshold = float(payload.get("threshold", self.threshold))
        else:
            self.model = payload
