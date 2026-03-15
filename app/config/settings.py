"""
Pydantic-based centralized configuration management.
All configuration is grouped by domain and validated at startup.
"""
import os
from pathlib import Path
from typing import Set, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env.flask instead of .env (to avoid conflict with Gemini CLI)
load_dotenv(BASE_DIR / ".env.flask")

# Keep Ultralytics runtime state inside the repo to avoid per-user profile permission issues.
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))


class AppSettings(BaseSettings):
    """Flask application settings."""
    secret_key: str = Field(default="dev-secret-key")
    debug: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="")


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    url: str = Field(
        default="mysql+pymysql://root:password@localhost/ebike_detection",
        alias="DATABASE_URL"
    )
    track_modifications: bool = Field(default=False)

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)


class StorageSettings(BaseSettings):
    """File storage settings."""
    upload_folder: Path = Field(default=BASE_DIR / "static" / "uploads")
    output_folder: Path = Field(default=BASE_DIR / "static" / "outputs")
    max_content_length: int = Field(default=500 * 1024 * 1024)  # 500MB
    allowed_extensions: Set[str] = Field(
        default={"png", "jpg", "jpeg", "mp4", "avi", "mov"}
    )

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)


class ModelSettings(BaseSettings):
    """YOLO model settings."""
    model_path: Path = Field(default=BASE_DIR / "models" / "bifpn_best.pt", alias="MODEL_PATH")
    candidate_model_path: Path = Field(default=BASE_DIR / "yolov8n.pt")
    conf_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_thresh: float = Field(default=0.45, ge=0.0, le=1.0)
    use_tensorrt: bool = Field(default=False)
    tensorrt_engine_path: Optional[Path] = Field(default=None)
    imgsz: int = Field(default=640)
    refine_imgsz: int = Field(default=1280)
    half: bool = Field(default=True)
    enable_candidate_refine: bool = Field(default=True)
    candidate_conf_thresh: float = Field(default=0.15, ge=0.0, le=1.0)
    refine_conf_thresh: float = Field(default=0.02, ge=0.0, le=1.0)
    enable_tile_refine: bool = Field(default=False)
    tile_refine_imgsz: int = Field(default=1280)
    tile_refine_conf_thresh: float = Field(default=0.005, ge=0.0, le=1.0)
    tile_refine_overlap: float = Field(default=0.40, ge=0.0, le=0.9)
    tile_refine_width_ratio: float = Field(default=0.55, gt=0.2, lt=1.0)
    tile_refine_height_ratio: float = Field(default=0.65, gt=0.2, lt=1.0)
    enable_temporal_stabilize: bool = Field(default=True)
    temporal_max_age: int = Field(default=6, ge=0)
    temporal_decay: float = Field(default=0.88, gt=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)

    @field_validator("model_path", "candidate_model_path", mode="before")
    @classmethod
    def resolve_model_path(cls, v):
        if isinstance(v, str):
            p = Path(v)
            if not p.is_absolute():
                p = BASE_DIR / p
            return p
        return v


class DetectionSettings(BaseSettings):
    """Detection class configuration."""
    class_names: List[str] = Field(
        default=["ebike", "driver", "passenger", "helmet"]
    )
    ebike_class_id: int = Field(default=0)
    driver_class_id: int = Field(default=1)
    passenger_class_id: int = Field(default=2)
    helmet_class_id: int = Field(default=3)
    helmet_detection_enabled: bool = Field(default=False)
    conf_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_thresh: float = Field(default=0.45, ge=0.0, le=1.0)


class TrackingSettings(BaseSettings):
    """ByteTrack tracking settings."""
    enabled: bool = Field(default=True)
    tracker_type: str = Field(default="bytetrack")
    track_thresh: float = Field(default=0.5)
    track_buffer: int = Field(default=30)
    match_thresh: float = Field(default=0.8)
    frame_rate: int = Field(default=30)
    fallback_iou_thresh: float = Field(default=0.18)
    fallback_max_missed: int = Field(default=6)
    fallback_min_confirmed_hits: int = Field(default=2)
    fallback_smooth_alpha: float = Field(default=0.65)
    fallback_conf_decay: float = Field(default=0.92)


class RuleSettings(BaseSettings):
    """Violation rule engine settings."""
    passenger_rule_enabled: bool = Field(default=True)
    helmet_rule_enabled: bool = Field(default=False)
    helmet_head_ratio: float = Field(default=0.35)
    helmet_overlap_threshold: float = Field(default=0.3)


class ViolationSettings(BaseSettings):
    """Violation deduplication FSM settings."""
    min_frames_to_confirm: int = Field(default=3)
    cooldown_frames: int = Field(default=30)
    max_gap_frames: int = Field(default=10)
    max_age_seconds: float = Field(default=5.0)


class StartupSettings(BaseSettings):
    """Startup/self-check behavior settings."""
    strict: bool = Field(default=False, alias="STARTUP_STRICT")
    check_database: bool = Field(default=True, alias="STARTUP_CHECK_DATABASE")
    check_model: bool = Field(default=True, alias="STARTUP_CHECK_MODEL")

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)


class Settings:
    """Aggregated settings container."""

    def __init__(self):
        self.app = AppSettings()
        self.database = DatabaseSettings()
        self.storage = StorageSettings()
        self.model = ModelSettings()
        self.detection = DetectionSettings()
        self.tracking = TrackingSettings()
        self.rules = RuleSettings()
        self.violations = ViolationSettings()
        self.startup = StartupSettings()

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.storage.ensure_dirs()
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


class FlaskConfig:
    """Flask-compatible configuration class."""

    def __init__(self, s: Settings = None):
        self._settings = s or settings

    @property
    def SECRET_KEY(self):
        return self._settings.app.secret_key

    @property
    def DEBUG(self):
        return self._settings.app.debug

    @property
    def SQLALCHEMY_DATABASE_URI(self):
        return self._settings.database.url

    @property
    def SQLALCHEMY_TRACK_MODIFICATIONS(self):
        return self._settings.database.track_modifications

    @property
    def UPLOAD_FOLDER(self):
        return str(self._settings.storage.upload_folder)

    @property
    def OUTPUT_FOLDER(self):
        return str(self._settings.storage.output_folder)

    @property
    def MAX_CONTENT_LENGTH(self):
        return self._settings.storage.max_content_length

    @property
    def ALLOWED_EXTENSIONS(self):
        return self._settings.storage.allowed_extensions

    @property
    def MODEL_PATH(self):
        return str(self._settings.model.model_path)

    @property
    def CONFIDENCE_THRESHOLD(self):
        return self._settings.model.conf_thresh

    @property
    def IOU_THRESHOLD(self):
        return self._settings.model.iou_thresh

    @property
    def CLASS_NAMES(self):
        return self._settings.detection.class_names


# Flask config instance
Config = FlaskConfig()
