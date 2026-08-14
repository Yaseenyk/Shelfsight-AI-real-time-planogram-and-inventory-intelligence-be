"""Central application configuration.

All tunables are environment-driven so that experiment runs (which must be
reproducible for the paper) can be pinned via a committed `.env` snapshot.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- Application ---------------------------------------------------
    APP_NAME: str = "ShelfSight AI"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # --- CORS ------------------------------------------------------------
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # --- Storage ---------------------------------------------------------
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    PLANOGRAM_DIR: Path = BASE_DIR / "data" / "planograms"
    REPORTS_DIR: Path = BASE_DIR / "evaluation" / "reports"

    # --- Database --------------------------------------------------------
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'shelfsight.db').as_posix()}"
    SQL_ECHO: bool = False

    # --- Detection (YOLOv8) ----------------------------------------------
    DETECTION_WEIGHTS: Path = BASE_DIR / "models" / "weights" / "yolov8n.pt"
    DETECTION_CONF_THRESHOLD: float = 0.35
    DETECTION_IOU_NMS: float = 0.45
    DETECTION_IMG_SIZE: int = 640
    DETECTION_DEVICE: str = "cpu"  # "cpu" | "cuda:0"
    DETECTION_MAX_DET: int = 300  # per frame; a dense shelf bay rarely exceeds ~150
    #: Class-agnostic NMS. On a shelf, two *different* SKU classes should never
    #: occupy the same box — a duplicate there is a detector error, not two facings.
    DETECTION_AGNOSTIC_NMS: bool = True
    #: Second NMS pass in our own code. Ultralytics already suppresses per class;
    #: this catches cross-class duplicates left behind when agnostic NMS is off.
    DETECTION_EXTRA_NMS: bool = False
    #: Drop boxes smaller than this fraction of the frame (label specks, artefacts).
    DETECTION_MIN_BOX_AREA: float = 0.0001
    #: Let Ultralytics fetch the pretrained checkpoint when weights are missing.
    #: Set False for air-gapped lab machines and reproducible published runs.
    DETECTION_ALLOW_DOWNLOAD: bool = True
    DETECTION_BASE_MODEL: str = "yolov8n.pt"
    #: Maps detector class names to catalogue SKUs (see data/class_map.json).
    DETECTION_CLASS_MAP: Path = BASE_DIR / "data" / "class_map.json"
    DETECTION_WARMUP: bool = True  # burn the first slow inference at startup

    # --- Planogram compliance --------------------------------------------
    COMPLIANCE_IOU_THRESHOLD: float = 0.50
    COMPLIANCE_CENTER_DISTANCE_THRESHOLD: float = 0.08  # normalised units
    COMPLIANCE_ROW_BAND_TOLERANCE: float = 0.05  # y-band used for row clustering

    # --- Freshness classifier --------------------------------------------
    FRESHNESS_WEIGHTS: Path = BASE_DIR / "models" / "weights" / "freshness_mobilenetv2.pt"
    FRESHNESS_BACKBONE: str = "mobilenet_v2"  # "mobilenet_v2" | "resnet50"
    FRESHNESS_INPUT_SIZE: int = 224
    FRESHNESS_CLASSES: List[str] = Field(
        default_factory=lambda: ["fresh", "ripening", "spoiled"]
    )

    # --- OCR expiry engine ------------------------------------------------
    OCR_LANGUAGES: List[str] = Field(default_factory=lambda: ["en"])
    OCR_GPU: bool = False
    OCR_MIN_CONFIDENCE: float = 0.30
    EXPIRY_NEAR_THRESHOLD_DAYS: int = 7
    EXPIRY_DAYFIRST: bool = True  # DD/MM/YYYY is the dominant retail format

    # --- Ollama (local LLM insights) --------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TIMEOUT_S: float = 120.0
    OLLAMA_TEMPERATURE: float = 0.2
    OLLAMA_NUM_PREDICT: int = 512

    @field_validator("CORS_ORIGINS", "FRESHNESS_CLASSES", "OCR_LANGUAGES", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow `A,B,C` style env values in addition to JSON arrays."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def ensure_directories(self) -> None:
        for path in (self.DATA_DIR, self.UPLOAD_DIR, self.PLANOGRAM_DIR, self.REPORTS_DIR):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
