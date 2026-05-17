"""
api/settings.py
pydantic-settings 기반 환경/설정 통합.

.env 파일 또는 환경변수에서 자동으로 값을 읽으며, 기본값은 단일 사용자 MVP에
적합한 안전한 값(127.0.0.1, dev 시크릿)으로 설정된다.
"""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/ 디렉토리를 기준점으로 고정 (어디서 uvicorn을 띄우든 동일 경로를 가리키도록).
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- 서버 -----
    host: str = Field(default="127.0.0.1", description="기본은 localhost. 외부 노출은 reverse proxy 권장.")
    port: int = 8000

    # ----- 보안 -----
    # 운영에서는 반드시 환경변수로 주입. 미설정 시 dev용 임시 키.
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))

    # ----- 업로드 -----
    upload_folder: str = str(BACKEND_DIR / "uploads")
    max_upload_mb: int = 16
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    cleanup_age_hours: int = 24
    delete_after_process: bool = True

    # ----- DB / 모델 -----
    db_path: str = str(BACKEND_DIR / "license_plates.db")
    yolo_model_path: str = str(BACKEND_DIR / "license_plate_det_yolov8.pt")
    confidence_threshold: float = 0.3
    ocr_engine: str = "auto"

    # ----- CORS -----
    # Vite dev 서버. 운영(정적 마운트)에선 same-origin이라 의미 없음.
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    # ----- 로깅 -----
    log_level: str = "INFO"
    log_file: str = str(BACKEND_DIR / "logs" / "license_plate_system.log")
    log_max_mb: int = 10
    log_backup_count: int = 5

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 수명 동안 1회만 로드되는 설정 싱글톤."""
    return Settings()
