"""
api/deps.py
FastAPI 의존성 — 무거운 객체(YOLO 모델, OCR, DB)는 앱 수명 동안 1개만.

추론 직렬화용 lock도 여기서 관리한다. Ultralytics YOLO와 Pororo brainocr는
thread-safe로 문서화돼 있지 않으므로, threadpool에서 실행되는 동기 엔드포인트
사이에서 동시 호출을 막아야 한다.
"""
from __future__ import annotations

import threading
from functools import lru_cache

from database_manager import DatabaseManager
from license_plate_recognizer import YOLOv8LicensePlateRecognizer

from api.settings import get_settings


# 추론 직렬화 락 (모듈 레벨 = 프로세스 전역)
inference_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_recognizer() -> YOLOv8LicensePlateRecognizer:
    settings = get_settings()
    return YOLOv8LicensePlateRecognizer(
        yolo_model_path=settings.yolo_model_path,
        confidence_threshold=settings.confidence_threshold,
        ocr_engine=settings.ocr_engine,
    )


@lru_cache(maxsize=1)
def get_db() -> DatabaseManager:
    settings = get_settings()
    return DatabaseManager(settings.db_path)
