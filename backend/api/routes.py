"""
api/routes.py
FastAPI 라우터.

* /api/detect  ─ 이미지 업로드 + 번호판 인식
* /api/history ─ 인식 이력 조회
* /api/statistics ─ 통계
* /api/results/{id}/image ─ 결과 이미지(JPEG) 단일 응답
* /api/health  ─ 헬스체크 (가벼운 SELECT 1)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from database_manager import PlateDetection

from api import deps
from api.schemas import (
    DetectResponse,
    DetectionItem,
    HealthResponse,
    HistoryResponse,
    Statistics,
    StatisticsResponse,
)
from api.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _validate_extension(filename: str, allowed: tuple[str, ...]) -> str:
    """확장자 검증 후 lowercase 확장자 반환. 실패 시 HTTPException."""
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 확장자: {ext}")
    return ext


@router.post("/detect", response_model=DetectResponse)
def detect(
    image: UploadFile = File(...),
    recognizer=Depends(deps.get_recognizer),
    db=Depends(deps.get_db),
) -> DetectResponse:
    """이미지 업로드 → YOLO 검출 → OCR → DB 저장."""
    settings = get_settings()

    if not image.filename:
        raise HTTPException(status_code=400, detail="파일이 선택되지 않았습니다")

    ext = _validate_extension(image.filename, settings.allowed_extensions)

    # 저장 파일명은 uuid 기반(한글 보존 + 충돌 방지). 원본명은 DB에 따로 저장.
    saved_name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex}{ext}"
    upload_dir = Path(settings.upload_folder)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = (upload_dir / saved_name).resolve()
    if upload_dir.resolve() not in saved_path.parents:
        raise HTTPException(status_code=400, detail="업로드 경로 위반")

    # SpooledTemporaryFile에서 디스크로 옮기기 (크기는 미들웨어/server가 한도 강제)
    content = image.file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다")

    saved_path.write_bytes(content)

    # 모델은 thread-safe 아니므로 추론 구간만 락
    start = time.time()
    try:
        with deps.inference_lock:
            plate_text, result_img = recognizer.process_image(
                str(saved_path), save_result=False
            )
        processing_time = time.time() - start
    finally:
        if settings.delete_after_process:
            try:
                saved_path.unlink()
            except OSError as e:
                logger.debug(f"업로드 파일 삭제 실패: {e}")

    if not plate_text:
        return DetectResponse(
            success=False,
            processing_time=round(processing_time, 3),
            error="번호판을 인식할 수 없습니다",
        )

    # 결과 이미지를 JPEG로 인코딩해 DB BLOB으로 저장
    ok, buf = cv2.imencode(".jpg", result_img)
    result_bytes = buf.tobytes() if ok else None

    detection = PlateDetection(
        plate_number=plate_text,
        confidence=0.8,  # OCR 자체 신뢰도가 없으므로 고정값. 추후 모델 출력 활용 시 교체.
        timestamp=datetime.now(),
        image_path=None,
        processing_time=processing_time,
        original_filename=image.filename,
        result_image=result_bytes,
    )
    detection_id = db.save_detection(detection)
    if detection_id is None:
        raise HTTPException(status_code=500, detail="DB 저장 실패")

    return DetectResponse(
        success=True,
        detection_id=detection_id,
        plate_number=plate_text,
        confidence=detection.confidence,
        processing_time=round(processing_time, 3),
        result_image_url=f"/api/results/{detection_id}/image",
    )


@router.get("/results/{detection_id}/image")
def get_result_image(detection_id: int, db=Depends(deps.get_db)) -> Response:
    img = db.get_result_image(detection_id)
    if img is None:
        raise HTTPException(status_code=404, detail="결과 이미지 없음")
    return Response(content=img, media_type="image/jpeg")


@router.get("/history", response_model=HistoryResponse)
def get_history(
    plate_number: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db=Depends(deps.get_db),
) -> HistoryResponse:
    rows = db.get_detections(plate_number=plate_number, limit=limit)
    items = [
        DetectionItem(
            id=r["id"],
            plate_number=r["plate_number"],
            confidence=r["confidence"],
            timestamp=r["timestamp"],
            original_filename=r.get("original_filename"),
            processing_time=r.get("processing_time"),
            result_image_url=f"/api/results/{r['id']}/image",
        )
        for r in rows
    ]
    return HistoryResponse(count=len(items), detections=items)


@router.get("/statistics", response_model=StatisticsResponse)
def get_statistics(db=Depends(deps.get_db)) -> StatisticsResponse:
    stats = db.get_statistics()
    return StatisticsResponse(statistics=Statistics(**{
        k: stats.get(k) for k in Statistics.model_fields.keys()
    }))


@router.get("/health", response_model=HealthResponse)
def health(db=Depends(deps.get_db)) -> HealthResponse:
    db_ok = db.ping()
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        timestamp=datetime.now(),
    )
