"""
api/schemas.py
요청/응답용 Pydantic v2 모델.

내부 dataclass(`PlateDetection`)는 도메인 모델, 이 파일은 API 경계 모델.
이 둘을 섞지 않는 것이 의도.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DetectResponse(BaseModel):
    """POST /api/detect 응답."""
    success: bool
    detection_id: Optional[int] = None
    plate_number: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: float = Field(description="초 단위")
    result_image_url: Optional[str] = None
    error: Optional[str] = None


class DetectionItem(BaseModel):
    id: int
    plate_number: str
    confidence: float
    timestamp: datetime
    original_filename: Optional[str] = None
    processing_time: Optional[float] = None
    result_image_url: Optional[str] = None


class HistoryResponse(BaseModel):
    success: bool = True
    count: int
    detections: list[DetectionItem]


class Statistics(BaseModel):
    total_detections: int
    unique_plates: int
    today_detections: int
    avg_confidence: float
    avg_processing_time: float
    last_updated: str


class StatisticsResponse(BaseModel):
    success: bool = True
    statistics: Statistics


class HealthResponse(BaseModel):
    status: str
    database: str
    timestamp: datetime
