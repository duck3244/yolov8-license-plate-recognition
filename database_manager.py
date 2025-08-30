"""
database_manager.py
번호판 인식 결과 데이터베이스 관리

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlateDetection:
    """번호판 탐지 결과 데이터 클래스"""
    plate_number: str
    confidence: float
    timestamp: datetime
    image_path: Optional[str] = None
    bbox: Optional[tuple] = None
    processing_time: Optional[float] = None

class DatabaseManager:
    """번호판 인식 결과 데이터베이스 관리"""

    def __init__(self, db_path: str = "license_plates.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    image_path TEXT,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    processing_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate_number ON detections(plate_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')

            conn.commit()
            conn.close()
            logger.info(f"데이터베이스 초기화 완료: {self.db_path}")

        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")

    def save_detection(self, detection: PlateDetection) -> bool:
        """탐지 결과 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            bbox_data = detection.bbox if detection.bbox else (None, None, None, None)

            cursor.execute('''
                INSERT INTO detections 
                (plate_number, confidence, timestamp, image_path, 
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection.plate_number,
                detection.confidence,
                detection.timestamp,
                detection.image_path,
                bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3],
                detection.processing_time
            ))

            conn.commit()
            conn.close()

            logger.debug(f"탐지 결과 저장 완료: {detection.plate_number}")
            return True

        except Exception as e:
            logger.error(f"탐지 결과 저장 실패: {e}")
            return False

    def get_detections(self,
                      plate_number: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 100) -> List[Dict]:
        """탐지 결과 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM detections WHERE 1=1"
            params = []

            if plate_number:
                query += " AND plate_number LIKE ?"
                params.append(f"%{plate_number}%")

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                results.append(result)

            conn.close()
            return results

        except Exception as e:
            logger.error(f"탐지 결과 조회 실패: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 총 탐지 수
            cursor.execute("SELECT COUNT(*) FROM detections")
            total_detections = cursor.fetchone()[0]

            # 고유 번호판 수
            cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM detections")
            unique_plates = cursor.fetchone()[0]

            # 오늘 탐지 수
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(*) FROM detections WHERE DATE(timestamp) = ?", (today,))
            today_detections = cursor.fetchone()[0]

            # 평균 신뢰도
            cursor.execute("SELECT AVG(confidence) FROM detections")
            avg_confidence = cursor.fetchone()[0] or 0

            # 평균 처리 시간
            cursor.execute("SELECT AVG(processing_time) FROM detections WHERE processing_time IS NOT NULL")
            avg_processing_time = cursor.fetchone()[0] or 0

            conn.close()

            return {
                'total_detections': total_detections,
                'unique_plates': unique_plates,
                'today_detections': today_detections,
                'avg_confidence': round(avg_confidence, 3),
                'avg_processing_time': round(avg_processing_time, 3),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {
                'total_detections': 0,
                'unique_plates': 0,
                'today_detections': 0,
                'avg_confidence': 0,
                'avg_processing_time': 0,
                'error': str(e)
            }

if __name__ == "__main__":
    # 테스트 코드
    db_manager = DatabaseManager("test_license_plates.db")

    # 샘플 데이터 추가
    sample_detection = PlateDetection(
        plate_number="12가3456",
        confidence=0.85,
        timestamp=datetime.now(),
        image_path="test.jpg",
        bbox=(100, 200, 300, 250),
        processing_time=0.5
    )

    print("📊 데이터베이스 테스트")
    print(f"   - 샘플 데이터 저장: {db_manager.save_detection(sample_detection)}")

    # 통계 확인
    stats = db_manager.get_statistics()
    print("   - 통계 정보:")
    for key, value in stats.items():
        print(f"     {key}: {value}")

    # 검색 테스트
    detections = db_manager.get_detections(limit=10)
    print(f"   - 최근 탐지 결과: {len(detections)}개")