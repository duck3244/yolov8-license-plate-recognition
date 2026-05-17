"""
database_manager.py
번호판 인식 결과 데이터베이스 관리

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import sqlite3
from contextlib import contextmanager
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
    # 클라이언트가 업로드한 원본 파일명 (한글 포함 가능). 저장 파일명과 별개.
    original_filename: Optional[str] = None
    # 결과 이미지(JPEG bytes). DB BLOB으로 저장하여 별도 GET endpoint로 서빙.
    # MVP 단일 사용자 규모에서는 디스크 분산보다 단순함이 더 가치 있음.
    result_image: Optional[bytes] = None

class DatabaseManager:
    """번호판 인식 결과 데이터베이스 관리"""

    def __init__(self, db_path: str = "license_plates.db"):
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def _connect(self):
        """예외 발생 시 자동으로 rollback + close 되는 sqlite3 연결.

        sqlite3 자체는 thread-local connection을 권장하므로 호출마다 새 연결을
        열고 닫는다 (이 모듈은 ThreadPoolExecutor 환경에서도 사용됨).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        """데이터베이스 초기화"""
        try:
            with self._connect() as conn:
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
                        original_filename TEXT,
                        result_image BLOB,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate_number ON detections(plate_number)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
                # 기존 DB에 새 컬럼이 없으면 추가 (idempotent)
                for col, ddl in [
                    ('original_filename', 'TEXT'),
                    ('result_image', 'BLOB'),
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE detections ADD COLUMN {col} {ddl}")
                    except sqlite3.OperationalError:
                        pass  # 이미 존재
            logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise

    def save_detection(self, detection: PlateDetection) -> Optional[int]:
        """탐지 결과 저장. 성공 시 새 row id, 실패 시 None."""
        bbox_data = detection.bbox if detection.bbox else (None, None, None, None)
        try:
            with self._connect() as conn:
                cursor = conn.execute('''
                    INSERT INTO detections
                    (plate_number, confidence, timestamp, image_path,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, processing_time,
                     original_filename, result_image)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection.plate_number,
                    detection.confidence,
                    detection.timestamp,
                    detection.image_path,
                    bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3],
                    detection.processing_time,
                    detection.original_filename,
                    detection.result_image,
                ))
                row_id = cursor.lastrowid
            logger.debug(f"탐지 결과 저장 완료: id={row_id} plate={detection.plate_number}")
            return row_id
        except sqlite3.Error as e:
            logger.error(f"탐지 결과 저장 실패: {e}")
            return None

    def get_result_image(self, detection_id: int) -> Optional[bytes]:
        """저장된 결과 이미지(JPEG bytes)를 조회."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT result_image FROM detections WHERE id = ?",
                    (detection_id,),
                ).fetchone()
            if row and row[0]:
                return row[0]
            return None
        except sqlite3.Error as e:
            logger.error(f"결과 이미지 조회 실패: {e}")
            return None

    def get_detections(self,
                      plate_number: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 100) -> List[Dict]:
        """탐지 결과 조회. BLOB 컬럼은 응답에서 제외."""
        query = (
            "SELECT id, plate_number, confidence, timestamp, image_path, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2, processing_time, "
            "original_filename, created_at "
            "FROM detections WHERE 1=1"
        )
        params: List[Any] = []

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

        try:
            with self._connect() as conn:
                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"탐지 결과 조회 실패: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 조회"""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM detections")
                total_detections = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM detections")
                unique_plates = cursor.fetchone()[0]

                today = datetime.now().date()
                cursor.execute(
                    "SELECT COUNT(*) FROM detections WHERE DATE(timestamp) = ?",
                    (today,),
                )
                today_detections = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(confidence) FROM detections")
                avg_confidence = cursor.fetchone()[0] or 0

                cursor.execute(
                    "SELECT AVG(processing_time) FROM detections "
                    "WHERE processing_time IS NOT NULL"
                )
                avg_processing_time = cursor.fetchone()[0] or 0

            return {
                'total_detections': total_detections,
                'unique_plates': unique_plates,
                'today_detections': today_detections,
                'avg_confidence': round(avg_confidence, 3),
                'avg_processing_time': round(avg_processing_time, 3),
                'last_updated': datetime.now().isoformat(),
            }
        except sqlite3.Error as e:
            logger.error(f"통계 조회 실패: {e}")
            return {
                'total_detections': 0,
                'unique_plates': 0,
                'today_detections': 0,
                'avg_confidence': 0,
                'avg_processing_time': 0,
                'error': str(e),
            }

    def ping(self) -> bool:
        """헬스체크용 경량 쿼리."""
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False


if __name__ == "__main__":
    db_manager = DatabaseManager("test_license_plates.db")

    sample_detection = PlateDetection(
        plate_number="12가3456",
        confidence=0.85,
        timestamp=datetime.now(),
        image_path="test.jpg",
        bbox=(100, 200, 300, 250),
        processing_time=0.5,
    )

    print("📊 데이터베이스 테스트")
    print(f"   - 샘플 데이터 저장: {db_manager.save_detection(sample_detection)}")

    stats = db_manager.get_statistics()
    print("   - 통계 정보:")
    for key, value in stats.items():
        print(f"     {key}: {value}")

    detections = db_manager.get_detections(limit=10)
    print(f"   - 최근 탐지 결과: {len(detections)}개")
