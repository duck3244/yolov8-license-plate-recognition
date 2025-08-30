"""
config_manager.py
시스템 설정 관리

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """모델 설정"""
    yolo_model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    device: str = "auto"

@dataclass
class OCRConfig:
    """OCR 설정"""
    tesseract_cmd: Optional[str] = None
    languages: str = "kor+eng"

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    path: str = "license_plates.db"

@dataclass
class WebConfig:
    """웹 인터페이스 설정"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    upload_folder: str = "uploads"

@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    file: str = "license_plate_system.log"

@dataclass
class SystemConfig:
    """전체 시스템 설정"""
    model: ModelConfig
    ocr: OCRConfig
    database: DatabaseConfig
    web: WebConfig
    logging: LoggingConfig

    def __init__(self):
        self.model = ModelConfig()
        self.ocr = OCRConfig()
        self.database = DatabaseConfig()
        self.web = WebConfig()
        self.logging = LoggingConfig()

class ConfigManager:
    """설정 관리자"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> SystemConfig:
        """설정 파일 로드 (YAML이 없으면 기본 설정 사용)"""
        try:
            # YAML 파일이 있으면 로드 시도
            if self.config_path.exists():
                try:
                    import yaml
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f) or {}

                    logger.info(f"설정 파일 로드 완료: {self.config_path}")
                    return self._parse_config_data(config_data)
                except ImportError:
                    logger.warning("PyYAML이 설치되지 않았습니다. 기본 설정을 사용합니다.")
                except Exception as e:
                    logger.error(f"설정 파일 로드 오류: {e}")

            # 기본 설정 사용
            logger.info("기본 설정을 사용합니다.")
            return SystemConfig()

        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            return SystemConfig()

    def _parse_config_data(self, config_data: Dict[str, Any]) -> SystemConfig:
        """설정 데이터 파싱"""
        try:
            config = SystemConfig()

            # 모델 설정
            if 'model' in config_data:
                model_data = config_data['model']
                config.model.yolo_model_path = model_data.get('yolo_model_path', config.model.yolo_model_path)
                config.model.confidence_threshold = model_data.get('confidence_threshold', config.model.confidence_threshold)
                config.model.device = model_data.get('device', config.model.device)

            # OCR 설정
            if 'ocr' in config_data:
                ocr_data = config_data['ocr']
                config.ocr.tesseract_cmd = ocr_data.get('tesseract_cmd', config.ocr.tesseract_cmd)
                config.ocr.languages = ocr_data.get('languages', config.ocr.languages)

            # 데이터베이스 설정
            if 'database' in config_data:
                db_data = config_data['database']
                config.database.path = db_data.get('path', config.database.path)

            # 웹 설정
            if 'web' in config_data:
                web_data = config_data['web']
                config.web.host = web_data.get('host', config.web.host)
                config.web.port = web_data.get('port', config.web.port)
                config.web.debug = web_data.get('debug', config.web.debug)
                config.web.upload_folder = web_data.get('upload_folder', config.web.upload_folder)

            # 로깅 설정
            if 'logging' in config_data:
                log_data = config_data['logging']
                config.logging.level = log_data.get('level', config.logging.level)
                config.logging.file = log_data.get('file', config.logging.file)

            return config

        except Exception as e:
            logger.error(f"설정 데이터 파싱 오류: {e}")
            return SystemConfig()

    def validate_config(self) -> Dict[str, Any]:
        """설정 검증"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            # 모델 파일 존재 확인
            model_path = Path(self.config.model.yolo_model_path)
            if not model_path.exists() and not model_path.name.startswith('yolov8'):
                validation_results['warnings'].append(
                    f"모델 파일이 존재하지 않습니다: {model_path}"
                )

            # 신뢰도 임계값 범위 확인
            if not 0.0 <= self.config.model.confidence_threshold <= 1.0:
                validation_results['errors'].append(
                    f"신뢰도 임계값이 범위를 벗어났습니다: {self.config.model.confidence_threshold}"
                )
                validation_results['valid'] = False

            # 포트 번호 확인
            if not 1 <= self.config.web.port <= 65535:
                validation_results['errors'].append(
                    f"잘못된 포트 번호: {self.config.web.port}"
                )
                validation_results['valid'] = False

            # 업로드 폴더 생성 가능 여부 확인
            upload_dir = Path(self.config.web.upload_folder)
            try:
                upload_dir.mkdir(exist_ok=True)
            except Exception as e:
                validation_results['errors'].append(
                    f"업로드 폴더를 생성할 수 없습니다: {e}"
                )
                validation_results['valid'] = False

            # 데이터베이스 디렉토리 확인
            db_path = Path(self.config.database.path)
            db_dir = db_path.parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    validation_results['errors'].append(
                        f"데이터베이스 디렉토리를 생성할 수 없습니다: {e}"
                    )
                    validation_results['valid'] = False

        except Exception as e:
            validation_results['errors'].append(f"설정 검증 중 오류: {e}")
            validation_results['valid'] = False

        return validation_results

    def get_config_value(self, path: str, default=None):
        """
        점 표기법으로 설정값 가져오기
        예: get_config_value('model.confidence_threshold')
        """
        try:
            keys = path.split('.')
            value = self.config

            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return default

            return value

        except Exception:
            return default

if __name__ == "__main__":
    # 테스트 코드
    print("⚙️ 설정 관리자 테스트")

    config_manager = ConfigManager()

    print("   - 기본 설정:")
    print(f"     모델 경로: {config_manager.config.model.yolo_model_path}")
    print(f"     신뢰도 임계값: {config_manager.config.model.confidence_threshold}")
    print(f"     웹 포트: {config_manager.config.web.port}")
    print(f"     데이터베이스 경로: {config_manager.config.database.path}")

    # 설정 검증
    validation = config_manager.validate_config()
    print(f"   - 설정 유효성: {'✅' if validation['valid'] else '❌'}")

    if validation['errors']:
        print("   - 오류:")
        for error in validation['errors']:
            print(f"     {error}")

    if validation['warnings']:
        print("   - 경고:")
        for warning in validation['warnings']:
            print(f"     {warning}")