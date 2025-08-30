#!/usr/bin/env python3
"""
main_app.py
YOLOv8 번호판 인식 시스템 메인 실행 파일

Usage:
    python main_app.py [command] [options]

Commands:
    server      - 웹 서버 실행
    camera      - 실시간 카메라 처리
    image       - 단일 이미지 처리
    batch       - 배치 처리
    config      - 설정 관리

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

# 프로젝트 모듈들 (import 에러 방지를 위한 try-except)
try:
    from license_plate_recognizer import YOLOv8LicensePlateRecognizer
    from database_manager import DatabaseManager
    from web_interface import WebInterface
    from config_manager import ConfigManager
except ImportError as e:
    print(f"⚠️ 모듈 import 오류: {e}")
    print("필요한 파일들이 같은 디렉토리에 있는지 확인해주세요.")
    sys.exit(1)


class LicensePlateSystem:
    """번호판 인식 시스템 메인 클래스"""

    def __init__(self, config_path: str = "config.yaml"):
        """시스템 초기화"""
        # 로깅 기본 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 설정 관리자 초기화
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            # 기본 설정으로 계속 진행
            from config_manager import SystemConfig
            self.config = SystemConfig()

        # 컴포넌트 초기화
        self._initialize_components()

        self.logger.info("🚀 YOLOv8 번호판 인식 시스템 초기화 완료")

    def _initialize_components(self):
        """주요 컴포넌트 초기화"""
        try:
            # 환경 변수에서 OCR 엔진 설정 확인
            ocr_engine = os.environ.get('OCR_ENGINE', 'auto')

            # 번호판 인식기 초기화 (OCR 엔진 지정)
            self.recognizer = YOLOv8LicensePlateRecognizer(
                yolo_model_path=getattr(self.config.model, 'yolo_model_path', 'yolov8n.pt'),
                confidence_threshold=getattr(self.config.model, 'confidence_threshold', 0.5),
                ocr_engine=ocr_engine  # OCR 엔진 추가
            )

            # 데이터베이스 매니저 초기화
            self.db_manager = DatabaseManager(
                getattr(self.config.database, 'path', 'license_plates.db')
            )

            # 웹 인터페이스 초기화
            self.web_interface = WebInterface(
                self.recognizer,
                self.db_manager,
                getattr(self.config.web, 'upload_folder', 'uploads')
            )

            self.logger.info("✅ 모든 컴포넌트 초기화 완료")
            self.logger.info(f"🔍 사용 중인 OCR 엔진: {getattr(self.recognizer, 'ocr_engine', 'unknown')}")

        except Exception as e:
            self.logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            raise

    def run_server(self, host=None, port=None, debug=None):
        """웹 서버 실행"""
        try:
            # 설정값 우선순위: 인수 > 설정파일 > 기본값
            server_host = host or getattr(self.config.web, 'host', '0.0.0.0')
            server_port = port or getattr(self.config.web, 'port', 5000)
            server_debug = debug if debug is not None else getattr(self.config.web, 'debug', False)

            self.logger.info(f"🌐 웹 서버 시작: http://{server_host}:{server_port}")

            self.web_interface.run(
                host=server_host,
                port=server_port,
                debug=server_debug
            )

        except Exception as e:
            self.logger.error(f"웹 서버 실행 실패: {e}")
            raise

    def process_image(self, image_path: str, show_result: bool = True):
        """단일 이미지 처리"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

            self.logger.info(f"🖼️ 이미지 처리 시작: {image_path}")
            start_time = time.time()

            plate_text, result_img = self.recognizer.process_image(image_path)
            processing_time = time.time() - start_time

            print(f"🚗 인식된 번호판: {plate_text}")
            print(f"⏱️ 처리 시간: {processing_time:.3f}초")

            # 결과 표시 (matplotlib 사용)
            if show_result and result_img is not None:
                try:
                    import matplotlib.pyplot as plt
                    import cv2

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # 원본 이미지
                    original_img = cv2.imread(image_path)
                    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                    ax1.set_title('원본 이미지')
                    ax1.axis('off')

                    # 결과 이미지
                    ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                    ax2.set_title(f'결과: {plate_text}')
                    ax2.axis('off')

                    plt.tight_layout()
                    plt.show()

                except ImportError:
                    self.logger.warning("matplotlib를 사용할 수 없습니다. 결과 표시를 건너뜁니다.")

        except Exception as e:
            self.logger.error(f"이미지 처리 실패: {e}")


def main():
    """메인 실행 함수"""

    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(
        description='YOLOv8 기반 차량 번호판 인식 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:
  python main_app.py server                              # 웹 서버 실행
  python main_app.py image car.jpg                       # 단일 이미지 처리
  python main_app.py config view                         # 설정 확인
        """
    )

    parser.add_argument('--config', default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')

    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령들')

    # 웹 서버 명령
    server_parser = subparsers.add_parser('server', help='웹 서버 실행')
    server_parser.add_argument('--host', help='서버 호스트 (기본값: 0.0.0.0)')
    server_parser.add_argument('--port', type=int, help='서버 포트 (기본값: 5000)')
    server_parser.add_argument('--debug', action='store_true', help='디버그 모드')

    # 단일 이미지 처리 명령
    image_parser = subparsers.add_parser('image', help='단일 이미지 처리')
    image_parser.add_argument('image_path', help='처리할 이미지 경로')
    image_parser.add_argument('--no-display', action='store_true', help='결과 표시 안함')
    image_parser.add_argument('--ocr-engine', choices=['auto', 'pororo', 'paddleocr', 'easyocr', 'tesseract'],
                              default='auto', help='사용할 OCR 엔진 (기본값: auto)')
    image_parser.add_argument('--confidence', type=float, default=0.5, help='YOLO 신뢰도 임계값')

    # 설정 관리 명령
    config_parser = subparsers.add_parser('config', help='설정 관리')
    config_subparsers = config_parser.add_subparsers(dest='config_command')

    config_view = config_subparsers.add_parser('view', help='설정 보기')
    config_view.add_argument('--section', help='특정 섹션만 보기')

    config_validate = config_subparsers.add_parser('validate', help='설정 검증')

    # 도움말 표시
    help_parser = subparsers.add_parser('help', help='상세 도움말 표시')

    args = parser.parse_args()

    # 명령이 없으면 도움말 표시
    if args.command is None:
        parser.print_help()
        return

    try:
        # 시스템 초기화
        system = LicensePlateSystem(args.config)

        # 상세 로그 설정
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # 명령 실행
        if args.command == 'server':
            # 웹 서버 실행
            system.run_server(args.host, args.port, args.debug)

        elif args.command == 'image':
            # 단일 이미지 처리
            # OCR 엔진 환경 변수 설정
            if hasattr(args, 'ocr_engine'):
                os.environ['OCR_ENGINE'] = args.ocr_engine
                print(f"🔍 OCR 엔진 설정: {args.ocr_engine}")

            # 시스템 재초기화 (OCR 엔진 변경 반영)
            system = LicensePlateSystem(args.config)

            system.process_image(args.image_path, show_result=not args.no_display)

        elif args.command == 'config':
            # 설정 관리
            if args.config_command == 'view':
                if args.section:
                    section_config = getattr(system.config, args.section, None)
                    if section_config:
                        import yaml
                        from dataclasses import asdict
                        print(f"[{args.section}]")
                        print(yaml.dump(asdict(section_config), default_flow_style=False, indent=2))
                    else:
                        print(f"❌ 섹션을 찾을 수 없습니다: {args.section}")
                else:
                    import yaml
                    from dataclasses import asdict
                    try:
                        config_dict = asdict(system.config)
                        print(yaml.dump(config_dict, default_flow_style=False, indent=2))
                    except Exception as e:
                        print(f"설정 표시 중 오류: {e}")

            elif args.config_command == 'validate':
                try:
                    validation = system.config_manager.validate_config()
                    if validation['valid']:
                        print("✅ 설정이 유효합니다!")
                    else:
                        print("❌ 설정에 오류가 있습니다:")
                        for error in validation['errors']:
                            print(f"   - {error}")

                    if validation.get('warnings'):
                        print("⚠️ 경고사항:")
                        for warning in validation['warnings']:
                            print(f"   - {warning}")
                except Exception as e:
                    print(f"설정 검증 중 오류: {e}")

            else:
                config_parser.print_help()

        elif args.command == 'help':
            # 상세 도움말 표시
            show_detailed_help()

    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def show_detailed_help():
    """상세 도움말 표시"""
    help_text = """
🚗 YOLOv8 번호판 인식 시스템 v2.0

📋 주요 기능:
  ✨ YOLOv8 기반 고정확도 번호판 탐지
  🎯 한국어 번호판 특화 OCR
  🌐 웹 인터페이스 제공
  📊 상세 통계 및 분석

🚀 빠른 시작:
  1. 의존성 설치: pip install -r requirements.txt
  2. Tesseract 설치 (OS별 설치 가이드 참고)
  3. 웹 서버 실행: python main_app.py server
  4. 브라우저에서 http://localhost:5000 접속

📖 상세 사용법:

  🌐 웹 서버 모드:
    python main_app.py server
    python main_app.py server --host 0.0.0.0 --port 8000

  🖼️ 이미지 처리 모드:
    python main_app.py image car.jpg                       # Auto OCR
    python main_app.py image car.jpg --ocr-engine pororo    # Pororo OCR 사용 (한국어 특화)
    python main_app.py image car.jpg --ocr-engine paddleocr # PaddleOCR 사용
    python main_app.py image car.jpg --ocr-engine easyocr   # EasyOCR 사용
    python main_app.py image car.jpg --ocr-engine tesseract # Tesseract 사용
    python main_app.py image car.jpg --no-display          # 결과 표시 안함

  ⚙️ 설정 관리:
    python main_app.py config view              # 전체 설정 보기
    python main_app.py config view --section model
    python main_app.py config validate          # 설정 검증

🔧 OCR 엔진 옵션:
  --ocr-engine auto       - 자동으로 최적 엔진 선택 (기본값)
  --ocr-engine pororo     - Pororo OCR 사용 (한국어 특화, 최고 정확도)
  --ocr-engine paddleocr  - PaddleOCR 사용 (높은 정확도)
  --ocr-engine easyocr    - EasyOCR 사용 (균형잡힌 성능) 
  --ocr-engine tesseract  - Tesseract 사용 (빠른 처리)

🔧 환경 변수:
  OCR_ENGINE             - 기본 OCR 엔진 설정
  LP_WEB_HOST            - 웹 서버 호스트
  LP_WEB_PORT            - 웹 서버 포트
  LP_LOG_LEVEL           - 로그 레벨 (DEBUG, INFO, WARNING, ERROR)

💡 팁:
  - GPU가 있다면 CUDA를 설치하여 성능을 대폭 향상시킬 수 있습니다
  - 실시간 처리는 웹 인터페이스에서 이용 가능합니다
    """
    print(help_text)


def check_dependencies():
    """의존성 확인"""
    required_modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('ultralytics', 'ultralytics'),
        ('pytesseract', 'pytesseract'),
        ('flask', 'flask'),
        ('yaml', 'pyyaml')
    ]

    missing_modules = []

    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(package_name)

    if missing_modules:
        print("❌ 필수 모듈이 설치되지 않았습니다:")
        for package in missing_modules:
            print(f"   - {package}")
        print("\n설치 명령: pip install -r requirements.txt")
        return False

    return True


def show_system_info():
    """시스템 정보 표시"""
    try:
        print("🖥️ 시스템 정보:")
        print(f"   - Python: {sys.version.split()[0]}")

        try:
            import cv2
            print(f"   - OpenCV: {cv2.__version__}")
        except ImportError:
            print("   - OpenCV: ❌ 설치되지 않음")

        try:
            import torch
            print(f"   - PyTorch: {torch.__version__}")
            print(f"   - CUDA 사용 가능: {'✅' if torch.cuda.is_available() else '❌'}")

            if torch.cuda.is_available():
                print(f"   - GPU 개수: {torch.cuda.device_count()}")
        except ImportError:
            print("   - PyTorch: ❌ 설치되지 않음")

        try:
            from ultralytics import __version__ as ultralytics_version
            print(f"   - Ultralytics: {ultralytics_version}")
        except ImportError:
            print("   - Ultralytics: ❌ 설치되지 않음")

        # Tesseract 확인
        try:
            import pytesseract
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"   - Tesseract: {tesseract_version}")

            languages = pytesseract.get_languages()
            has_korean = 'kor' in languages
            print(f"   - 한국어 지원: {'✅' if has_korean else '❌'}")

        except Exception:
            print("   - Tesseract: ❌ 설치되지 않음")

    except Exception as e:
        print(f"시스템 정보 확인 실패: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚗 YOLOv8 번호판 인식 시스템 v2.0")
    print("=" * 60)

    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)

    # 시스템 정보 표시
    show_system_info()
    print("=" * 60)

    # 메인 실행
    main()