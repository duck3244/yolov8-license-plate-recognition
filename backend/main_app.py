#!/usr/bin/env python3
"""
main_app.py
YOLOv8 번호판 인식 CLI 도구.

웹 서버는 FastAPI(`backend/main.py`)로 분리되었다. 이 파일은 단발성
이미지 처리·설정 검증·시스템 정보 조회용 CLI.

Usage:
    python main_app.py image car.jpg
    python main_app.py config view
    python main_app.py config validate
    # 웹 서버는: uvicorn main:app --port 8000
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from config_manager import ConfigManager, SystemConfig
from database_manager import DatabaseManager
from license_plate_recognizer import YOLOv8LicensePlateRecognizer


class LicensePlateSystem:
    """CLI용 시스템 컨테이너 — recognizer + DB + 설정 + 로깅."""

    def __init__(self, config_path: str = "config.yaml"):
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        except Exception as e:
            print(f"⚠️ 설정 파일 로드 실패, 기본 설정 사용: {e}", file=sys.stderr)
            self.config_manager = None
            self.config = SystemConfig()

        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 CLI 초기화 시작")

        ocr_engine = os.environ.get("OCR_ENGINE", "auto")
        self.recognizer = YOLOv8LicensePlateRecognizer(
            yolo_model_path=self.config.model.yolo_model_path,
            confidence_threshold=self.config.model.confidence_threshold,
            ocr_engine=ocr_engine,
        )
        self.db_manager = DatabaseManager(self.config.database.path)

    def _setup_logging(self) -> None:
        log_cfg = self.config.logging
        Path(log_cfg.file).parent.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))

        formatter = logging.Formatter(log_cfg.format)
        file_handler = logging.handlers.RotatingFileHandler(
            log_cfg.file,
            maxBytes=log_cfg.max_size_mb * 1024 * 1024,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    def process_image(self, image_path: str, show_result: bool = True) -> None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

        self.logger.info(f"🖼️ 이미지 처리 시작: {image_path}")
        start = time.time()
        plate_text, result_img = self.recognizer.process_image(image_path)
        elapsed = time.time() - start

        print(f"🚗 인식된 번호판: {plate_text}")
        print(f"⏱️ 처리 시간: {elapsed:.3f}초")

        if show_result and result_img is not None:
            try:
                import cv2
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                original_img = cv2.imread(image_path)
                ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                ax1.set_title("원본 이미지")
                ax1.axis("off")
                ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                ax2.set_title(f"결과: {plate_text}")
                ax2.axis("off")
                plt.tight_layout()
                plt.show()
            except ImportError:
                self.logger.warning("matplotlib를 사용할 수 없습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8 번호판 인식 CLI (웹 서버는 'uvicorn main:app' 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--verbose", "-v", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    image_parser = subparsers.add_parser("image", help="단일 이미지 처리")
    image_parser.add_argument("image_path")
    image_parser.add_argument("--no-display", action="store_true")
    image_parser.add_argument(
        "--ocr-engine",
        choices=["auto", "pororo", "paddleocr", "easyocr", "tesseract"],
        default="auto",
    )

    config_parser = subparsers.add_parser("config", help="설정 관리")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_view = config_sub.add_parser("view")
    config_view.add_argument("--section")
    config_sub.add_parser("validate")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "image":
        os.environ["OCR_ENGINE"] = args.ocr_engine

    system = LicensePlateSystem(args.config)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "image":
        system.process_image(args.image_path, show_result=not args.no_display)
        return

    if args.command == "config":
        import yaml
        if args.config_command == "view":
            data = asdict(system.config)
            if args.section:
                if args.section in data:
                    print(f"[{args.section}]")
                    print(yaml.dump(data[args.section], default_flow_style=False, indent=2))
                else:
                    print(f"❌ 섹션을 찾을 수 없습니다: {args.section}")
            else:
                print(yaml.dump(data, default_flow_style=False, indent=2))
        elif args.config_command == "validate" and system.config_manager:
            validation = system.config_manager.validate_config()
            print("✅ 설정이 유효합니다!" if validation["valid"] else "❌ 설정 오류:")
            for e in validation["errors"]:
                print(f"   - {e}")
            for w in validation.get("warnings", []):
                print(f"   ⚠️ {w}")
        else:
            config_parser.print_help()


if __name__ == "__main__":
    main()
