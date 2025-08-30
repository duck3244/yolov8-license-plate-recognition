"""
license_plate_recognizer.py
YOLOv8 기반 차량 번호판 인식 엔진

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import cv2
import numpy as np
import os
import re
import logging
from typing import List, Tuple, Optional

# 의존성 체크 및 import
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    print("⚠️ ultralytics가 설치되지 않았습니다. YOLOv8 기능이 제한됩니다.")
    HAS_YOLO = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    print("⚠️ pytesseract가 설치되지 않았습니다. OCR 기능이 제한됩니다.")
    HAS_TESSERACT = False

logger = logging.getLogger(__name__)

class YOLOv8LicensePlateRecognizer:
    """YOLOv8 기반 번호판 인식기"""

    def __init__(self,
                 yolo_model_path: str = 'yolov8n.pt',
                 tesseract_cmd: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 ocr_engine: str = 'auto'):
        """
        YOLOv8 기반 번호판 인식기 초기화

        Args:
            yolo_model_path: YOLOv8 모델 경로
            tesseract_cmd: Tesseract 실행 파일 경로
            confidence_threshold: YOLO 탐지 임계값
            ocr_engine: OCR 엔진 선택 ('auto', 'pororo', 'paddleocr', 'easyocr', 'tesseract')
        """

        # YOLOv8 모델 로드 (가능한 경우)
        self.yolo_model = None
        if HAS_YOLO:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                logger.info(f"YOLOv8 모델 로드 완료: {yolo_model_path}")
            except Exception as e:
                logger.warning(f"YOLOv8 모델 로드 실패: {e}. OpenCV 방식을 사용합니다.")

        # OCR 엔진 설정
        self.ocr_engine = self._setup_ocr_engine(ocr_engine)

        # Tesseract 설정 (가능한 경우)
        if HAS_TESSERACT and tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.confidence_threshold = confidence_threshold

        # 한국 번호판 패턴
        self.korean_plate_patterns = [
            r'\d{2}[가-힣]\d{4}',  # 12가1234
            r'\d{3}[가-힣]\d{4}',  # 123가1234
            r'[가-힣]\d{2}[가-힣]\d{4}',  # 서12가1234
        ]

        logger.info(f"OCR 엔진: {self.ocr_engine}")

    def _setup_ocr_engine(self, preferred_engine: str):
        """OCR 엔진 설정"""

        # 특정 엔진 지정된 경우
        if preferred_engine == 'pororo' and HAS_PORORO:
            try:
                self.pororo_ocr = Pororo(task="ocr", lang="ko", model="brainocr")
                return 'pororo'
            except Exception as e:
                logger.warning(f"Pororo OCR 초기화 실패: {e}")

        elif preferred_engine == 'paddleocr' and HAS_PADDLEOCR:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)
                return 'paddleocr'
            except Exception as e:
                logger.warning(f"PaddleOCR 초기화 실패: {e}")

        elif preferred_engine == 'easyocr' and HAS_EASYOCR:
            try:
                self.easy_reader = easyocr.Reader(['ko', 'en'], gpu=True, verbose=False)
                return 'easyocr'
            except Exception as e:
                logger.warning(f"EasyOCR 초기화 실패: {e}")

        elif preferred_engine == 'tesseract' and HAS_TESSERACT:
            return 'tesseract'

        # Auto 모드: 우선순위에 따라 최적 엔진 선택
        elif preferred_engine == 'auto':
            # 1순위: Pororo (한국어 특화)
            if HAS_PORORO:
                try:
                    self.pororo_ocr = Pororo(task="ocr", lang="ko", model="brainocr")
                    return 'pororo'
                except Exception as e:
                    logger.warning(f"Pororo OCR 초기화 실패: {e}")

            # 2순위: PaddleOCR (높은 정확도)
            if HAS_PADDLEOCR:
                try:
                    self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)
                    return 'paddleocr'
                except Exception as e:
                    logger.warning(f"PaddleOCR 초기화 실패: {e}")

            # 3순위: EasyOCR (균형잡힌 성능)
            if HAS_EASYOCR:
                try:
                    self.easy_reader = easyocr.Reader(['ko', 'en'], gpu=True, verbose=False)
                    return 'easyocr'
                except Exception as e:
                    logger.warning(f"EasyOCR 초기화 실패: {e}")

            # 4순위: Tesseract (기본)
            if HAS_TESSERACT:
                return 'tesseract'

        logger.warning("사용 가능한 OCR 엔진이 없습니다.")
        return 'none'

    def detect_license_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        번호판 탐지 (YOLOv8 또는 OpenCV 사용)

        Args:
            image: 입력 이미지 (BGR format)

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        if self.yolo_model is not None:
            return self._yolo_detect(image)
        else:
            return self._opencv_detect(image)

    def _yolo_detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """YOLOv8을 사용한 번호판 탐지"""
        try:
            results = self.yolo_model(image)
            plates = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf)
                        if confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            plates.append((x1, y1, x2, y2, confidence))

            return plates
        except Exception as e:
            logger.error(f"YOLO 탐지 오류: {e}")
            return []

    def _opencv_detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """OpenCV를 사용한 번호판 탐지 (fallback)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 가우시안 블러
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 엣지 검출
            edges = cv2.Canny(blurred, 10, 200)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            plates = []
            for contour in contours:
                # 윤곽선의 경계 사각형
                x, y, w, h = cv2.boundingRect(contour)

                # 번호판 크기 필터링 (가로세로 비율 및 최소 크기)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)

                if (2.0 < aspect_ratio < 5.0 and
                    area > 2000 and
                    w > 80 and h > 20):

                    plates.append((x, y, x + w, y + h, 0.7))  # 기본 신뢰도 0.7

            # 면적 순으로 정렬하여 상위 5개만 반환
            plates.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            return plates[:5]

        except Exception as e:
            logger.error(f"OpenCV 탐지 오류: {e}")
            return []

    def preprocess_plate_region(self, plate_img: np.ndarray) -> np.ndarray:
        """번호판 영역 전처리"""
        try:
            # 그레이스케일 변환
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img

            # 크기 조정 (OCR 성능 향상)
            height, width = gray.shape
            if height < 50:
                scale_factor = 50 / height
                new_width = int(width * scale_factor)
                gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)

            # 노이즈 제거
            denoised = cv2.medianBlur(gray, 3)

            # 이진화
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            logger.error(f"전처리 오류: {e}")
            return plate_img

    def recognize_text(self, plate_img: np.ndarray) -> str:
        """통합 텍스트 인식 - 선택된 OCR 엔진 사용"""

        if self.ocr_engine == 'none':
            return "OCR_NOT_AVAILABLE"

        # 전처리
        processed_img = self.preprocess_plate_region(plate_img)

        try:
            if self.ocr_engine == 'pororo' and hasattr(self, 'pororo_ocr'):
                return self._recognize_with_pororo(processed_img)

            elif self.ocr_engine == 'paddleocr' and hasattr(self, 'paddle_ocr'):
                return self._recognize_with_paddleocr(processed_img)

            elif self.ocr_engine == 'easyocr' and hasattr(self, 'easy_reader'):
                return self._recognize_with_easyocr(processed_img)

            elif self.ocr_engine == 'tesseract' and HAS_TESSERACT:
                return self._recognize_with_tesseract(processed_img)

            else:
                # Fallback to tesseract
                if HAS_TESSERACT:
                    return self._recognize_with_tesseract(processed_img)
                else:
                    return "NO_OCR_AVAILABLE"

        except Exception as e:
            logger.error(f"OCR 처리 중 오류: {e}")
            return ""

    def _recognize_with_pororo(self, plate_img: np.ndarray) -> str:
        """Pororo OCR을 사용한 텍스트 인식"""
        try:
            # Pororo는 PIL Image 또는 이미지 경로를 받음
            from PIL import Image
            import tempfile
            import os

            # numpy 이미지를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name

            # 이미지 저장
            cv2.imwrite(temp_path, plate_img)

            try:
                # Pororo OCR 수행
                result = self.pororo_ocr(temp_path)

                # 결과 처리
                if isinstance(result, dict) and 'description' in result:
                    text = result['description']
                elif isinstance(result, str):
                    text = result
                elif isinstance(result, list) and len(result) > 0:
                    # 리스트 형태의 결과인 경우 첫 번째 항목 사용
                    if isinstance(result[0], dict) and 'description' in result[0]:
                        text = result[0]['description']
                    else:
                        text = str(result[0])
                else:
                    text = ""

                return self.clean_plate_text(text)

            finally:
                # 임시 파일 삭제
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Pororo OCR 처리 오류: {e}")
            return ""

    def _recognize_with_easyocr(self, plate_img: np.ndarray) -> str:
        """EasyOCR을 사용한 텍스트 인식"""
        try:
            results = self.easy_reader.readtext(plate_img)

            if results:
                best_result = ""
                best_confidence = 0

                for (bbox, text, confidence) in results:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = text

                return self.clean_plate_text(best_result)

            return ""

        except Exception as e:
            logger.error(f"EasyOCR 처리 오류: {e}")
            return ""

    def _recognize_with_paddleocr(self, plate_img: np.ndarray) -> str:
        """PaddleOCR을 사용한 텍스트 인식"""
        try:
            results = self.paddle_ocr.ocr(plate_img, cls=True)

            if results and results[0]:
                best_result = ""
                best_confidence = 0

                for line in results[0]:
                    if len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = text

                return self.clean_plate_text(best_result)

            return ""

        except Exception as e:
            logger.error(f"PaddleOCR 처리 오류: {e}")
            return ""

    def _recognize_with_tesseract(self, plate_img: np.ndarray) -> str:
        """Tesseract를 사용한 텍스트 인식"""
    def _recognize_with_tesseract(self, plate_img: np.ndarray) -> str:
        """Tesseract를 사용한 텍스트 인식"""
        if not HAS_TESSERACT:
            return "TESSERACT_NOT_AVAILABLE"

        try:
            # OCR 설정
            config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후'

            # 텍스트 인식
            text = pytesseract.image_to_string(plate_img, lang='kor+eng', config=config).strip()

            # 텍스트 정리
            cleaned_text = self.clean_plate_text(text)

            return cleaned_text

        except Exception as e:
            logger.error(f"Tesseract OCR 처리 중 오류: {e}")
            return ""

    def clean_plate_text(self, text: str) -> str:
        """번호판 텍스트 정리"""
        if not text:
            return ""

        # 불필요한 문자 제거
        cleaned = re.sub(r'[^\w가-힣]', '', text)

        # 일반적인 OCR 오류 수정
        corrections = {
            'O': '0', 'I': '1', 'l': '1', 'S': '5', 'Z': '2',
            'B': '8', 'G': '6', 'D': '0'
        }

        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)

        return cleaned

    def is_valid_korean_plate(self, text: str) -> bool:
        """한국 번호판 형식 검증"""
        if not text:
            return False

        cleaned_text = re.sub(r'[^\w가-힣]', '', text)

        for pattern in self.korean_plate_patterns:
            if re.match(pattern, cleaned_text):
                return True
        return False

    def process_image(self, image_path: str, save_result: bool = True) -> Tuple[str, np.ndarray]:
        """
        이미지에서 번호판 인식 수행

        Args:
            image_path: 입력 이미지 경로
            save_result: 결과 이미지 저장 여부

        Returns:
            (인식된 번호판 텍스트, 결과 이미지)
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

            # 번호판 탐지
            plates = self.detect_license_plates(image)

            if not plates:
                logger.warning("번호판을 탐지하지 못했습니다.")
                return "", image

            # 가장 신뢰도가 높은 번호판 선택
            best_plate = max(plates, key=lambda x: x[4])
            x1, y1, x2, y2, confidence = best_plate

            # 번호판 영역 추출
            plate_region = image[y1:y2, x1:x2]

            # 텍스트 인식
            plate_text = self.recognize_text(plate_region)

            # 결과 시각화
            result_image = image.copy()
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 텍스트 표시 (한글 지원을 위해 간단하게)
            label = f"{plate_text} ({confidence:.2f})" if plate_text else f"Detected ({confidence:.2f})"
            cv2.putText(result_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 결과 저장
            if save_result and plate_text:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                result_path = f"result_{base_name}.jpg"
                cv2.imwrite(result_path, result_image)
                logger.info(f"결과 이미지 저장: {result_path}")

            return plate_text, result_image

        except Exception as e:
            logger.error(f"이미지 처리 실패: {e}")
            return "", cv2.imread(image_path) if os.path.exists(image_path) else np.zeros((100, 100, 3))

def main():
    """테스트 실행"""
    print("🚗 YOLOv8 번호판 인식기 테스트")

    # 사용 가능한 OCR 엔진 확인
    print("📊 사용 가능한 OCR 엔진:")
    print(f"   - Pororo: {'✅' if HAS_PORORO else '❌'}")
    print(f"   - PaddleOCR: {'✅' if HAS_PADDLEOCR else '❌'}")
    print(f"   - EasyOCR: {'✅' if HAS_EASYOCR else '❌'}")
    print(f"   - Tesseract: {'✅' if HAS_TESSERACT else '❌'}")
    print(f"   - YOLOv8: {'✅' if HAS_YOLO else '❌'}")

    # 각 OCR 엔진 테스트
    engines_to_test = []
    if HAS_PORORO:
        engines_to_test.append('pororo')
    if HAS_PADDLEOCR:
        engines_to_test.append('paddleocr')
    if HAS_EASYOCR:
        engines_to_test.append('easyocr')
    if HAS_TESSERACT:
        engines_to_test.append('tesseract')

    test_image = "test_car.jpg"
    if os.path.exists(test_image):
        print(f"\n📷 테스트 이미지: {test_image}")

        for engine in engines_to_test:
            print(f"\n🔍 {engine.upper()} 엔진 테스트:")
            try:
                recognizer = YOLOv8LicensePlateRecognizer(ocr_engine=engine)
                plate_text, _ = recognizer.process_image(test_image, save_result=False)
                print(f"   결과: {plate_text}")
            except Exception as e:
                print(f"   오류: {e}")
    else:
        print(f"\n⚠️ 테스트 이미지가 없습니다: {test_image}")
        print("테스트 이미지를 추가하고 다시 실행해보세요.")

    # 자동 모드 테스트
    print(f"\n🤖 Auto 모드 테스트:")
    try:
        auto_recognizer = YOLOv8LicensePlateRecognizer(ocr_engine='auto')
        print(f"   선택된 엔진: {auto_recognizer.ocr_engine}")
    except Exception as e:
        print(f"   오류: {e}")

if __name__ == "__main__":
    main()