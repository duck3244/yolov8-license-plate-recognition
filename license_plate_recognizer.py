"""
license_plate_recognizer.py
YOLOv8 기반 차량 번호판 인식 엔진 (고급 OpenCV 전처리 통합)

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.1.0
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
    print("⚠️ pytesseract가 설치되지 않았습니다. Tesseract OCR 기능이 제한됩니다.")
    HAS_TESSERACT = False

# Pororo OCR 체크 (pororo-ocr 라이브러리 사용)
try:
    import prrocr
    HAS_PORORO = True
    HAS_ORIGINAL_PORORO = False
except ImportError:
    try:
        # 원본 Pororo도 체크 (fallback)
        from pororo import Pororo
        HAS_PORORO = True
        HAS_ORIGINAL_PORORO = True
    except ImportError:
        print("⚠️ Pororo가 설치되지 않았습니다.")
        print("   설치 방법 1: pip install pororo-ocr  (가벼운 OCR 전용)")
        print("   설치 방법 2: pip install pororo      (전체 기능)")
        HAS_PORORO = False
        HAS_ORIGINAL_PORORO = False

# PaddleOCR 체크
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    print("⚠️ PaddleOCR이 설치되지 않았습니다. 설치: pip install paddleocr")
    HAS_PADDLEOCR = False

# EasyOCR 체크
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    print("⚠️ EasyOCR이 설치되지 않았습니다. 설치: pip install easyocr")
    HAS_EASYOCR = False

logger = logging.getLogger(__name__)

class YOLOv8LicensePlateRecognizer:
    """YOLOv8 기반 번호판 인식기 (고급 OpenCV 전처리 포함)"""

    def __init__(self,
                 yolo_model_path: str = 'yolov8n.pt',
                 tesseract_cmd: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 ocr_engine: str = 'auto',
                 use_advanced_preprocessing: bool = True):
        """
        YOLOv8 기반 번호판 인식기 초기화

        Args:
            yolo_model_path: YOLOv8 모델 경로
            tesseract_cmd: Tesseract 실행 파일 경로
            confidence_threshold: YOLO 탐지 임계값
            ocr_engine: OCR 엔진 선택 ('auto', 'pororo', 'paddleocr', 'easyocr', 'tesseract')
            use_advanced_preprocessing: 고급 OpenCV 전처리 사용 여부
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
        self.use_advanced_preprocessing = use_advanced_preprocessing

        # 한국 번호판 패턴
        self.korean_plate_patterns = [
            r'\d{2,3}[가-힣]\d{4}',  # 12가1234, 123가1234
            r'[가-힣]\d{2,3}[가-힣]\d{4}',  # 서12가1234
            r'\d{2,3}[가-힣]\d{3,4}',  # 유연한 패턴
        ]

        # 고급 전처리 파라미터
        self.char_detection_params = {
            'MIN_AREA': 80,
            'MIN_WIDTH': 2,
            'MIN_HEIGHT': 8,
            'MIN_RATIO': 0.25,
            'MAX_RATIO': 1.0,
            'MAX_DIAG_MULTIPLYER': 5,
            'MAX_ANGLE_DIFF': 12.0,
            'MAX_AREA_DIFF': 0.5,
            'MAX_WIDTH_DIFF': 0.8,
            'MAX_HEIGHT_DIFF': 0.2,
            'MIN_N_MATCHED': 3,
            'PLATE_WIDTH_PADDING': 1.3,
            'PLATE_HEIGHT_PADDING': 1.5,
            'MIN_PLATE_RATIO': 3,
            'MAX_PLATE_RATIO': 10
        }

        logger.info(f"OCR 엔진: {self.ocr_engine}")
        logger.info(f"고급 전처리: {'✅' if self.use_advanced_preprocessing else '❌'}")

    def _setup_ocr_engine(self, preferred_engine: str):
        """OCR 엔진 설정"""

        # 특정 엔진 지정된 경우
        if preferred_engine == 'pororo' and HAS_PORORO:
            try:
                if not HAS_ORIGINAL_PORORO:
                    # pororo-ocr 라이브러리 사용
                    self.pororo_ocr = prrocr.ocr(lang="ko")
                    logger.info("Pororo OCR (prrocr) 초기화 완료")
                    return 'pororo'
                else:
                    # 원본 Pororo 라이브러리 사용
                    self.pororo_ocr = Pororo(task="ocr", lang="ko", model="brainocr")
                    logger.info("Pororo OCR (원본) 초기화 완료")
                    return 'pororo'
            except Exception as e:
                logger.warning(f"Pororo OCR 초기화 실패: {e}")

        elif preferred_engine == 'paddleocr' and HAS_PADDLEOCR:
            try:
                # PaddleOCR 버전별 호환성 처리
                try:
                    # 최신 버전 시도 (show_log 파라미터 없음)
                    self.paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='korean'
                    )
                    logger.info("PaddleOCR 초기화 완료 (최신 버전)")
                    return 'paddleocr'
                except Exception as e1:
                    logger.debug(f"최신 PaddleOCR 초기화 실패: {e1}")
                    try:
                        # 구버전 시도 (show_log 파라미터 포함)
                        self.paddle_ocr = PaddleOCR(
                            use_angle_cls=True,
                            lang='korean',
                            show_log=False
                        )
                        logger.info("PaddleOCR 초기화 완료 (구버전)")
                        return 'paddleocr'
                    except Exception as e2:
                        logger.debug(f"구버전 PaddleOCR 초기화 실패: {e2}")
                        # 최소 설정으로 시도
                        self.paddle_ocr = PaddleOCR(lang='korean')
                        logger.info("PaddleOCR 초기화 완료 (최소 설정)")
                        return 'paddleocr'
            except Exception as e:
                logger.warning(f"PaddleOCR 초기화 실패: {e}")

        elif preferred_engine == 'easyocr' and HAS_EASYOCR:
            try:
                self.easy_reader = easyocr.Reader(['ko', 'en'], gpu=True, verbose=False)
                logger.info("EasyOCR 초기화 완료")
                return 'easyocr'
            except Exception as e:
                logger.warning(f"EasyOCR 초기화 실패: {e}")

        elif preferred_engine == 'tesseract' and HAS_TESSERACT:
            logger.info("Tesseract OCR 사용")
            return 'tesseract'

        # Auto 모드: 우선순위에 따라 최적 엔진 선택
        elif preferred_engine == 'auto':
            # 1순위: Pororo (한국어 특화)
            if HAS_PORORO:
                try:
                    if not HAS_ORIGINAL_PORORO:
                        self.pororo_ocr = prrocr.ocr(lang="ko")
                        logger.info("Auto 모드: Pororo OCR (prrocr) 선택됨")
                        return 'pororo'
                    else:
                        self.pororo_ocr = Pororo(task="ocr", lang="ko", model="brainocr")
                        logger.info("Auto 모드: Pororo OCR (원본) 선택됨")
                        return 'pororo'
                except Exception as e:
                    logger.warning(f"Pororo OCR 초기화 실패: {e}")

            # 2순위: PaddleOCR (높은 정확도)
            if HAS_PADDLEOCR:
                try:
                    # PaddleOCR 버전별 호환성 처리
                    try:
                        # 최신 버전 시도
                        self.paddle_ocr = PaddleOCR(
                            use_angle_cls=True,
                            lang='korean'
                        )
                        logger.info("Auto 모드: PaddleOCR 선택됨 (최신 버전)")
                        return 'paddleocr'
                    except Exception as e1:
                        logger.debug(f"최신 PaddleOCR 초기화 실패: {e1}")
                        try:
                            # 구버전 시도
                            self.paddle_ocr = PaddleOCR(
                                use_angle_cls=True,
                                lang='korean',
                                show_log=False
                            )
                            logger.info("Auto 모드: PaddleOCR 선택됨 (구버전)")
                            return 'paddleocr'
                        except Exception as e2:
                            logger.debug(f"구버전 PaddleOCR 초기화 실패: {e2}")
                            # 최소 설정으로 시도
                            self.paddle_ocr = PaddleOCR(lang='korean')
                            logger.info("Auto 모드: PaddleOCR 선택됨 (최소 설정)")
                            return 'paddleocr'
                except Exception as e:
                    logger.warning(f"PaddleOCR 초기화 실패: {e}")

            # 3순위: EasyOCR (균형잡힌 성능)
            if HAS_EASYOCR:
                try:
                    self.easy_reader = easyocr.Reader(['ko', 'en'], gpu=True, verbose=False)
                    logger.info("Auto 모드: EasyOCR 선택됨")
                    return 'easyocr'
                except Exception as e:
                    logger.warning(f"EasyOCR 초기화 실패: {e}")

            # 4순위: Tesseract (기본)
            if HAS_TESSERACT:
                logger.info("Auto 모드: Tesseract 선택됨")
                return 'tesseract'

        logger.warning("사용 가능한 OCR 엔진이 없습니다.")
        return 'none'

    def detect_license_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        번호판 탐지 (YOLOv8 또는 고급 OpenCV 사용)

        Args:
            image: 입력 이미지 (BGR format)

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        # YOLOv8 사용 가능하면 우선 시도
        if self.yolo_model is not None:
            yolo_plates = self._yolo_detect(image)
            if yolo_plates:
                logger.debug(f"YOLO로 {len(yolo_plates)}개 번호판 탐지됨")
                return yolo_plates

        # YOLO 실패 시 또는 사용 불가능한 경우 고급 OpenCV 방식 사용
        if self.use_advanced_preprocessing:
            opencv_plates = self._advanced_opencv_detect(image)
            if opencv_plates:
                logger.debug(f"고급 OpenCV로 {len(opencv_plates)}개 번호판 탐지됨")
                return opencv_plates

        # 기본 OpenCV 방식
        basic_plates = self._opencv_detect(image)
        logger.debug(f"기본 OpenCV로 {len(basic_plates)}개 번호판 탐지됨")
        return basic_plates

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

    def _advanced_opencv_detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        고급 OpenCV 기반 번호판 탐지 (첨부된 코드 기반)

        Args:
            image: 입력 이미지 (BGR format)

        Returns:
            탐지된 번호판 영역 리스트
        """
        try:
            height, width, channel = image.shape

            # 1. 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 2. 형태학적 연산으로 이미지 향상
            structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            img_top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuring_element)
            img_black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuring_element)

            img_enhanced = cv2.add(gray, img_top_hat)
            gray = cv2.subtract(img_enhanced, img_black_hat)

            # 3. 가우시안 블러
            img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

            # 4. 적응적 임계값
            img_thresh = cv2.adaptiveThreshold(
                img_blurred,
                maxValue=255.0,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=19,
                C=9
            )

            # 5. 윤곽선 찾기
            contours, _ = cv2.findContours(
                img_thresh,
                mode=cv2.RETR_LIST,
                method=cv2.CHAIN_APPROX_SIMPLE
            )

            # 6. 문자 후보 영역 필터링
            possible_contours = self._filter_char_contours(contours)

            if not possible_contours:
                return []

            # 7. 문자 그룹핑으로 번호판 영역 찾기
            plate_candidates = self._find_plate_regions(possible_contours, img_thresh, width, height)

            # 8. 번호판 검증 및 반환
            valid_plates = []
            for plate_info in plate_candidates:
                x, y, w, h = plate_info['x'], plate_info['y'], plate_info['w'], plate_info['h']
                confidence = plate_info.get('confidence', 0.8)
                valid_plates.append((x, y, x + w, y + h, confidence))

            return valid_plates

        except Exception as e:
            logger.error(f"고급 OpenCV 탐지 오류: {e}")
            return []

    def _filter_char_contours(self, contours) -> List[dict]:
        """문자 후보 윤곽선 필터링"""
        params = self.char_detection_params
        possible_contours = []

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h

            # 문자 크기 및 비율 필터링
            if (area > params['MIN_AREA'] and
                w > params['MIN_WIDTH'] and h > params['MIN_HEIGHT'] and
                params['MIN_RATIO'] < ratio < params['MAX_RATIO']):

                possible_contours.append({
                    'contour': contour,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': x + (w / 2), 'cy': y + (h / 2),
                    'idx': idx
                })

        return possible_contours

    def _find_char_groups(self, contour_list) -> List[List[int]]:
        """문자들을 그룹핑하여 번호판 후보 찾기"""
        params = self.char_detection_params
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []

            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                # 거리 및 각도 계산
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))

                # 면적, 너비, 높이 차이 계산
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                # 유사한 문자인지 판단
                if (distance < diagonal_length1 * params['MAX_DIAG_MULTIPLYER'] and
                    angle_diff < params['MAX_ANGLE_DIFF'] and
                    area_diff < params['MAX_AREA_DIFF'] and
                    width_diff < params['MAX_WIDTH_DIFF'] and
                    height_diff < params['MAX_HEIGHT_DIFF']):

                    matched_contours_idx.append(d2['idx'])

            # 현재 윤곽선도 포함
            matched_contours_idx.append(d1['idx'])

            # 최소 문자 수 확인
            if len(matched_contours_idx) >= params['MIN_N_MATCHED']:
                matched_result_idx.append(matched_contours_idx)
                break

        return matched_result_idx

    def _find_plate_regions(self, possible_contours, img_thresh, width, height) -> List[dict]:
        """번호판 영역 추출"""
        params = self.char_detection_params

        # 문자 그룹 찾기
        char_groups = self._find_char_groups(possible_contours)

        plate_candidates = []

        for group_indices in char_groups:
            try:
                # 그룹에서 실제 문자 객체들 가져오기
                matched_chars = [possible_contours[i] for i in range(len(possible_contours))
                               if possible_contours[i]['idx'] in group_indices]

                if len(matched_chars) < params['MIN_N_MATCHED']:
                    continue

                # x 좌표로 정렬
                sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

                # 번호판 영역 계산
                plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
                plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

                plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * params['PLATE_WIDTH_PADDING']

                # 평균 높이 계산
                sum_height = sum(d['h'] for d in sorted_chars)
                plate_height = int(sum_height / len(sorted_chars) * params['PLATE_HEIGHT_PADDING'])

                # 기울기 계산 및 회전
                triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
                triangle_hypotenus = np.linalg.norm(
                    np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                    np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
                )

                if triangle_hypotenus > 0:
                    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
                else:
                    angle = 0

                # 회전 행렬 적용
                rotation_matrix = cv2.getRotationMatrix2D(
                    center=(plate_cx, plate_cy),
                    angle=angle,
                    scale=1.0
                )

                img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

                # 번호판 영역 추출
                img_cropped = cv2.getRectSubPix(
                    img_rotated,
                    patchSize=(int(plate_width), int(plate_height)),
                    center=(int(plate_cx), int(plate_cy))
                )

                # 번호판 비율 검증
                if img_cropped.shape[1] > 0 and img_cropped.shape[0] > 0:
                    plate_ratio = img_cropped.shape[1] / img_cropped.shape[0]

                    if params['MIN_PLATE_RATIO'] < plate_ratio < params['MAX_PLATE_RATIO']:
                        plate_candidates.append({
                            'x': int(plate_cx - plate_width / 2),
                            'y': int(plate_cy - plate_height / 2),
                            'w': int(plate_width),
                            'h': int(plate_height),
                            'cropped_img': img_cropped,
                            'confidence': 0.8,
                            'char_count': len(sorted_chars)
                        })

            except Exception as e:
                logger.warning(f"번호판 영역 처리 중 오류: {e}")
                continue

        # 문자 수가 많은 순으로 정렬
        plate_candidates.sort(key=lambda x: x['char_count'], reverse=True)
        return plate_candidates

    def _opencv_detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """기본 OpenCV를 사용한 번호판 탐지 (fallback)"""
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
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)

                if (2.0 < aspect_ratio < 5.0 and
                    area > 2000 and
                    w > 80 and h > 20):

                    plates.append((x, y, x + w, y + h, 0.7))

            plates.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            return plates[:5]

        except Exception as e:
            logger.error(f"기본 OpenCV 탐지 오류: {e}")
            return []

    def preprocess_plate_region_advanced(self, plate_img: np.ndarray) -> np.ndarray:
        """고급 번호판 영역 전처리 (첨부된 코드 기반)"""
        try:
            # 이미지 크기 조정
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)

            # 그레이스케일 변환
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img

            # OTSU 이진화
            _, binary = cv2.threshold(gray, thresh=0.0, maxval=255.0,
                                    type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # 문자 영역만 추출하여 더 정확한 번호판 영역 구하기
            contours, _ = cv2.findContours(binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            # 문자 영역의 경계 찾기
            plate_min_x, plate_min_y = binary.shape[1], binary.shape[0]
            plate_max_x, plate_max_y = 0, 0

            params = self.char_detection_params
            valid_char_found = False

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                ratio = w / h

                if (area > params['MIN_AREA'] and
                    w > params['MIN_WIDTH'] and h > params['MIN_HEIGHT'] and
                    params['MIN_RATIO'] < ratio < params['MAX_RATIO']):

                    valid_char_found = True
                    if x < plate_min_x:
                        plate_min_x = x
                    if y < plate_min_y:
                        plate_min_y = y
                    if x + w > plate_max_x:
                        plate_max_x = x + w
                    if y + h > plate_max_y:
                        plate_max_y = y + h

            if valid_char_found:
                # 문자 영역만 추출
                img_result = binary[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

                # 추가 블러 및 이진화
                img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
                _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0,
                                            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # 패딩 추가
                img_result = cv2.copyMakeBorder(
                    img_result,
                    top=10, bottom=10, left=10, right=10,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )

                return img_result
            else:
                return binary

        except Exception as e:
            logger.error(f"고급 전처리 오류: {e}")
            return plate_img

    def preprocess_plate_region(self, plate_img: np.ndarray) -> np.ndarray:
        """번호판 영역 전처리 (일반 + 고급 방식)"""
        if self.use_advanced_preprocessing:
            return self.preprocess_plate_region_advanced(plate_img)
        else:
            return self._basic_preprocess(plate_img)

    def _basic_preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """기본 전처리"""
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
            logger.error(f"기본 전처리 오류: {e}")
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
                return self._recognize_with_tesseract_advanced(processed_img)

            else:
                # Fallback to tesseract
                if HAS_TESSERACT:
                    return self._recognize_with_tesseract_advanced(processed_img)
                else:
                    return "NO_OCR_AVAILABLE"

        except Exception as e:
            logger.error(f"OCR 처리 중 오류: {e}")
            return ""

    def _recognize_with_tesseract_advanced(self, plate_img: np.ndarray) -> str:
        """고급 Tesseract OCR (첨부된 코드 기반)"""
        if not HAS_TESSERACT:
            return "TESSERACT_NOT_AVAILABLE"

        try:
            # 여러 PSM 모드로 시도
            psm_modes = [7, 8, 6, 13]  # 7: 단일 텍스트 라인, 8: 단일 단어
            best_result = ""
            best_confidence = 0

            for psm in psm_modes:
                try:
                    # 한국어 + 영어 설정
                    config = f'--psm {psm} --oem 0'

                    # 원본 이미지로 시도
                    text1 = pytesseract.image_to_string(
                        plate_img,
                        lang='kor',
                        config=config
                    ).strip()

                    # 반전 이미지로도 시도
                    inverted = cv2.bitwise_not(plate_img)
                    text2 = pytesseract.image_to_string(
                        inverted,
                        lang='kor',
                        config=config
                    ).strip()

                    # 더 나은 결과 선택
                    for text in [text1, text2]:
                        cleaned = self.clean_plate_text_advanced(text)
                        if self.is_valid_korean_plate(cleaned):
                            return cleaned

                        if len(cleaned) > len(best_result):
                            best_result = cleaned

                except Exception as e:
                    logger.debug(f"PSM {psm} 모드 실패: {e}")
                    continue

            return best_result

        except Exception as e:
            logger.error(f"고급 Tesseract OCR 처리 중 오류: {e}")
            return ""

    def clean_plate_text_advanced(self, text: str) -> str:
        """고급 번호판 텍스트 정리 (첨부된 코드 기반)"""
        if not text:
            return ""

        result_chars = ''
        has_digit = False

        # 한글 문자와 숫자만 추출
        for c in text:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        # 숫자가 포함된 경우만 유효한 번호판으로 판단
        if not has_digit:
            return ""

        # 일반적인 OCR 오류 수정
        corrections = {
            'O': '0', 'I': '1', 'l': '1', 'S': '5', 'Z': '2',
            'B': '8', 'G': '6', 'D': '0', 'Q': '0'
        }

        for wrong, correct in corrections.items():
            result_chars = result_chars.replace(wrong, correct)

        return result_chars

    def _recognize_with_pororo(self, plate_img: np.ndarray) -> str:
        """Pororo OCR을 사용한 텍스트 인식"""
        try:
            if HAS_ORIGINAL_PORORO:
                # 원본 Pororo 사용 - PIL Image 또는 이미지 경로 필요
                import tempfile

                # numpy 이미지를 임시 파일로 저장
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name

                cv2.imwrite(temp_path, plate_img)

                try:
                    # Pororo OCR 수행
                    result = self.pororo_ocr(temp_path)

                    # 결과 처리
                    if isinstance(result, dict) and 'description' in result:
                        text = ' '.join(result['description'])
                    elif isinstance(result, str):
                        text = result
                    elif isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'description' in result[0]:
                            text = result[0]['description']
                        else:
                            text = ' '.join(str(item) for item in result)
                    else:
                        text = ""

                    return self.clean_plate_text_advanced(text)

                finally:
                    # 임시 파일 삭제
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

            else:
                # pororo-ocr 라이브러리 사용 - numpy 배열 직접 처리 가능
                result = self.pororo_ocr(plate_img)

                # 결과 처리
                if isinstance(result, dict) and 'description' in result:
                    text = ' '.join(result['description'])
                elif isinstance(result, str):
                    text = result
                elif isinstance(result, list):
                    text = ' '.join(str(item) for item in result)
                else:
                    text = str(result)

                return self.clean_plate_text_advanced(text)

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

                return self.clean_plate_text_advanced(best_result)

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

                return self.clean_plate_text_advanced(best_result)

            return ""

        except Exception as e:
            logger.error(f"PaddleOCR 처리 오류: {e}")
            return ""

    def clean_plate_text(self, text: str) -> str:
        """기본 번호판 텍스트 정리 (호환성 유지)"""
        return self.clean_plate_text_advanced(text)

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
        이미지에서 번호판 인식 수행 (향상된 버전)

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

            logger.info(f"이미지 크기: {image.shape}")

            # 번호판 탐지
            plates = self.detect_license_plates(image)

            if not plates:
                logger.warning("번호판을 탐지하지 못했습니다.")
                return "", image

            logger.info(f"탐지된 번호판 수: {len(plates)}")

            # 모든 탐지된 번호판에서 텍스트 인식 시도
            best_plate_text = ""
            best_confidence = 0
            best_plate_info = None

            for i, plate in enumerate(plates):
                x1, y1, x2, y2, confidence = plate
                logger.debug(f"번호판 {i+1}: ({x1},{y1}) - ({x2},{y2}), 신뢰도: {confidence:.3f}")

                # 번호판 영역 추출
                plate_region = image[y1:y2, x1:x2]

                if plate_region.size == 0:
                    continue

                # 텍스트 인식
                plate_text = self.recognize_text(plate_region)
                logger.debug(f"번호판 {i+1} 인식 결과: '{plate_text}'")

                # 유효한 번호판이고 더 높은 신뢰도면 업데이트
                if plate_text and self.is_valid_korean_plate(plate_text):
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_plate_text = plate_text
                        best_plate_info = (x1, y1, x2, y2)
                        logger.info(f"✅ 유효한 번호판 발견: {plate_text} (신뢰도: {confidence:.3f})")
                elif plate_text and not best_plate_text:
                    # 유효하지 않지만 다른 결과가 없으면 임시로 저장
                    best_plate_text = plate_text
                    best_plate_info = (x1, y1, x2, y2)

            # 결과 시각화
            result_image = image.copy()

            if best_plate_info:
                x1, y1, x2, y2 = best_plate_info
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 텍스트 표시
                label = f"{best_plate_text} ({best_confidence:.2f})" if best_plate_text else f"Detected ({best_confidence:.2f})"
                cv2.putText(result_image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 모든 탐지된 영역 표시 (반투명)
            for plate in plates:
                x1, y1, x2, y2, conf = plate
                if (x1, y1, x2, y2) != best_plate_info:
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # 결과 저장
            if save_result and best_plate_text:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                # 파일명에서 특수문자 제거
                safe_plate_text = re.sub(r'[^\w가-힣]', '_', best_plate_text)
                result_path = f"result_{base_name}_{safe_plate_text}.jpg"
                cv2.imwrite(result_path, result_image)
                logger.info(f"결과 이미지 저장: {result_path}")

            return best_plate_text, result_image

        except Exception as e:
            logger.error(f"이미지 처리 실패: {e}")
            return "", cv2.imread(image_path) if os.path.exists(image_path) else np.zeros((100, 100, 3))

    def debug_detection_process(self, image_path: str, save_debug: bool = True):
        """
        탐지 과정을 단계별로 디버깅하여 시각화

        Args:
            image_path: 입력 이미지 경로
            save_debug: 디버그 이미지 저장 여부
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 이미지 로드 실패: {image_path}")
                return

            print(f"🔍 디버그 모드: {image_path}")
            print(f"   이미지 크기: {image.shape}")
            print(f"   고급 전처리: {'✅' if self.use_advanced_preprocessing else '❌'}")

            # 1. YOLO 탐지 시도
            if self.yolo_model is not None:
                yolo_plates = self._yolo_detect(image)
                print(f"   YOLO 탐지 결과: {len(yolo_plates)}개")
                for i, plate in enumerate(yolo_plates):
                    x1, y1, x2, y2, conf = plate
                    print(f"     YOLO {i+1}: ({x1},{y1})-({x2},{y2}) 신뢰도:{conf:.3f}")

            # 2. 고급 OpenCV 탐지
            if self.use_advanced_preprocessing:
                opencv_plates = self._advanced_opencv_detect(image)
                print(f"   고급 OpenCV 탐지 결과: {len(opencv_plates)}개")
                for i, plate in enumerate(opencv_plates):
                    x1, y1, x2, y2, conf = plate
                    print(f"     OpenCV {i+1}: ({x1},{y1})-({x2},{y2}) 신뢰도:{conf:.3f}")

            # 3. 기본 OpenCV 탐지
            basic_plates = self._opencv_detect(image)
            print(f"   기본 OpenCV 탐지 결과: {len(basic_plates)}개")

            # 4. 최종 탐지 결과
            final_plates = self.detect_license_plates(image)
            print(f"   최종 탐지 결과: {len(final_plates)}개")

            # 각 탐지된 영역에서 OCR 시도
            for i, plate in enumerate(final_plates):
                x1, y1, x2, y2, conf = plate
                plate_region = image[y1:y2, x1:x2]

                if plate_region.size > 0:
                    ocr_result = self.recognize_text(plate_region)
                    is_valid = self.is_valid_korean_plate(ocr_result)
                    print(f"   영역 {i+1} OCR: '{ocr_result}' {'✅' if is_valid else '❌'}")

                    # 디버그 이미지 저장
                    if save_debug:
                        debug_filename = f"debug_plate_{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                        cv2.imwrite(debug_filename, plate_region)

        except Exception as e:
            logger.error(f"디버그 처리 실패: {e}")

def main():
    """테스트 실행"""
    print("🚗 YOLOv8 번호판 인식기 v2.1 테스트 (고급 OpenCV 통합)")

    # 사용 가능한 OCR 엔진 확인
    print("📊 사용 가능한 OCR 엔진:")
    print(f"   - Pororo: {'✅' if HAS_PORORO else '❌'}")
    if HAS_PORORO:
        print(f"     타입: {'pororo-ocr (가벼운 버전)' if not HAS_ORIGINAL_PORORO else '원본 pororo'}")
        if not HAS_ORIGINAL_PORORO:
            try:
                langs = prrocr.ocr.get_available_langs()
                models = prrocr.ocr.get_available_models()
                print(f"     지원 언어: {langs}")
                print(f"     지원 모델: {models}")
            except:
                pass
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

        # 고급 전처리 ON/OFF 비교
        for advanced in [True, False]:
            mode_name = "고급 모드" if advanced else "기본 모드"
            print(f"\n🔧 {mode_name} 테스트:")

            for engine in engines_to_test:
                print(f"\n🔍 {engine.upper()} + {mode_name}:")
                try:
                    recognizer = YOLOv8LicensePlateRecognizer(
                        ocr_engine=engine,
                        use_advanced_preprocessing=advanced,
                        confidence_threshold=0.3  # 낮은 임계값으로 더 많은 탐지
                    )

                    plate_text, _ = recognizer.process_image(test_image, save_result=False)
                    print(f"   결과: {plate_text}")

                    # 디버그 정보
                    if engine == 'tesseract':  # 첫 번째만 디버그
                        recognizer.debug_detection_process(test_image, save_debug=False)

                except Exception as e:
                    print(f"   오류: {e}")
    else:
        print(f"\n⚠️ 테스트 이미지가 없습니다: {test_image}")
        print("테스트 이미지를 추가하고 다시 실행해보세요.")

    # 자동 모드 테스트
    print(f"\n🤖 Auto 모드 테스트:")
    try:
        auto_recognizer = YOLOv8LicensePlateRecognizer(
            ocr_engine='auto',
            use_advanced_preprocessing=True,
            confidence_threshold=0.3
        )
        print(f"   선택된 엔진: {auto_recognizer.ocr_engine}")
        print(f"   고급 전처리: {auto_recognizer.use_advanced_preprocessing}")
    except Exception as e:
        print(f"   오류: {e}")

if __name__ == "__main__":
    main()