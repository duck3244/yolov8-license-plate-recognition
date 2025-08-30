"""
license_plate_recognizer.py
YOLOv8 기반 차량 번호판 인식 엔진

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
import re
from typing import List, Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv8LicensePlateRecognizer:
    """YOLOv8 기반 번호판 인식기"""
    
    def __init__(self, 
                 yolo_model_path: str = 'yolov8n.pt',
                 tesseract_cmd: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """
        YOLOv8 기반 번호판 인식기 초기화
        
        Args:
            yolo_model_path: YOLOv8 모델 경로 (custom trained model 또는 pretrained)
            tesseract_cmd: Tesseract 실행 파일 경로
            confidence_threshold: YOLO 탐지 임계값
        """
        # YOLOv8 모델 로드
        try:
            self.yolo_model = YOLO(yolo_model_path)
            logger.info(f"YOLOv8 모델 로드 완료: {yolo_model_path}")
        except Exception as e:
            logger.error(f"YOLOv8 모델 로드 실패: {e}")
            raise
        
        # Tesseract 설정
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.confidence_threshold = confidence_threshold
        
        # 한국 번호판 패턴 (예: 12가1234, 123가1234)
        self.korean_plate_patterns = [
            r'\d{2}[가-힣]\d{4}',  # 12가1234
            r'\d{3}[가-힣]\d{4}',  # 123가1234
            r'\d{2}[가-힣]\d{3,4}',  # 유연한 패턴
        ]
    
    def detect_license_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        YOLOv8을 사용하여 번호판 탐지
        
        Args:
            image: 입력 이미지 (BGR format)
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        results = self.yolo_model(image)
        plates = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 번호판 클래스 또는 충분한 신뢰도를 가진 객체 필터링
                    confidence = float(box.conf)
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        plates.append((x1, y1, x2, y2, confidence))
        
        return plates
    
    def preprocess_plate_region(self, plate_img: np.ndarray) -> np.ndarray:
        """
        번호판 영역 전처리 (OCR 정확도 향상을 위해)
        
        Args:
            plate_img: 번호판 영역 이미지
            
        Returns:
            전처리된 이미지
        """
        # 그레이스케일 변환
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        # 이미지 크기 조정 (OCR 정확도 향상)
        height, width = gray.shape
        if height < 50:
            scale_factor = 50 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 이진화 (여러 방법 시도)
        binary_methods = [
            lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2),
            lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        ]
        
        # 가장 좋은 결과를 위해 여러 방법 시도
        best_result = None
        best_score = 0
        
        for method in binary_methods:
            try:
                binary = method(enhanced)
                # 텍스트 영역 비율로 품질 평가
                text_ratio = np.sum(binary == 0) / binary.size
                if 0.1 < text_ratio < 0.5:  # 적절한 텍스트 비율
                    score = 1.0 - abs(0.3 - text_ratio)
                    if score > best_score:
                        best_score = score
                        best_result = binary
            except:
                continue
        
        return best_result if best_result is not None else enhanced
    
    def recognize_text(self, plate_img: np.ndarray) -> str:
        """
        Tesseract를 사용하여 번호판 텍스트 인식
        
        Args:
            plate_img: 번호판 이미지
            
        Returns:
            인식된 텍스트
        """
        # 전처리
        processed_img = self.preprocess_plate_region(plate_img)
        
        # Tesseract 설정 (한국어 + 영어)
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후'
        
        try:
            # 여러 PSM 모드 시도
            psm_modes = [8, 7, 6, 13]
            best_text = ""
            best_confidence = 0
            
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후'
                
                # 원본 이미지로 시도
                text1 = pytesseract.image_to_string(processed_img, lang='kor+eng', config=config).strip()
                
                # 반전 이미지로도 시도
                inverted = cv2.bitwise_not(processed_img)
                text2 = pytesseract.image_to_string(inverted, lang='kor+eng', config=config).strip()
                
                # 더 나은 결과 선택
                for text in [text1, text2]:
                    if self.is_valid_korean_plate(text):
                        return self.clean_plate_text(text)
                    
                    if len(text) > len(best_text):
                        best_text = text
            
            return self.clean_plate_text(best_text)
            
        except Exception as e:
            logger.error(f"OCR 처리 중 오류: {e}")
            return ""
    
    def is_valid_korean_plate(self, text: str) -> bool:
        """
        한국 번호판 형식 검증
        
        Args:
            text: 인식된 텍스트
            
        Returns:
            유효한 번호판 형식인지 여부
        """
        cleaned_text = re.sub(r'[^\w가-힣]', '', text)
        
        for pattern in self.korean_plate_patterns:
            if re.match(pattern, cleaned_text):
                return True
        return False
    
    def clean_plate_text(self, text: str) -> str:
        """
        번호판 텍스트 정리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정리된 텍스트
        """
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
    
    def process_image(self, image_path: str, save_result: bool = True) -> Tuple[str, np.ndarray]:
        """
        이미지에서 번호판 인식 수행
        
        Args:
            image_path: 입력 이미지 경로
            save_result: 결과 이미지 저장 여부
            
        Returns:
            (인식된 번호판 텍스트, 결과 이미지)
        """
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
        cv2.putText(result_image, f"{plate_text} ({confidence:.2f})", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 결과 저장
        if save_result:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = f"result_{base_name}.jpg"
            cv2.imwrite(result_path, result_image)
            logger.info(f"결과 이미지 저장: {result_path}")
        
        return plate_text, result_image
    
    def process_video(self, video_path: str, output_path: str = "output_video.mp4"):
        """
        비디오에서 번호판 인식 수행
        
        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
        """
        cap = cv2.VideoCapture(video_path)
        
        # 비디오 정보 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 비디오 작성기 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        recognized_plates = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 일정 간격으로만 처리 (성능 최적화)
            if frame_count % 10 == 0:
                plates = self.detect_license_plates(frame)
                
                for x1, y1, x2, y2, confidence in plates:
                    plate_region = frame[y1:y2, x1:x2]
                    plate_text = self.recognize_text(plate_region)
                    
                    if plate_text:
                        recognized_plates.append(plate_text)
                        
                    # 결과 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{plate_text} ({confidence:.2f})", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        logger.info(f"비디오 처리 완료: {output_path}")
        logger.info(f"인식된 번호판들: {set(recognized_plates)}")

def main():
    """메인 실행 함수"""
    
    # 번호판 인식기 초기화
    recognizer = YOLOv8LicensePlateRecognizer(
        yolo_model_path='yolov8n.pt',  # 또는 custom trained model
        confidence_threshold=0.5
    )
    
    # 예제 이미지 처리
    try:
        image_path = "test_car.jpg"  # 테스트 이미지 경로
        if os.path.exists(image_path):
            plate_text, result_img = recognizer.process_image(image_path)
            print(f"인식된 번호판: {plate_text}")
            
            # 결과 표시
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            plt.title('원본 이미지')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f'결과: {plate_text}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"테스트 이미지가 없습니다: {image_path}")
            print("이미지 파일을 준비한 후 다시 실행해주세요.")
            
    except Exception as e:
        logger.error(f"이미지 처리 중 오류: {e}")

if __name__ == "__main__":
    main()
