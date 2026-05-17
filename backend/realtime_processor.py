"""
realtime_processor.py
실시간 번호판 인식 및 스트림 처리

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import logging

from database_manager import DatabaseManager, PlateDetection

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """실시간 번호판 인식 처리기"""
    
    def __init__(self, 
                 recognizer, 
                 db_manager: DatabaseManager,
                 frame_skip: int = 10,
                 max_queue_size: int = 10):
        """
        실시간 처리기 초기화
        
        Args:
            recognizer: 번호판 인식기 객체
            db_manager: 데이터베이스 매니저
            frame_skip: 처리할 프레임 간격 (성능 최적화)
            max_queue_size: 최대 큐 크기
        """
        self.recognizer = recognizer
        self.db_manager = db_manager
        self.frame_skip = frame_skip
        
        # 스레딩을 위한 큐
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        
        # 상태 관리
        self.is_running = False
        self.frame_count = 0
        self.process_frame_count = 0
        self.detection_count = 0
        self._lock = threading.Lock()

        # 스레드
        self.capture_thread = None
        self.processing_thread = None
        
        # 콜백 함수들
        self.detection_callback = None
        self.frame_callback = None
        
        # 성능 통계
        self.stats = {
            'fps': 0,
            'detection_rate': 0,
            'avg_processing_time': 0,
            'start_time': None
        }
    
    def set_detection_callback(self, callback: Callable[[PlateDetection, np.ndarray], None]):
        """
        탐지 결과 콜백 함수 설정
        
        Args:
            callback: (detection, frame) -> None 형태의 콜백 함수
        """
        self.detection_callback = callback
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        프레임 처리 콜백 함수 설정
        
        Args:
            callback: (frame) -> None 형태의 콜백 함수
        """
        self.frame_callback = callback
    
    def start_camera_processing(self, camera_index: int = 0, resolution: tuple = (1280, 720)) -> bool:
        """
        카메라를 사용한 실시간 처리 시작
        
        Args:
            camera_index: 카메라 인덱스
            resolution: 해상도 (width, height)
            
        Returns:
            시작 성공 여부
        """
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            # 캡처 스레드 시작
            self.capture_thread = threading.Thread(
                target=self._capture_camera_frames,
                args=(camera_index, resolution),
                daemon=True
            )
            
            # 처리 스레드 시작
            self.processing_thread = threading.Thread(
                target=self._process_frames,
                daemon=True
            )
            
            self.capture_thread.start()
            self.processing_thread.start()
            
            logger.info(f"🎥 실시간 카메라 처리 시작 (카메라: {camera_index}, 해상도: {resolution})")
            return True
            
        except Exception as e:
            logger.error(f"카메라 처리 시작 실패: {e}")
            self.is_running = False
            return False
    
    def start_video_processing(self, video_path: str) -> bool:
        """
        비디오 파일을 사용한 실시간 처리 시작
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            시작 성공 여부
        """
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            # 캡처 스레드 시작
            self.capture_thread = threading.Thread(
                target=self._capture_video_frames,
                args=(video_path,),
                daemon=True
            )
            
            # 처리 스레드 시작
            self.processing_thread = threading.Thread(
                target=self._process_frames,
                daemon=True
            )
            
            self.capture_thread.start()
            self.processing_thread.start()
            
            logger.info(f"📹 비디오 파일 처리 시작: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"비디오 처리 시작 실패: {e}")
            self.is_running = False
            return False
    
    def _capture_camera_frames(self, camera_index: int, resolution: tuple):
        """카메라 프레임 캡처"""
        cap = cv2.VideoCapture(camera_index)
        
        # 카메라 설정
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error(f"카메라를 열 수 없습니다: {camera_index}")
            self.is_running = False
            return
        
        logger.info(f"카메라 초기화 완료: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        frame_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("카메라에서 프레임을 읽을 수 없습니다")
                continue
            
            current_time = time.time()

            # FPS 계산
            with self._lock:
                self.frame_count += 1
                if current_time - frame_time > 1.0:
                    self.stats['fps'] = self.frame_count / (current_time - frame_time)
                    self.frame_count = 0
                    frame_time = current_time
            
            # 프레임 콜백 호출
            if self.frame_callback:
                self.frame_callback(frame.copy())
            
            try:
                # 큐가 가득 찬 경우 이전 프레임 제거
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put((frame, current_time), timeout=0.1)
                
            except queue.Full:
                # 큐가 가득 찬 경우 스킵
                pass
            except Exception as e:
                logger.error(f"프레임 큐 처리 오류: {e}")
        
        cap.release()
        logger.info("카메라 캡처 종료")
    
    def _capture_video_frames(self, video_path: str):
        """비디오 파일 프레임 캡처"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {video_path}")
            self.is_running = False
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"비디오 로드 완료: {total_frames} 프레임, {fps} FPS")
        
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30
        last_frame_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logger.info("비디오 파일 처리 완료")
                self.is_running = False
                break
            
            current_time = time.time()

            # 원본 FPS에 맞춰 재생
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            with self._lock:
                self.frame_count += 1
            
            # 프레임 콜백 호출
            if self.frame_callback:
                self.frame_callback(frame.copy())
            
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put((frame, current_time), timeout=0.1)
                
            except queue.Full:
                pass
            except Exception as e:
                logger.error(f"비디오 프레임 큐 처리 오류: {e}")
            
            last_frame_time = current_time
        
        cap.release()
        logger.info("비디오 캡처 종료")
    
    def _process_frames(self):
        """프레임 처리 스레드"""
        processing_times = []
        
        while self.is_running:
            try:
                # 프레임 가져오기 (타임아웃 1초)
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, capture_time = frame_data

                # 프레임 스키핑 (성능 최적화)
                self.process_frame_count += 1
                if self.process_frame_count % self.frame_skip != 0:
                    continue
                
                start_time = time.time()
                
                # 번호판 탐지
                plates = self.recognizer.detect_license_plates(frame)
                
                if plates:
                    # 가장 신뢰도가 높은 번호판 처리
                    best_plate = max(plates, key=lambda x: x[4])
                    x1, y1, x2, y2, confidence = best_plate
                    
                    # 번호판 영역 추출
                    plate_region = frame[y1:y2, x1:x2]
                    
                    # 텍스트 인식
                    plate_text = self.recognizer.recognize_text(plate_region)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # 평균 처리 시간 계산 (최근 100개)
                    if len(processing_times) > 100:
                        processing_times = processing_times[-100:]
                    
                    self.stats['avg_processing_time'] = np.mean(processing_times)
                    
                    if plate_text and len(plate_text.strip()) > 3:
                        with self._lock:
                            self.detection_count += 1

                            # 탐지율 계산
                            if self.stats['start_time']:
                                elapsed_time = time.time() - self.stats['start_time']
                                self.stats['detection_rate'] = self.detection_count / elapsed_time * 60  # 분당 탐지수
                        
                        # 탐지 결과 객체 생성
                        detection = PlateDetection(
                            plate_number=plate_text,
                            confidence=confidence,
                            timestamp=datetime.fromtimestamp(capture_time),
                            bbox=(x1, y1, x2, y2),
                            processing_time=processing_time
                        )
                        
                        # 데이터베이스에 저장
                        self.db_manager.save_detection(detection)
                        
                        # 결과 시각화
                        result_frame = frame.copy()
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(result_frame, f"{plate_text} ({confidence:.2f})", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        
                        # 탐지 콜백 호출
                        if self.detection_callback:
                            self.detection_callback(detection, result_frame)
                        
                        # 결과 큐에 추가
                        try:
                            self.result_queue.put({
                                'frame': result_frame,
                                'detection': detection,
                                'timestamp': capture_time
                            }, timeout=0.1)
                        except queue.Full:
                            # 큐가 가득 찬 경우 이전 결과 제거
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put({
                                    'frame': result_frame,
                                    'detection': detection,
                                    'timestamp': capture_time
                                })
                            except queue.Empty:
                                pass
                        
                        logger.debug(f"번호판 탐지: {plate_text} (신뢰도: {confidence:.3f})")
                
            except queue.Empty:
                # 타임아웃 - 정상적인 상황
                continue
            except Exception as e:
                logger.error(f"프레임 처리 중 오류: {e}")
                time.sleep(0.1)
    
    def get_latest_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        최신 처리 결과 가져오기
        
        Args:
            timeout: 타임아웃 (초)
            
        Returns:
            최신 처리 결과 또는 None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        실시간 처리 통계 반환
        
        Returns:
            처리 통계 정보
        """
        stats = self.stats.copy()
        stats['detection_count'] = self.detection_count
        stats['frame_count'] = self.frame_count
        stats['is_running'] = self.is_running
        stats['queue_sizes'] = {
            'frame_queue': self.frame_queue.qsize(),
            'result_queue': self.result_queue.qsize()
        }
        
        return stats
    
    def stop_processing(self):
        """실시간 처리 중단"""
        if self.is_running:
            logger.info("실시간 처리 중단 중...")
            self.is_running = False
            
            # 스레드 종료 대기
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5.0)
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            logger.info("실시간 처리 완전 중단")

class StreamingServer:
    """웹 스트리밍 서버"""
    
    def __init__(self, realtime_processor: RealTimeProcessor, port: int = 8080):
        self.realtime_processor = realtime_processor
        self.port = port
        self.is_running = False
        
    def start_streaming(self):
        """스트리밍 시작"""
        from flask import Flask, Response, jsonify
        
        app = Flask(__name__)
        
        @app.route('/stream')
        def video_stream():
            """비디오 스트림 엔드포인트"""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @app.route('/stream_info')
        def stream_info():
            """스트림 정보 API"""
            stats = self.realtime_processor.get_statistics()
            return jsonify(stats)
        
        self.is_running = True
        logger.info(f"🔴 스트리밍 서버 시작: http://localhost:{self.port}/stream")
        app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
    
    def _generate_frames(self):
        """프레임 생성기"""
        while self.is_running and self.realtime_processor.is_running:
            result = self.realtime_processor.get_latest_result(timeout=1.0)
            
            if result:
                frame = result['frame']
                
                # JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # 빈 프레임 전송
                time.sleep(0.1)
    
    def stop_streaming(self):
        """스트리밍 중단"""
        self.is_running = False

class WebcamViewer:
    """웹캠 실시간 뷰어 (OpenCV 윈도우)"""
    
    def __init__(self, realtime_processor: RealTimeProcessor):
        self.realtime_processor = realtime_processor
        self.display_window = "License Plate Recognition - Live"
        
    def start_viewer(self, camera_index: int = 0):
        """웹캠 뷰어 시작"""
        # 실시간 처리 시작
        if not self.realtime_processor.start_camera_processing(camera_index):
            logger.error("실시간 처리 시작 실패")
            return
        
        # OpenCV 윈도우 생성
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_window, 1280, 720)
        
        logger.info("웹캠 뷰어 시작. 'q' 키를 눌러 종료하세요.")
        
        last_update = time.time()
        
        while True:
            # 최신 결과 가져오기
            result = self.realtime_processor.get_latest_result(timeout=0.1)
            
            if result:
                frame = result['frame']
                detection = result['detection']
                
                # 통계 정보 오버레이
                stats = self.realtime_processor.get_statistics()
                self._draw_overlay(frame, stats, detection)
                
                cv2.imshow(self.display_window, frame)
                last_update = time.time()
            
            # 키보드 입력 체크
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # 스크린샷
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"screenshot_{timestamp}.jpg"
                if result:
                    cv2.imwrite(screenshot_path, result['frame'])
                    logger.info(f"스크린샷 저장: {screenshot_path}")
            elif key == ord(' '):  # 일시정지/재개
                if self.realtime_processor.is_running:
                    self.realtime_processor.stop_processing()
                    logger.info("처리 일시정지")
                else:
                    self.realtime_processor.start_camera_processing(camera_index)
                    logger.info("처리 재개")
        
        # 정리
        self.realtime_processor.stop_processing()
        cv2.destroyAllWindows()
        logger.info("웹캠 뷰어 종료")
    
    def _draw_overlay(self, frame: np.ndarray, stats: Dict[str, Any], detection: Optional[PlateDetection]):
        """프레임에 통계 정보 오버레이"""
        height, width = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 텍스트 정보
        texts = [
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Detections: {stats.get('detection_count', 0)}",
            f"Detection Rate: {stats.get('detection_rate', 0):.1f}/min",
            f"Avg Process Time: {stats.get('avg_processing_time', 0):.3f}s"
        ]
        
        if detection:
            texts.append(f"Last Detected: {detection.plate_number}")
        
        y_offset = 30
        for text in texts:
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # 제어 안내
        controls = [
            "Controls:",
            "Q - Quit",
            "S - Screenshot", 
            "SPACE - Pause/Resume"
        ]
        
        y_offset = height - 100
        for control in controls:
            cv2.putText(frame, control, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

def main():
    """메인 실행 함수 - 웹캠 뷰어 데모"""
    try:
        from license_plate_recognizer import YOLOv8LicensePlateRecognizer
        
        # 컴포넌트 초기화
        recognizer = YOLOv8LicensePlateRecognizer()
        db_manager = DatabaseManager()
        realtime_processor = RealTimeProcessor(recognizer, db_manager)
        
        # 탐지 콜백 설정
        def on_detection(detection: PlateDetection, frame: np.ndarray):
            print(f"🚗 탐지됨: {detection.plate_number} (신뢰도: {detection.confidence:.3f})")
        
        realtime_processor.set_detection_callback(on_detection)
        
        # 웹캠 뷰어 시작
        viewer = WebcamViewer(realtime_processor)
        viewer.start_viewer(camera_index=0)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"실시간 처리 오류: {e}")

if __name__ == "__main__":
    main()