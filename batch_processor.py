"""
batch_processor.py
대용량 이미지 배치 처리

Author: License Plate Recognition Team
Date: 2025-08-29
Version: 2.0.0
"""

import cv2
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import pandas as pd

from database_manager import DatabaseManager, PlateDetection

logger = logging.getLogger(__name__)

class BatchProcessor:
    """배치 이미지 처리기"""
    
    def __init__(self, recognizer, db_manager: DatabaseManager):
        """
        배치 처리기 초기화
        
        Args:
            recognizer: 번호판 인식기
            db_manager: 데이터베이스 매니저
        """
        self.recognizer = recognizer
        self.db_manager = db_manager
        
        # 지원하는 이미지 확장자
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 진행 상황 추적
        self.progress_callback = None
        self.error_callback = None
        
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        진행 상황 콜백 설정
        
        Args:
            callback: (current, total, message) -> None
        """
        self.progress_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, Exception], None]):
        """
        에러 콜백 설정
        
        Args:
            callback: (image_path, error) -> None
        """
        self.error_callback = callback
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str = None, 
                         max_workers: int = 4,
                         save_results: bool = True,
                         create_report: bool = True) -> Dict[str, Any]:
        """
        디렉토리의 모든 이미지 배치 처리
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리 (None이면 입력 디렉토리/processed)
            max_workers: 최대 워커 스레드 수
            save_results: 결과 이미지 저장 여부
            create_report: 보고서 생성 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_path = input_path / "processed"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True)
        
        # 이미지 파일 목록 수집
        image_files = self._collect_image_files(input_path)
        
        if not image_files:
            logger.warning(f"처리할 이미지가 없습니다: {input_dir}")
            return {
                'success': False,
                'error': '처리할 이미지가 없습니다',
                'processed_count': 0,
                'total_count': 0
            }
        
        logger.info(f"📁 배치 처리 시작: {len(image_files)}개 이미지")
        start_time = time.time()
        
        # 병렬 처리
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_file = {
                executor.submit(
                    self._process_single_image, 
                    str(img_file), 
                    output_path if save_results else None
                ): img_file 
                for img_file in image_files
            }
            
            # 결과 수집
            for i, future in enumerate(as_completed(future_to_file)):
                img_file = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 진행 상황 업데이트
                    if self.progress_callback:
                        self.progress_callback(
                            i + 1, 
                            len(image_files), 
                            f"처리 완료: {img_file.name}"
                        )
                    
                    if result['success']:
                        logger.debug(f"✅ {img_file.name}: {result['plate_number']}")
                    else:
                        logger.warning(f"❌ {img_file.name}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ {img_file.name}: {e}")
                    
                    # 실패 결과 추가
                    results.append({
                        'image_path': str(img_file),
                        'success': False,
                        'error': str(e),
                        'plate_number': '',
                        'confidence': 0.0,
                        'processing_time': 0.0
                    })
                    
                    # 에러 콜백 호출
                    if self.error_callback:
                        self.error_callback(str(img_file), e)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 결과 통계
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        summary = {
            'success': True,
            'total_count': len(results),
            'processed_count': len(successful_results),
            'failed_count': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'total_time': total_time,
            'avg_time_per_image': total_time / len(results),
            'results': results,
            'input_dir': str(input_path),
            'output_dir': str(output_path),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📊 배치 처리 완료:")
        logger.info(f"   - 총 이미지: {summary['total_count']}")
        logger.info(f"   - 성공: {summary['processed_count']}")
        logger.info(f"   - 실패: {summary['failed_count']}")
        logger.info(f"   - 성공률: {summary['success_rate']:.1f}%")
        logger.info(f"   - 총 처리 시간: {summary['total_time']:.2f}초")
        logger.info(f"   - 평균 처리 시간: {summary['avg_time_per_image']:.3f}초/이미지")
        
        # 보고서 생성
        if create_report:
            report_path = output_path / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"📋 처리 보고서 생성: {report_path}")
            
            # CSV 보고서도 생성
            self._create_csv_report(results, output_path)
        
        return summary
    
    def _collect_image_files(self, directory: Path) -> List[Path]:
        """디렉토리에서 이미지 파일 수집"""
        image_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def _process_single_image(self, image_path: str, output_dir: Optional[Path]) -> Dict[str, Any]:
        """
        단일 이미지 처리
        
        Args:
            image_path: 이미지 경로
            output_dir: 결과 저장 디렉토리 (None이면 저장 안함)
            
        Returns:
            처리 결과
        """
        start_time = time.time()
        
        try:
            # 이미지 로드 및 처리
            plate_text, result_img = self.recognizer.process_image(
                image_path, 
                save_result=False
            )
            
            processing_time = time.time() - start_time
            
            # 신뢰도 계산 (실제 탐지에서 가져와야 하지만, 여기서는 추정)
            confidence = 0.8 if plate_text else 0.0
            
            # 결과 이미지 저장
            if output_dir and result_img is not None:
                output_filename = f"result_{Path(image_path).stem}.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), result_img)
            
            # 데이터베이스에 저장
            if plate_text:
                detection = PlateDetection(
                    plate_number=plate_text,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    image_path=image_path,
                    processing_time=processing_time
                )
                self.db_manager.save_detection(detection)
            
            return {
                'image_path': image_path,
                'success': True,
                'plate_number': plate_text,
                'confidence': confidence,
                'processing_time': processing_time,
                'output_path': str(output_path) if output_dir else None
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'plate_number': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def _create_csv_report(self, results: List[Dict], output_dir: Path):
        """CSV 형태의 상세 보고서 생성"""
        try:
            df = pd.DataFrame(results)
            csv_path = output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"📊 CSV 보고서 생성: {csv_path}")
        except ImportError:
            logger.warning("pandas가 설치되지 않아 CSV 보고서를 생성할 수 없습니다")
        except Exception as e:
            logger.error(f"CSV 보고서 생성 실패: {e}")
    
    def process_file_list(self, 
                         file_list: List[str],
                         output_dir: str = None,
                         max_workers: int = 4) -> Dict[str, Any]:
        """
        파일 목록 배치 처리
        
        Args:
            file_list: 처리할 파일 경로 목록
            output_dir: 출력 디렉토리
            max_workers: 최대 워커 수
            
        Returns:
            처리 결과
        """
        if not file_list:
            return {
                'success': False,
                'error': '처리할 파일이 없습니다',
                'processed_count': 0,
                'total_count': 0
            }
        
        # 출력 디렉토리 설정
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = None
        
        logger.info(f"📋 파일 목록 배치 처리 시작: {len(file_list)}개 파일")
        start_time = time.time()
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._process_single_image,
                    file_path,
                    output_path
                ): file_path
                for file_path in file_list
            }
            
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if self.progress_callback:
                        self.progress_callback(
                            i + 1,
                            len(file_list),
                            f"처리 완료: {Path(file_path).name}"
                        )
                        
                except Exception as e:
                    logger.error(f"파일 처리 실패 {file_path}: {e}")
                    results.append({
                        'image_path': file_path,
                        'success': False,
                        'error': str(e),
                        'plate_number': '',
                        'confidence': 0.0,
                        'processing_time': 0.0
                    })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful_results = [r for r in results if r['success']]
        
        summary = {
            'success': True,
            'total_count': len(results),
            'processed_count': len(successful_results),
            'failed_count': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'total_time': total_time,
            'avg_time_per_image': total_time / len(results),
            'results': results,
            'output_dir': str(output_path) if output_path else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def resume_failed_processing(self, report_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        실패한 처리 작업 재개
        
        Args:
            report_path: 이전 처리 보고서 경로
            output_dir: 출력 디렉토리
            
        Returns:
            재처리 결과
        """
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                previous_report = json.load(f)
            
            # 실패한 이미지만 추출
            failed_images = [
                result['image_path'] 
                for result in previous_report['results'] 
                if not result['success']
            ]
            
            if not failed_images:
                logger.info("재처리할 실패한 이미지가 없습니다")
                return {
                    'success': True,
                    'message': '재처리할 실패한 이미지가 없습니다',
                    'processed_count': 0,
                    'total_count': 0
                }
            
            logger.info(f"🔄 실패한 {len(failed_images)}개 이미지 재처리 시작")
            
            return self.process_file_list(failed_images, output_dir)
            
        except Exception as e:
            logger.error(f"재처리 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed_count': 0,
                'total_count': 0
            }
    
    def validate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        처리 결과 검증
        
        Args:
            results: 처리 결과 목록
            
        Returns:
            검증 통계
        """
        validation_stats = {
            'total_results': len(results),
            'valid_plates': 0,
            'invalid_plates': 0,
            'empty_results': 0,
            'confidence_distribution': {
                'high': 0,    # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0      # < 0.5
            },
            'processing_time_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0
            }
        }
        
        processing_times = []
        
        for result in results:
            if result['success']:
                plate_number = result['plate_number']
                confidence = result['confidence']
                processing_time = result['processing_time']
                
                processing_times.append(processing_time)
                
                if plate_number:
                    if self.recognizer.is_valid_korean_plate(plate_number):
                        validation_stats['valid_plates'] += 1
                    else:
                        validation_stats['invalid_plates'] += 1
                    
                    # 신뢰도 분포
                    if confidence > 0.8:
                        validation_stats['confidence_distribution']['high'] += 1
                    elif confidence > 0.5:
                        validation_stats['confidence_distribution']['medium'] += 1
                    else:
                        validation_stats['confidence_distribution']['low'] += 1
                else:
                    validation_stats['empty_results'] += 1
        
        # 처리 시간 통계
        if processing_times:
            validation_stats['processing_time_stats'] = {
                'min': min(processing_times),
                'max': max(processing_times),
                'avg': sum(processing_times) / len(processing_times)
            }
        
        return validation_stats

def main():
    """메인 실행 함수 - 배치 처리 데모"""
    import argparse
    
    parser = argparse.ArgumentParser(description='번호판 인식 배치 처리')
    parser.add_argument('input_dir', help='입력 디렉토리')
    parser.add_argument('--output_dir', help='출력 디렉토리')
    parser.add_argument('--workers', type=int, default=4, help='워커 스레드 수')
    parser.add_argument('--no-save', action='store_true', help='결과 이미지 저장 안함')
    
    args = parser.parse_args()
    
    try:
        from license_plate_recognizer import YOLOv8LicensePlateRecognizer
        
        # 진행 상황 콜백
        def show_progress(current: int, total: int, message: str):
            percent = (current / total) * 100
            print(f"\r진행률: {percent:.1f}% ({current}/{total}) - {message}", end='', flush=True)
        
        # 에러 콜백
        def show_error(image_path: str, error: Exception):
            print(f"\n❌ 오류 발생 - {image_path}: {error}")
        
        # 컴포넌트 초기화
        recognizer = YOLOv8LicensePlateRecognizer()
        db_manager = DatabaseManager()
        batch_processor = BatchProcessor(recognizer, db_manager)
        
        # 콜백 설정
        batch_processor.set_progress_callback(show_progress)
        batch_processor.set_error_callback(show_error)
        
        # 배치 처리 실행
        results = batch_processor.process_directory(
            args.input_dir,
            args.output_dir,
            max_workers=args.workers,
            save_results=not args.no_save
        )
        
        print()  # 새로운 줄
        
        if results['success']:
            print("🎉 배치 처리 성공!")
            
            # 검증 수행
            validation = batch_processor.validate_results(results['results'])
            
            print("\n📊 검증 결과:")
            print(f"   - 유효한 번호판: {validation['valid_plates']}")
            print(f"   - 무효한 번호판: {validation['invalid_plates']}")
            print(f"   - 빈 결과: {validation['empty_results']}")
            
            print("\n🎯 신뢰도 분포:")
            conf_dist = validation['confidence_distribution']
            print(f"   - 높음 (>80%): {conf_dist['high']}")
            print(f"   - 중간 (50-80%): {conf_dist['medium']}")
            print(f"   - 낮음 (<50%): {conf_dist['low']}")
            
            time_stats = validation['processing_time_stats']
            print(f"\n⏱️ 처리 시간:")
            print(f"   - 최소: {time_stats['min']:.3f}초")
            print(f"   - 최대: {time_stats['max']:.3f}초")
            print(f"   - 평균: {time_stats['avg']:.3f}초")
        else:
            print(f"❌ 배치 처리 실패: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"배치 처리 데모 실행 실패: {e}")

if __name__ == "__main__":
    main()