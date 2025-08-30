# 📁 YOLOv8 번호판 인식 시스템 - 완성된 프로젝트 구조

## 🗂️ 전체 파일 구조

```
yolov8-license-plate-recognition/
├── 📄 main_app.py                    # 메인 실행 파일
├── 🔧 license_plate_recognizer.py    # 핵심 인식 엔진
├── 🗄️ database_manager.py            # 데이터베이스 관리
├── 🌐 web_interface.py               # 웹 인터페이스
├── 📹 realtime_processor.py          # 실시간 처리
├── 📦 batch_processor.py             # 배치 처리
├── ⚙️ config_manager.py              # 설정 관리
├── 🧪 test_train.py                  # 테스트 및 훈련
├── 📋 requirements.txt               # 의존성 목록
├── 🐳 docker-compose.yml             # Docker 설정
├── 📄 Dockerfile                     # Docker 이미지
├── ⚙️ config.yaml                    # 시스템 설정
├── 🔒 .env.example                   # 환경 변수 예제
├── 📖 README.md                      # 프로젝트 문서
├── 📜 LICENSE                        # 라이선스
├── 🛠️ Makefile                       # 자동화 스크립트
├── 📊 setup.py                       # 패키지 설정
├── 🔧 pyproject.toml                 # 프로젝트 설정
├── 🧪 pytest.ini                     # 테스트 설정
├── 📏 .flake8                        # 코드 스타일
├── 🚫 .gitignore                     # Git 제외 파일
├── 📁 uploads/                       # 업로드된 이미지 저장
├── 📊 logs/                          # 로그 파일 저장
├── 💾 data/                          # 모델 및 데이터 파일
├── 🧪 tests/                         # 테스트 파일들
│   ├── test_recognizer.py
│   ├── test_database.py
│   └── test_web_interface.py
└── 📖 docs/                          # 문서 파일들
    ├── installation.md
    ├── usage.md
    └── api.md
```

## 🎯 각 파일의 역할

### 📄 핵심 모듈

1. **license_plate_recognizer.py**
   - YOLOv8 기반 번호판 탐지
   - Tesseract OCR 통합
   - 이미지/비디오 처리
   - 한국어 번호판 특화 전처리

2. **database_manager.py**
   - SQLite 데이터베이스 관리
   - 탐지 결과 저장/조회
   - 통계 생성
   - 데이터 백업/복원

3. **web_interface.py**
   - Flask 웹 서버
   - REST API 제공
   - 업로드 및 결과 표시
   - 대시보드 UI

4. **realtime_processor.py**
   - 실시간 카메라/비디오 처리
   - 멀티스레딩 최적화
   - 스트리밍 서버
   - 성능 모니터링

5. **batch_processor.py**
   - 대용량 이미지 배치 처리
   - 병렬 처리 최적화
   - 진행 상황 추적
   - 결과 보고서 생성

6. **config_manager.py**
   - YAML 설정 파일 관리
   - 환경 변수 오버라이드
   - 설정 검증
   - 동적 설정 업데이트

7. **main_app.py**
   - 통합 실행 인터페이스
   - CLI 명령 처리
   - 시스템 초기화
   - 에러 핸들링

### 🔧 설정 및 배포 파일

- **config.yaml**: 시스템 전체 설정
- **requirements.txt**: Python 의존성
- **docker-compose.yml**: 컨테이너 배포
- **Dockerfile**: 컨테이너 이미지 정의
- **.env.example**: 환경 변수 템플릿

## 🛠️ 설치 및 실행 가이드

### 1️⃣ 기본 설치

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/yolov8-license-plate-recognition.git
cd yolov8-license-plate-recognition

# 2. 가상환경 생성 (권장)
python -m venv venv

# 3. 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 의존성 설치
pip install -r requirements.txt

# 5. Tesseract OCR 설치 (OS별)
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-kor
# Windows: Tesseract 다운로드 후 설치
# macOS: brew install tesseract tesseract-lang
```

### 2️⃣ 환경 설정

```bash
# 환경 변수 파일 복사 및 수정
cp .env.example .env
# .env 파일을 편집하여 필요한 설정 변경
```

### 3️⃣ 실행 방법

#### 🌐 웹 서버 모드 (권장)
```bash
python main_app.py server
# 웹 브라우저에서 http://localhost:5000 접속
```

#### 📹 실시간 카메라 모드
```bash
python main_app.py camera              # 기본 카메라
python main_app.py camera --camera 1   # 두 번째 카메라
```

#### 🖼️ 단일 이미지 처리
```bash
python main_app.py image test_car.jpg
```

#### 📦 배치 처리
```bash
python main_app.py batch ./input_images --output_dir ./results --workers 8
```

#### 📺 스트리밍 서버
```bash
python main_app.py stream --port 8080
# 브라우저에서 http://localhost:8080/stream 접속
```

### 4️⃣ Docker 실행

```bash
# Docker Compose로 실행 (권장)
docker-compose up -d

# 또는 Docker 직접 빌드
docker build -t yolov8-license-plate .
docker run -p 5000:5000 yolov8-license-plate
```

### 5️⃣ 개발 환경 설정

```bash
# 개발 의존성 설치
make install-dev

# 코드 포맷팅
make format

# 테스트 실행
make test

# 코드 스타일 검사
make lint
```

## 🎮 사용법 예제

### Python API 사용

```python
from license_plate_recognizer import YOLOv8LicensePlateRecognizer

# 인식기 초기화
recognizer = YOLOv8LicensePlateRecognizer()

# 이미지에서 번호판 인식
plate_text, result_img = recognizer.process_image('car_image.jpg')
print(f"인식된 번호판: {plate_text}")
```

### REST API 사용

```python
import requests

# 이미지 업로드 및 인식
with open('car_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/detect',
        files={'image': f}
    )
    result = response.json()
    print(f"인식 결과: {result['plate_number']}")
```

### CLI 사용

```bash
# 시스템 정보 확인
python main_app.py --help

# 설정 확인
python main_app.py config view

# 성능 벤치마크
python test_train.py benchmark --images_dir ./test_images
```

## 🔧 커스터마이징

### 1️⃣ 설정 수정

`config.yaml` 파일을 편집하여 시스템 동작 조정:

```yaml
model:
  confidence_threshold: 0.7  # 신뢰도 임계값 조정
  
realtime:
  frame_skip: 5             # 처리 프레임 간격 조정
  
performance:
  max_workers: 8            # 병렬 처리 워커 수 증가
```

### 2️⃣ 커스텀 모델 훈련

```bash
# 샘플 데이터셋 생성
python test_train.py create_dataset --output_dir ./custom_dataset --num_images 1000

# 모델 훈련
python test_train.py train --dataset_dir ./custom_dataset

# 훈련된 모델 사용
# config.yaml에서 yolo_model_path를 훈련된 모델 경로로 변경
```

### 3️⃣ 새로운 기능 추가

각 모듈은 독립적으로 설계되어 있어 쉽게 확장 가능:

- **새로운 전처리 방법**: `license_plate_recognizer.py`의 `preprocess_plate_region` 수정
- **다른 데이터베이스**: `database_manager.py`에 PostgreSQL/MySQL 지원 추가
- **새로운 API 엔드포인트**: `web_interface.py`에 라우트 추가
- **알림 기능**: `config_manager.py`의 notification 설정 활용

## 🚨 문제해결

### 자주 발생하는 문제

1. **Tesseract 경로 오류**
   ```bash
   # Windows의 경우 .env 파일에 추가:
   LP_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

2. **CUDA 메모리 부족**
   ```yaml
   # config.yaml에서 조정:
   performance:
     gpu_memory_fraction: 0.5
   ```

3. **포트 충돌**
   ```bash
   python main_app.py server --port 8000
   ```

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f logs/license_plate_system.log

# 또는 Makefile 사용
make logs
```

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 만듭니다: `git checkout -b feature/새기능`
3. 변경사항을 커밋합니다: `git commit -am '새 기능 추가'`
4. 브랜치에 푸시합니다: `git push origin feature/새기능`
5. Pull Request를 생성합니다

## 📞 지원

- **버그 리포트**: [GitHub Issues](https://github.com/your-username/yolov8-license-plate-recognition/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/your-username/yolov8-license-plate-recognition/discussions)
- **문서**: [Wiki](https://github.com/your-username/yolov8-license-plate-recognition/wiki)