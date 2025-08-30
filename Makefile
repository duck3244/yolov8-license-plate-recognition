# Makefile
# 개발 및 배포를 위한 Makefile

.PHONY: install install-dev test clean docker-build docker-run lint format

# 기본 설치
install:
	pip install -r requirements.txt

# 개발 환경 설치
install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# 테스트 실행
test:
	python -m pytest tests/ -v --cov=. --cov-report=html

# 코드 정리
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/

# 코드 스타일 검사
lint:
	flake8 *.py
	black --check *.py

# 코드 포맷팅
format:
	black *.py
	isort *.py

# Docker 이미지 빌드
docker-build:
	docker build -t yolov8-license-plate:latest .

# Docker 실행
docker-run:
	docker-compose up -d

# Docker 중지
docker-stop:
	docker-compose down

# 로그 확인
logs:
	tail -f logs/license_plate_system.log

# 데이터베이스 백업
backup:
	python -c "from database_manager import DatabaseManager; db = DatabaseManager(); db.backup_database('backup_$(shell date +%Y%m%d_%H%M%S).db')"

# 성능 테스트 (테스트 이미지가 있는 경우)
benchmark:
	python test_train.py benchmark --images_dir test_images --iterations 10

# 개발 서버 실행
dev:
	python main_app.py server --config config.yaml

# 프로덕션 서버 실행
prod:
	gunicorn -w 4 -b 0.0.0.0:5000 "main_app:create_app()"

# 설정 검증
validate-config:
	python main_app.py config validate

# 도움말 표시
help:
	@echo "사용 가능한 명령들:"
	@echo "  install      - 기본 패키지 설치"
	@echo "  install-dev  - 개발 환경 설치"
	@echo "  test         - 테스트 실행"
	@echo "  clean        - 임시 파일 정리"
	@echo "  lint         - 코드 스타일 검사"
	@echo "  format       - 코드 포맷팅"
	@echo "  docker-build - Docker 이미지 빌드"
	@echo "  docker-run   - Docker 컨테이너 실행"
	@echo "  logs         - 로그 확인"
	@echo "  backup       - 데이터베이스 백업"
	@echo "  benchmark    - 성능 벤치마크"
	@echo "  dev          - 개발 서버 실행"
	@echo "  prod         - 프로덕션 서버 실행"