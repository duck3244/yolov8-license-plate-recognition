# 루트 Makefile — backend + frontend 통합 개발 워크플로

PY ?= /home/duck/miniconda3/envs/py310_pt/bin/python

.PHONY: help dev backend-dev frontend-dev build prod clean install

help:
	@echo "사용 가능한 타깃:"
	@echo "  make install        - backend(pip) + frontend(npm) 의존성 설치"
	@echo "  make dev            - backend(uvicorn) + frontend(vite) 동시 실행"
	@echo "  make backend-dev    - FastAPI 단독 (uvicorn --reload, port 8000)"
	@echo "  make frontend-dev   - Vite 단독 (port 5173, /api 프록시)"
	@echo "  make build          - frontend 운영 빌드 (frontend/dist)"
	@echo "  make prod           - 빌드된 SPA + uvicorn 단일 워커로 서빙"
	@echo "  make clean          - dist/, __pycache__, *.pyc 제거"

install:
	$(PY) -m pip install -r backend/requirements.txt
	cd frontend && npm install

backend-dev:
	cd backend && $(PY) -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 \
		--reload-dir api --reload-dir .

frontend-dev:
	cd frontend && npm run dev

# 두 프로세스를 한 터미널에서 동시 실행. Ctrl-C 한 번에 모두 종료.
dev:
	@trap 'kill 0' INT; \
	$(MAKE) backend-dev & \
	$(MAKE) frontend-dev & \
	wait

build:
	cd frontend && npm run build

# 운영: SPA 빌드 후 FastAPI 단일 워커. 빌드 산출물(frontend/dist)을 main.py가 자동 마운트.
prod: build
	cd backend && $(PY) -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1

clean:
	rm -rf frontend/dist
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete
