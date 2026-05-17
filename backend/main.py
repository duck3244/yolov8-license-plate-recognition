"""
main.py
FastAPI 진입점.

실행:
    uvicorn main:app --host 127.0.0.1 --port 8000              # 개발
    uvicorn main:app --workers 1 --port 8000 --reload           # 개발 + 리로드
    uvicorn main:app --workers 1 --port 8000                    # 운영(단일 워커 + threadpool)

운영에서는 빌드된 frontend/dist 를 정적 서빙한다.
"""
from __future__ import annotations

import logging
import logging.handlers
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import deps
from api.routes import router
from api.settings import get_settings


def _configure_logging() -> None:
    """RotatingFileHandler + 콘솔 핸들러. uvicorn 로거에도 share."""
    settings = get_settings()
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_mb * 1024 * 1024,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # uvicorn 자체 로거(uvicorn, uvicorn.error, uvicorn.access)를 같은 핸들러로 연결.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [file_handler, console]
        lg.propagate = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """앱 시작 시 모델·DB를 한 번 로드해두면 첫 요청 지연을 없앨 수 있다."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 모델 사전 로딩 시작")
    _ = deps.get_recognizer()
    _ = deps.get_db()
    logger.info("✅ 사전 로딩 완료")

    # 오래된 업로드 정리
    settings = get_settings()
    upload_dir = Path(settings.upload_folder)
    if upload_dir.exists():
        import time
        threshold = time.time() - settings.cleanup_age_hours * 3600
        removed = 0
        for p in upload_dir.iterdir():
            if p.is_file() and p.stat().st_mtime < threshold:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.info(f"오래된 업로드 {removed}개 정리")

    yield


def create_app() -> FastAPI:
    _configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="YOLOv8 License Plate Recognition",
        version="3.0.0",
        lifespan=lifespan,
    )

    # CORS — dev에서 Vite(:5173) 와 분리된 origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,  # 토큰 기반이면 False로 두는 게 안전
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API 라우터를 정적 마운트보다 먼저 등록 (StaticFiles 가 catch-all)
    app.include_router(router)

    # 운영 빌드된 SPA가 있으면 정적 서빙
    spa_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if spa_dist.is_dir():
        app.mount("/", StaticFiles(directory=spa_dist, html=True), name="spa")

    return app


app = create_app()
