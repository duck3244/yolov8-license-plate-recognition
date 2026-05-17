# 아키텍처 (Architecture)

> YOLOv8 + OCR 기반 한국 차량 번호판 인식 시스템 (v3.0 — FastAPI + React SPA)

## 1. 개요

본 프로젝트는 업로드된 이미지에서 한국 차량 번호판을 검출(YOLOv8) 하고, 검출된 영역에서 텍스트를 OCR (Pororo / PaddleOCR / EasyOCR / Tesseract 중 자동 선택) 로 인식하여 결과를 SQLite 에 저장·조회하는 **단일 사용자 MVP** 웹 애플리케이션이다.

- 백엔드: **FastAPI** (Python 3.10, uvicorn)
- 추론 엔진: **Ultralytics YOLOv8** + 다중 OCR (auto-fallback)
- 프런트엔드: **React 18 + TypeScript + Vite + Tailwind CSS**
- 데이터 저장: **SQLite** (단일 파일, BLOB 으로 결과 이미지 보관)
- 배포 모드: 개발(Vite proxy) / 운영(FastAPI 가 SPA 정적 서빙)

---

## 2. 디렉토리 구조

```
yolov8-license-plate-recognition/
├── Makefile                       # dev / build / prod 통합 워크플로
├── README.md
├── backend/
│   ├── main.py                    # FastAPI 진입점 (uvicorn main:app)
│   ├── api/
│   │   ├── routes.py              # /api/detect, /history, /statistics, /health
│   │   ├── deps.py                # 싱글톤 의존성 (recognizer, db, inference_lock)
│   │   ├── schemas.py             # Pydantic v2 요청/응답 모델
│   │   └── settings.py            # pydantic-settings (.env / 환경변수)
│   ├── license_plate_recognizer.py  # YOLO + OCR 통합 엔진
│   ├── database_manager.py        # SQLite CRUD + PlateDetection dataclass
│   ├── config_manager.py          # YAML 기반 레거시 설정 (호환)
│   ├── batch_processor.py         # 배치 처리 유틸 (CLI)
│   ├── realtime_processor.py      # 실시간 처리 (실험)
│   ├── pororo/                    # Vendored Pororo OCR
│   ├── uploads/                   # 임시 업로드 (처리 후 자동 삭제)
│   ├── logs/                      # RotatingFileHandler 로그
│   ├── license_plates.db          # SQLite 데이터베이스
│   └── license_plate_det_yolov8.pt# YOLOv8 가중치
└── frontend/
    ├── index.html
    ├── vite.config.ts             # /api → http://127.0.0.1:8000 proxy
    └── src/
        ├── main.tsx
        ├── App.tsx                # 업로드 + 결과 + 이력 UI
        └── api/client.ts          # axios 기반 API 클라이언트
```

---

## 3. 레이어 구성

```
┌──────────────────────────────────────────────────────────────┐
│  Presentation                                                 │
│  React SPA (App.tsx)  ─  axios  ─  /api/*                    │
├──────────────────────────────────────────────────────────────┤
│  Transport / Web                                              │
│  FastAPI (main.py)                                            │
│    ├─ CORS Middleware (dev 한정: localhost:5173)              │
│    ├─ Lifespan: 모델 사전 로딩 + uploads 정리                  │
│    └─ StaticFiles mount (/) → frontend/dist (운영 only)       │
├──────────────────────────────────────────────────────────────┤
│  API                                                          │
│  api/routes.py                                                │
│    POST /api/detect        ── 업로드·검증·추론·저장             │
│    GET  /api/results/{id}/image                                │
│    GET  /api/history                                          │
│    GET  /api/statistics                                       │
│    GET  /api/health                                           │
├──────────────────────────────────────────────────────────────┤
│  Application Services                                         │
│  api/deps.py  (lru_cache 싱글톤)                              │
│    ├─ get_recognizer() → YOLOv8LicensePlateRecognizer         │
│    ├─ get_db()         → DatabaseManager                       │
│    └─ inference_lock   → threading.Lock (추론 직렬화)          │
├──────────────────────────────────────────────────────────────┤
│  Domain / Inference                                           │
│  license_plate_recognizer.py                                  │
│    ├─ detect_license_plates()   (YOLO → Advanced CV → Basic CV)│
│    ├─ recognize_text()          (Pororo/Paddle/Easy/Tesseract)│
│    ├─ preprocess_plate_region() (top-hat, blur, threshold...) │
│    ├─ is_valid_korean_plate()   (정규식 검증)                  │
│    └─ process_image()           (파이프라인 진입점)            │
├──────────────────────────────────────────────────────────────┤
│  Persistence                                                  │
│  database_manager.py                                          │
│    ├─ PlateDetection (dataclass)                              │
│    └─ DatabaseManager (sqlite3, per-call connection)          │
└──────────────────────────────────────────────────────────────┘
```

### 경계(Boundary) 원칙
- **`api/schemas.py`** 는 API 경계 모델 (Pydantic)
- **`database_manager.PlateDetection`** 은 도메인 모델 (dataclass)
- 두 모델을 의도적으로 분리하여, API 변경이 도메인에 침투하지 않도록 한다.

---

## 4. 런타임 컴포넌트

| 컴포넌트 | 책임 | 수명 |
|---|---|---|
| `FastAPI app` | HTTP 핸들링, 라우팅, CORS, SPA 서빙 | 프로세스 수명 |
| `Settings` | `.env` / 환경변수 통합 (pydantic-settings) | 싱글톤 (`lru_cache`) |
| `YOLOv8LicensePlateRecognizer` | 검출 + OCR + 후처리 | 싱글톤 (`lru_cache`) |
| `DatabaseManager` | SQLite 연결/CRUD/통계 | 싱글톤 (메서드 호출 시 conn open/close) |
| `inference_lock` | YOLO/OCR 동시 호출 차단 | 모듈 전역 `threading.Lock` |
| `uploads/` | 멀티파트 업로드 임시 저장 | 처리 후 즉시 삭제 + lifespan 시 24h 초과분 정리 |
| `logs/license_plate_system.log` | RotatingFileHandler (10MB × 5) | 항시 |

### 동시성 모델
- FastAPI 동기 엔드포인트는 **threadpool** 에서 실행됨 → 멀티스레드 동시 호출 가능.
- Ultralytics YOLO 와 Pororo brainocr 는 thread-safe 보장 없음 → `inference_lock` 으로 추론 구간만 직렬화.
- SQLite 연결은 호출마다 새로 열고 컨텍스트 매니저로 commit/rollback (스레드별 분리 권장 사항 준수).

---

## 5. 요청 처리 흐름 (`POST /api/detect`)

```
[Browser]
   │ multipart/form-data (image=...)
   ▼
[Vite dev proxy]  ──(prod 에서는 same-origin)──►  [FastAPI]
                                                      │
                                                      ▼
                                            api/routes.detect()
                                                      │
                  ┌───────────────────────────────────┤
                  │ 1) 파일명 / 확장자 / 크기 검증
                  │ 2) uuid 기반 안전 파일명 생성
                  │ 3) uploads/ 에 디스크 저장
                  ▼
              with inference_lock:
                  recognizer.process_image()
                      ├─ cv2.imread
                      ├─ detect_license_plates()    ── YOLO or CV fallback
                      ├─ for each plate:
                      │     plate_region = image[y1:y2, x1:x2]
                      │     recognize_text(plate_region)
                      │       └─ Pororo/Paddle/Easy/Tesseract
                      │     is_valid_korean_plate()
                      └─ best_plate 시각화 → result_image (np.ndarray)
                  │
                  ▼
            cv2.imencode(".jpg") → bytes
                  │
                  ▼
            db.save_detection(PlateDetection(... result_image=bytes))
                  │
                  ▼
            DetectResponse(success, detection_id, plate_number,
                           confidence, processing_time,
                           result_image_url="/api/results/{id}/image")
                  │
                  ▼  (finally) uploads/<file> 삭제
[Browser]  ◄── JSON
```

별도 GET `/api/results/{id}/image` 가 BLOB을 JPEG 로 스트리밍하여 SPA 에서 `<img src>` 로 표시한다.

---

## 6. 데이터 모델

### SQLite — `detections` 테이블
| 컬럼 | 타입 | 비고 |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `plate_number` | TEXT NOT NULL | idx |
| `confidence` | REAL NOT NULL | |
| `timestamp` | DATETIME NOT NULL | idx |
| `image_path` | TEXT | 사용 안 함 (현 버전) |
| `bbox_x1..y2` | INTEGER | 옵션 |
| `processing_time` | REAL | 초 |
| `original_filename` | TEXT | 사용자 원본 파일명(한글 가능) |
| `result_image` | BLOB | JPEG 인코딩 결과 이미지 |
| `created_at` | DATETIME DEFAULT CURRENT_TIMESTAMP | |

인덱스: `idx_plate_number`, `idx_timestamp`.
스키마 진화: 기존 DB 에 `original_filename`, `result_image` 가 없으면 `ALTER TABLE ... ADD COLUMN` 으로 idempotent 추가.

### Pydantic 응답 모델 (`api/schemas.py`)
- `DetectResponse`
- `DetectionItem`, `HistoryResponse`
- `Statistics`, `StatisticsResponse`
- `HealthResponse`

---

## 7. 설정 (Configuration)

`backend/api/settings.py` — pydantic-settings v2 로 `.env` 와 환경변수를 통합한다.

| 키 | 기본값 | 설명 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 외부 노출은 reverse proxy 권장 |
| `SECRET_KEY` | 무작위 hex (dev) | 운영 시 반드시 환경변수 주입 |
| `UPLOAD_FOLDER` | `backend/uploads` | |
| `MAX_UPLOAD_MB` | `16` | 초과 시 413 |
| `ALLOWED_EXTENSIONS` | `.jpg/.jpeg/.png/.bmp/.tiff` | |
| `CLEANUP_AGE_HOURS` | `24` | lifespan 시 오래된 업로드 삭제 |
| `DELETE_AFTER_PROCESS` | `true` | 처리 직후 업로드 삭제 |
| `DB_PATH` | `backend/license_plates.db` | |
| `YOLO_MODEL_PATH` | `backend/license_plate_det_yolov8.pt` | 없으면 HF Hub 자동 다운로드 |
| `CONFIDENCE_THRESHOLD` | `0.3` | YOLO 임계값 |
| `OCR_ENGINE` | `auto` | `pororo`/`paddleocr`/`easyocr`/`tesseract` |
| `CORS_ORIGINS` | `localhost:5173`, `127.0.0.1:5173` | Vite dev 한정 |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / `logs/...log` | RotatingFileHandler |

`backend/config.yaml` 은 레거시 호환용 (`config_manager.py`).

---

## 8. 검출/인식 파이프라인

1. **검출(Detection)** — `YOLOv8LicensePlateRecognizer.detect_license_plates`
   1. YOLOv8 (`yolo_model_path`) → 임계값 ≥ `confidence_threshold` 인 박스만 채택.
   2. YOLO 결과가 비면 **고급 OpenCV** (top-hat/black-hat → adaptive threshold → contour → 문자 그룹핑 → 회전 보정) fallback.
   3. 그래도 비면 **기본 OpenCV** (Canny + aspect-ratio 필터) fallback.
2. **OCR 엔진 선택** — `_setup_ocr_engine`
   - 우선순위: Pororo → PaddleOCR → EasyOCR → Tesseract (auto 모드 기본).
   - 첫 번째로 초기화 성공한 엔진을 선택, 실패는 모두 warning 로그.
3. **전처리(Preprocessing)** — Tesseract 는 다중 변형(`_generate_preprocess_variants`) 후 PSM 6/7/8 조합으로 score+빈도 기반 선택. 그 외 엔진은 단일 전처리.
4. **후처리(Postprocess)** — `clean_plate_text_advanced` 가 OCR 오자(O→0, I→1 등) 교정, `is_valid_korean_plate` 가 정규식(`\d{2,3}[가-힣]\d{4}` 등) 으로 검증.
5. **시각화** — 최선 후보 박스(녹색) + 한글 라벨(`draw_label_korean` via PIL+NanumGothic), 그 외 후보(파랑 얇은 선).

---

## 9. 보안 / 안전

- **경로 탈출 방지**: 업로드 저장 경로가 `upload_folder` 하위인지 `Path.resolve()` 로 검증.
- **확장자 화이트리스트**: `_validate_extension` 으로 사전 차단.
- **파일 크기 한도**: `max_upload_bytes` 초과 시 413 응답.
- **임시 파일 정리**: 처리 직후 즉시 삭제 (`DELETE_AFTER_PROCESS`) + lifespan 시 24h 초과분 청소.
- **추론 직렬화**: 모듈 전역 lock 으로 비-스레드세이프 모델 호출을 안전하게 직렬화.
- **CORS**: 기본은 Vite dev 만 허용, `allow_credentials=False`.
- **시크릿**: `.env` 는 커밋 금지, `.env.example` 로 키만 노출.

---

## 10. 배포 토폴로지

```
[Dev]
┌────────────┐   /api proxy   ┌──────────────┐
│ Vite :5173 │ ─────────────► │ FastAPI :8000│
└────────────┘                └──────────────┘
        ▲  HMR
        └── React SPA (개발 모드)

[Prod]
┌────────────────────────────┐
│  uvicorn (single worker)   │
│   ├─ /api/*  → FastAPI 라우터│
│   └─ /      → StaticFiles(frontend/dist) │
└────────────────────────────┘
   (필요 시 nginx / reverse proxy + HTTPS 가 앞단에 위치)
```

`make dev` 는 backend(reload) + frontend(vite) 동시 기동, `make prod` 는 `npm run build` 후 빌드된 SPA 를 FastAPI 가 정적 마운트로 서빙.

---

## 11. 확장 포인트

- **추가 OCR 엔진**: `license_plate_recognizer._setup_ocr_engine` 의 `init_map` 에 `(name, init_callable)` 추가.
- **인증/멀티 사용자**: 현재 단일 사용자 MVP. JWT 미들웨어 + `cors_origins`, `allow_credentials=True` 전환 시 토큰 기반 권장.
- **분산 스토리지**: 결과 이미지 BLOB 을 S3/MinIO 로 이전 시 `database_manager.save_detection` 와 `/api/results/{id}/image` 만 교체.
- **비동기 추론**: 현재는 threadpool + lock. 큐잉(redis + Celery / arq) 으로 분리 가능.
- **모델 교체**: HF Hub 자동 다운로드 (`_ensure_model_exists`) 가 있어 가중치 파일 경로만 바꾸면 됨.
