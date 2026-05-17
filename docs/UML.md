# UML 다이어그램

> 본 문서는 Mermaid 로 작성되었으며, GitHub / VS Code Markdown Preview / IntelliJ 등에서 바로 렌더링된다.

## 1. Use Case Diagram

```mermaid
flowchart LR
    user((사용자))

    subgraph System["License Plate Recognition System"]
        uc1(["이미지 업로드 & 번호판 인식"])
        uc2(["결과 이미지 조회"])
        uc3(["인식 이력 조회"])
        uc4(["통계 조회"])
        uc5(["헬스체크"])
    end

    user --- uc1
    user --- uc2
    user --- uc3
    user --- uc4
    user --- uc5

    yolo[["YOLOv8 모델"]]
    ocr[["OCR 엔진<br/>(Pororo/Paddle/Easy/Tesseract)"]]
    db[(SQLite)]
    hf[["HuggingFace Hub"]]

    uc1 -.uses.-> yolo
    uc1 -.uses.-> ocr
    uc1 -.persist.-> db
    uc2 -.read.-> db
    uc3 -.read.-> db
    uc4 -.read.-> db
    yolo -.auto download.-> hf
```

---

## 2. Component Diagram

```mermaid
flowchart TB
    subgraph Browser["Browser"]
        spa["React SPA<br/>App.tsx"]
        cli["api/client.ts<br/>(axios)"]
        spa --> cli
    end

    subgraph FE["Frontend (Vite dev / built dist)"]
        vite["Vite Dev Server :5173<br/>/api → :8000 proxy"]
        dist["frontend/dist (built)"]
    end

    subgraph BE["Backend (FastAPI / uvicorn :8000)"]
        main["main.py<br/>create_app / lifespan / CORS / StaticFiles"]
        routes["api/routes.py<br/>/api/detect, /history, /statistics, /health, /results/{id}/image"]
        deps["api/deps.py<br/>get_recognizer · get_db · inference_lock"]
        schemas["api/schemas.py (Pydantic)"]
        settings["api/settings.py (pydantic-settings)"]
        rec["YOLOv8LicensePlateRecognizer"]
        dbm["DatabaseManager"]
        uploads[("uploads/")]
        logs[("logs/")]
    end

    sqlite[(SQLite<br/>license_plates.db)]
    yolow[["YOLOv8 weights<br/>license_plate_det_yolov8.pt"]]
    ocrLibs[["Pororo / PaddleOCR<br/>EasyOCR / Tesseract"]]

    cli -- HTTPS/JSON --> vite
    vite -- proxy --> main
    cli -- prod same-origin --> main
    main --> routes
    main --> dist
    routes --> deps
    routes --> schemas
    routes --> uploads
    deps --> rec
    deps --> dbm
    deps --> settings
    rec --> yolow
    rec --> ocrLibs
    dbm --> sqlite
    main --> logs
```

---

## 3. Class Diagram

```mermaid
classDiagram
    direction LR

    class Settings {
      +str host
      +int port
      +str secret_key
      +str upload_folder
      +int max_upload_mb
      +tuple allowed_extensions
      +int cleanup_age_hours
      +bool delete_after_process
      +str db_path
      +str yolo_model_path
      +float confidence_threshold
      +str ocr_engine
      +tuple cors_origins
      +str log_level
      +str log_file
      +max_upload_bytes() int
    }

    class PlateDetection {
      +str plate_number
      +float confidence
      +datetime timestamp
      +Optional~str~ image_path
      +Optional~tuple~ bbox
      +Optional~float~ processing_time
      +Optional~str~ original_filename
      +Optional~bytes~ result_image
    }

    class DatabaseManager {
      -str db_path
      +__init__(db_path)
      +init_database()
      -_connect() Connection
      +save_detection(d: PlateDetection) Optional~int~
      +get_result_image(id: int) Optional~bytes~
      +get_detections(plate_number, start_date, end_date, limit) List~Dict~
      +get_statistics() Dict
      +ping() bool
    }

    class YOLOv8LicensePlateRecognizer {
      -YOLO yolo_model
      -str ocr_engine
      -float confidence_threshold
      -bool use_advanced_preprocessing
      -dict char_detection_params
      -list korean_plate_patterns
      +__init__(yolo_model_path, tesseract_cmd, confidence_threshold, ocr_engine, use_advanced_preprocessing)
      -_ensure_model_exists(path) str
      -_init_pororo() bool
      -_init_paddleocr() bool
      -_init_easyocr() bool
      -_setup_ocr_engine(preferred) str
      +detect_license_plates(image) List
      -_yolo_detect(image) List
      -_advanced_opencv_detect(image) List
      -_opencv_detect(image) List
      +preprocess_plate_region(img) ndarray
      +preprocess_plate_region_advanced(img) ndarray
      +recognize_text(plate_img) str
      +clean_plate_text_advanced(text) str
      +is_valid_korean_plate(text) bool
      +process_image(image_path, save_result) Tuple~str, ndarray~
    }

    class FastAPIApp {
      +create_app() FastAPI
      +lifespan(app) AsyncContextManager
      -_configure_logging()
    }

    class Router {
      +detect(image, recognizer, db) DetectResponse
      +get_result_image(detection_id, db) Response
      +get_history(plate_number, limit, db) HistoryResponse
      +get_statistics(db) StatisticsResponse
      +health(db) HealthResponse
    }

    class Deps {
      +threading.Lock inference_lock
      +get_recognizer() YOLOv8LicensePlateRecognizer
      +get_db() DatabaseManager
    }

    class DetectResponse {
      +bool success
      +Optional~int~ detection_id
      +Optional~str~ plate_number
      +Optional~float~ confidence
      +float processing_time
      +Optional~str~ result_image_url
      +Optional~str~ error
    }

    class DetectionItem {
      +int id
      +str plate_number
      +float confidence
      +datetime timestamp
      +Optional~str~ original_filename
      +Optional~float~ processing_time
      +Optional~str~ result_image_url
    }

    class HistoryResponse {
      +bool success
      +int count
      +List~DetectionItem~ detections
    }

    class Statistics {
      +int total_detections
      +int unique_plates
      +int today_detections
      +float avg_confidence
      +float avg_processing_time
      +str last_updated
    }

    class StatisticsResponse {
      +bool success
      +Statistics statistics
    }

    class HealthResponse {
      +str status
      +str database
      +datetime timestamp
    }

    FastAPIApp --> Router : include_router
    FastAPIApp --> Settings : get_settings()
    Router --> Deps : Depends
    Router --> DetectResponse : returns
    Router --> HistoryResponse : returns
    Router --> StatisticsResponse : returns
    Router --> HealthResponse : returns
    Router --> PlateDetection : creates
    Deps --> YOLOv8LicensePlateRecognizer : lru_cache singleton
    Deps --> DatabaseManager : lru_cache singleton
    Deps --> Settings : reads
    DatabaseManager --> PlateDetection : persists
    YOLOv8LicensePlateRecognizer ..> PlateDetection : produces text/image
    HistoryResponse "1" *-- "0..*" DetectionItem
    StatisticsResponse "1" *-- "1" Statistics
```

---

## 4. Sequence Diagram — `POST /api/detect`

```mermaid
sequenceDiagram
    autonumber
    actor U as User (Browser)
    participant FE as React SPA (App.tsx)
    participant AX as axios (client.ts)
    participant FA as FastAPI Router
    participant DP as deps (lock + singletons)
    participant RC as YOLOv8LicensePlateRecognizer
    participant CV as OpenCV / YOLO / OCR
    participant DB as DatabaseManager

    U->>FE: 파일 선택 + "번호판 인식" 클릭
    FE->>AX: uploadImage(file)
    AX->>FA: POST /api/detect (multipart)
    FA->>FA: 확장자/크기 검증, uuid 파일명 생성
    FA->>FA: uploads/<uuid>.jpg 디스크 저장
    FA->>DP: with inference_lock
    DP-->>FA: lock acquired
    FA->>RC: process_image(saved_path)
    RC->>CV: cv2.imread → detect_license_plates
    CV-->>RC: bbox list
    loop each plate
        RC->>CV: recognize_text(region)
        CV-->>RC: plate_text
        RC->>RC: is_valid_korean_plate?
    end
    RC->>RC: 시각화 (best bbox + 한글 라벨)
    RC-->>FA: (best_text, result_image)
    FA->>FA: cv2.imencode(".jpg") → bytes
    FA->>DB: save_detection(PlateDetection(...))
    DB-->>FA: detection_id
    FA->>FA: uploads/<file> 삭제 (delete_after_process)
    FA-->>AX: DetectResponse JSON
    AX-->>FE: result
    FE->>FE: 결과 표시 + history 갱신
    FE->>AX: fetchHistory(10)
    AX->>FA: GET /api/history?limit=10
    FA->>DB: get_detections(limit=10)
    DB-->>FA: rows
    FA-->>AX: HistoryResponse
    AX-->>FE: list
```

---

## 5. Sequence Diagram — `GET /api/results/{id}/image`

```mermaid
sequenceDiagram
    autonumber
    actor U as Browser
    participant FA as FastAPI Router
    participant DB as DatabaseManager
    participant S as SQLite

    U->>FA: GET /api/results/{id}/image
    FA->>DB: get_result_image(id)
    DB->>S: SELECT result_image FROM detections WHERE id=?
    S-->>DB: BLOB or NULL
    DB-->>FA: bytes or None
    alt None
        FA-->>U: 404 결과 이미지 없음
    else bytes
        FA-->>U: 200 image/jpeg
    end
```

---

## 6. Activity Diagram — `process_image` 파이프라인

```mermaid
flowchart TD
    A(["process_image 호출"]) --> B["cv2.imread"]
    B --> C{"이미지 로드 성공?"}
    C -- no --> Z1(["빈 문자열 + 원본 이미지 반환"])
    C -- yes --> D["detect_license_plates"]
    D --> D1{"YOLO 사용 가능?"}
    D1 -- yes --> D2["YOLO 추론"]
    D2 --> D3{"결과 있음?"}
    D3 -- no --> D4["고급 OpenCV"]
    D1 -- no --> D4
    D4 --> D5{"결과 있음?"}
    D5 -- no --> D6["기본 OpenCV"]
    D5 -- yes --> E
    D3 -- yes --> E
    D6 --> E["bbox 리스트"]
    E --> F{"plates 비었음?"}
    F -- yes --> Z2(["빈 문자열 + 원본 이미지 반환"])
    F -- no --> G["for each plate"]
    G --> H["plate_region = image crop"]
    H --> I["recognize_text"]
    I --> J{"유효한 한국 번호판?"}
    J -- yes --> K{"confidence가 best보다 큰가?"}
    J -- no --> L{"best 비어있음?"}
    K -- yes --> M["best 업데이트"]
    K -- no --> N["다음 plate"]
    L -- yes --> M
    L -- no --> N
    M --> N
    N --> O{"모든 plate 처리?"}
    O -- no --> G
    O -- yes --> P["best bbox 시각화"]
    P --> Q["나머지 bbox는 얇은 선으로 표시"]
    Q --> R{"save_result?"}
    R -- yes --> S["result_*.jpg 저장"]
    R -- no --> T
    S --> T(["best_plate_text + result_image 반환"])
```

---

## 7. State Diagram — Frontend 업로드 화면

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> FileSelected : onFileChange (preview URL 생성)
    FileSelected --> Idle : 파일 해제
    FileSelected --> Busy : onSubmit (uploadImage)
    Busy --> Success : DetectResponse(success=true)
    Busy --> NoPlate : DetectResponse(success=false)
    Busy --> Error : axios reject
    Success --> FileSelected : 새 파일 선택
    NoPlate --> FileSelected : 새 파일 선택
    Error --> FileSelected : 새 파일 선택
    Success --> Success : history reload (fire-and-forget)
```

---

## 8. ER Diagram — SQLite `detections`

```mermaid
erDiagram
    DETECTIONS {
        INTEGER id PK
        TEXT plate_number "indexed"
        REAL confidence
        DATETIME timestamp "indexed"
        TEXT image_path
        INTEGER bbox_x1
        INTEGER bbox_y1
        INTEGER bbox_x2
        INTEGER bbox_y2
        REAL processing_time
        TEXT original_filename
        BLOB result_image
        DATETIME created_at "DEFAULT CURRENT_TIMESTAMP"
    }
```

> 단일 테이블 — 사용자/세션/이미지 메타 정규화는 MVP 범위 밖. 결과 이미지는 BLOB 으로 동일 행에 저장되어 단일 사용자 규모에서 분산 스토리지의 복잡도를 회피한다.

---

## 9. Deployment Diagram

```mermaid
flowchart LR
    subgraph dev["Development"]
        b1["Browser :5173"]
        v1["Vite Dev<br/>HMR + /api proxy"]
        u1["uvicorn --reload :8000"]
        b1 --> v1 --> u1
    end

    subgraph prod["Production (single host)"]
        b2["Browser"]
        rp["(Optional) nginx<br/>TLS · gzip"]
        u2["uvicorn (workers=1)<br/>main:app<br/>/api/* + StaticFiles(frontend/dist)"]
        fs[("uploads/ · logs/ · *.db · *.pt")]
        b2 --> rp --> u2 --> fs
    end
```

> 운영은 단일 워커 + threadpool + `inference_lock` 으로 모델 호출을 직렬화하는 단일 호스트 구성. 수평 확장이 필요하면 모델 호출을 큐(Redis + arq/Celery)로 분리하고 DB 를 PostgreSQL 로 마이그레이션하는 것이 다음 단계.
