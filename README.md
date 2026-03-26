# GraphLec: 통합 강의 분석 및 지식그래프 생성 파이프라인

GraphLec은 강의 영상에서 시각적 정보(슬라이드)와 청각적 정보(음성)를 결합하여, 고정밀 전사, 강조 구간 분석, **지식 그래프(Parquet)**·**벡터 검색(LanceDB)**·**웹 질의(Django + FastAPI)**까지 이어지는 멀티모달 파이프라인입니다.

---

## 1. 주요 기능

### 영상 및 슬라이드 분석

- 슬라이드 변화 감지: MSE(Mean Squared Error) 기반으로 영상 내 슬라이드 전환 시점을 추적하여 이미지를 추출합니다.
- 시각 텍스트 추출: Gemini Vision을 사용하여 각 슬라이드의 텍스트와 관련 정보를 추출합니다.

### 음성 전사 및 강조 분석

- 슬라이드별 맞춤 전사: Groq Whisper(whisper-large-v3-turbo) 등을 사용하여 슬라이드 구간에 맞춰 오디오를 잘라 전사합니다.
- 2단계 텍스트 교정: 전문용어·오타 교정 후, 슬라이드 컨텍스트를 반영해 자연스러운 문장으로 정리합니다.
- 강조 구간 탐지: 오디오 통계·키워드·LLM을 결합해 핵심 구간을 탐지합니다.

### 지식그래프·검색·Q&A

- 멀티모달 통합: 슬라이드 텍스트와 오디오 전사를 타임스탬프 기준으로 병합(`fused.json`)합니다.
- 지식 그래프: 개념·관계 트리플을 수집해 `{stem}_graph_triples.parquet`, `{stem}_nodes.parquet`, `{stem}_edges.parquet`로 저장합니다.
- 벡터 검색: 청크 임베딩 후 LanceDB(`data/lancedb` 등)에 적재하여 stem 단위 검색합니다.
- 질의응답: FastAPI 질의 서비스 + Django 웹 UI에서 자연어 질의에 답합니다.

---

## 2. 파이프라인 구조

`main.py` 기준 단계 요약입니다.

```text
강의 영상 (.mp4)
    ├─ [병렬] Stage 1A 슬라이드 추출 · Stage 1B 오디오 품질
    ├─ Stage 2 슬라이드 텍스트/강조
    ├─ [병렬] Stage 3A 필기 강조 · Stage 3B 오디오 전사·교정·강조
    ├─ [병렬] Stage 4A 슬라이드 분류 · Stage 4B by_slide 저장
    ├─ Stage 5 퓨전 → {stem}_fused.json
    ├─ Stage 6 그래프 트리플 → Parquet (triples / nodes / edges)
    └─ Stage 7 Lance 인덱스 → Parquet 백업 + LanceDB

질의: LanceDB 검색 + Gemini 답변 (query_service) ← Django 웹이 프록시
```

---

## 3. 설치 및 설정

### 시스템

- **ffmpeg** / **ffprobe** (영상·오디오 처리). `requirements.txt`로 설치되지 않으므로 OS 패키지로 설치합니다.

### 패키지 설치

```bash
pip install -r requirements.txt
```

GPU용 PyTorch가 필요하면 `requirements.txt` 상단 주석의 CUDA 인덱스 URL을 참고합니다.

### 환경 변수 (프로젝트 루트 `.env`)

파이프라인(`config.py`)에서 필요한 예:

- `GROQ_API_KEY`: Whisper 전사
- `GOOGLE_API_KEY_1` 또는 `GOOGLE_API_KEY`: Gemini(비디오/슬라이드 등)
- `GOOGLE_API_KEY_2`: 오디오 파이프라인·그래프(Stage 6) 등(미설정 시 1번 키로 대체 가능)

질의 서비스(`query_service`)는 `GOOGLE_API_KEY_2`·`GOOGLE_API_KEY`·`GEMINI_API_KEY` 등으로 Gemini를 찾습니다. LanceDB 경로는 `GRAPHLEC_LANCE_ROOT`(미설정 시 저장소 루트의 `data/lancedb`).

---

## 4. 사용법

### 전체 실행

```bash
python main.py --input input/lecture.mp4
python main.py --input input/lecture.mp4 --output output --slides output_slides
```

### 주요 옵션

- `--output`, `--slides`: 산출물·슬라이드 디렉터리
- `--skip-extract`: 이미 슬라이드가 있을 때 Stage 1A 추출 생략
- `--force`: 기존 출력이 있어도 강제 재실행
- `--skip-graph-triples`: Stage 6(그래프 Parquet) 생략
- `--skip-lance-index`: Stage 7(Lance 인덱스) 생략
- `--lance-root`: LanceDB 경로(기본: 환경변수 또는 `data/lancedb`)
- `--debug`, `--masks`: Stage 1 디버그 / Stage 3A 마스크 저장

---

## 5. 결과물 파일 안내 (`output/` 등)

| 구분 | 예시 | 설명 |
|------|------|------|
| 퓨전 | `{stem}_fused.json` | 슬라이드·세그먼트 통합 본문 |
| 그래프 | `{stem}_graph_triples.parquet`, `{stem}_nodes.parquet`, `{stem}_edges.parquet` | 트리플·정규화 노드/엣지 |
| 검색 | `{stem}_chunks_lance.parquet`, `data/lancedb/` | 청크 백업·LanceDB 테이블 `chunks`(stem 컬럼) |
| 기타 | `{stem}_segments.json`, `{stem}_by_slide.json`, … | 전사·분류·강조 등 (`config.output_paths` 참고) |
| 슬라이드 이미지 | `output_slides/` 등 | 추출 프레임·메타데이터 |

---

## 6. 요구 사항 (`requirements.txt` 요약)

opencv-python, numpy, Pillow, google-generativeai, google-genai, groq, torch, transformers, librosa, lancedb, pyarrow, pandas, fastapi, uvicorn, django, httpx 등 (전체 목록은 파일 참고).

---

## 7. 웹·질의 서비스 (Django + FastAPI)

브라우저는 **Django만** 호출하고, Django가 **FastAPI** 질의 서비스로 프록시합니다.

### 구성

- **Django** (`web/`): 강의 메타(`Lecture`), 질의 페이지, `/api/query/`·`/api/graph/full/` 등
- **FastAPI** (`query_service/`): `POST /internal/query` — `stem` + `question`(LanceDB 검색 + Gemini 답변)

### 실행 (터미널 2개, 프로젝트 루트에서 가상환경 활성화 후)

**1) FastAPI 질의 서비스 (포트 8001)**

```bash
uvicorn query_service.main:app --host 127.0.0.1 --port 8001
```

**2) Django (포트 8000)**

```bash
cd web
python manage.py migrate
python manage.py runserver 8000
```

브라우저: `http://127.0.0.1:8000/`  
`main.py` 실행이 완료되면 해당 영상의 `stem`이 자동으로 `Lecture`에 등록됩니다.

### 환경 변수 (웹·질의 연동)

- `QUERY_SERVICE_URL` (기본 `http://127.0.0.1:8001`)
- `GRAPHLEC_LANCE_ROOT` — LanceDB 디렉터리 (미설정 시 프로젝트 루트 `data/lancedb`)
- `GRAPHLEC_OUTPUT_DIR` — 전체 그래프 Parquet 위치 (미설정 시 프로젝트 루트 `output/`)
- `GOOGLE_API_KEY` 등 — Gemini(질의 서비스·파이프라인과 공통 키 이름 사용 가능)

질의 서비스는 **프로젝트 루트**에서 실행하는 것을 권장합니다(`python main.py`로 만든 LanceDB 경로와 맞추기 쉬움).  
Django는 로컬 테스트 기준으로 SQLite(`web/db.sqlite3`)를 사용합니다.

---

## 8. 데이터베이스

로컬 테스트는 SQLite 단일 DB(`web/db.sqlite3`)로 동작합니다.
