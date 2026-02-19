# graphLec

# GraphBrief / EduCurator

강의 영상을 입력으로 받아 슬라이드 이미지 추출 → 텍스트/벡터 추출 → 오디오 통합 → 지식그래프 생성까지 자동으로 처리하는 멀티모달 파이프라인입니다.

---

## 전체 파이프라인 구조

```
lecture.mp4  ──┐
audio.json   ──┤
               ▼
        ┌──────────────┐
        │   main.py    │  전체 파이프라인 오케스트레이터
        └──────┬───────┘
               │
     ┌─────────▼──────────────────────────────────────────────────┐
     │ Stage 0  main_0.py        영상 → 슬라이드 이미지 (MSE 감지) │
     ├─────────▼──────────────────────────────────────────────────┤
     │ Stage 1  video_extract.py  슬라이드 → t1 텍스트 + 이미지벡터│
     ├─────────▼──────────────────────────────────────────────────┤
     │ Stage 2  integrate_text.py t1 + 오디오(t2) → t3 + 텍스트벡터│
     ├─────────▼──────────────────────────────────────────────────┤
     │ Stage 3  multimodal_graph.py t3 + 벡터 → 지식그래프         │
     └─────────▼──────────────────────────────────────────────────┘
               │
        ┌──────▼───────┐
        │  graph_qa.py │  완성된 그래프 기반 Q&A (별도 실행)
        └──────────────┘
```

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `main.py` | 전체 파이프라인 순차 실행 진입점 |
| `config.py` | 모든 스테이지의 공통 설정값 관리 |
| `main_0.py` | MSE 기반 슬라이드 변화 감지 및 이미지 저장 |
| `video_extract.py` | Gemini Vision으로 슬라이드 텍스트(t1) 추출 + ColPali 이미지 벡터화 |
| `integrate_text.py` | 슬라이드(t1)와 오디오 전사(t2)를 타임스탬프 기준으로 통합(t3) + Gemini 임베딩 |
| `multimodal_graph.py` | t3 기반 개념/관계 추출 → 지식그래프 구축 및 시각화 |
| `graph_qa.py` | 지식그래프 기반 Q&A 시스템 |

---

## 입력 / 출력

### 입력

| 파일 | 설명 | 필수 여부 |
|------|------|-----------|
| `lecture.mp4` | 분석할 강의 영상 (mp4, avi 등 OpenCV 지원 형식) | Stage 0 필수 |
| `audio.json` | Whisper 등으로 생성한 오디오 전사 JSON | Stage 2 필수 |

**audio.json 형식 (두 가지 모두 지원)**

```json
// 형식 1 - segments 배열 포함
{
  "segments": [
    { "start": 0.0, "end": 5.2, "text": "안녕하세요" },
    { "start": 5.2, "end": 10.4, "text": "오늘은 운영체제에 대해 배우겠습니다" }
  ]
}

// 형식 2 - 최상위 배열
[
  { "start": 0.0, "end": 5.2, "text": "안녕하세요" }
]
```

### 출력

모든 결과 파일은 `./output/` 폴더에 저장됩니다.

| 파일 | 생성 스테이지 | 설명 |
|------|-------------|------|
| `output/slide_NNN_Xs.jpg` | Stage 0 | 감지된 슬라이드 이미지 |
| `output/report.txt` | Stage 0 | 슬라이드 감지 리포트 (타임스탬프 목록) |
| `output/slide_extracted.json` | Stage 1 | t1(슬라이드 텍스트) + image_vector(ColPali) |
| `output/slide_extracted_light.json` | Stage 1 | 벡터 제외 경량 버전 |
| `output/integrated_text.json` | Stage 2 | t3(통합 텍스트) + text_vector(Gemini 임베딩) |
| `output/integrated_text_light.json` | Stage 2 | 벡터 제외 경량 버전 |
| `output/knowledge_graph.json` | Stage 3 | 개념 노드 + 관계 엣지 + 벡터 통합 그래프 |
| `output/knowledge_graph_light.json` | Stage 3 | 벡터 제외 경량 버전 |
| `output/knowledge_graph.html` | Stage 3 | PyVis 인터랙티브 그래프 시각화 |

---

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/your-repo/graphbrief.git
cd graphbrief
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

GPU 사용 시 PyTorch를 CUDA 버전에 맞게 먼저 설치하세요:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. API 키 설정

```bash
# 환경변수로 설정 (권장)
export GOOGLE_API_KEY="your_api_key_here"

# 또는 .env 파일 생성
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Google API 키는 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받을 수 있습니다.

---

## 실행

### 전체 파이프라인 실행

```bash
python main.py --video lecture.mp4 --audio audio.json
```

### 주요 옵션

```bash
python main.py \
  --video lecture.mp4 \       # 입력 영상
  --audio audio.json \        # 오디오 전사 JSON
  --output ./output \         # 결과 저장 폴더 (기본값: ./output)
  --threshold 500             # MSE 감지 임계값 (기본값: 500, 범위: 500~2000)
```

### 슬라이드 이미지가 이미 있는 경우 (Stage 0 건너뜀)

```bash
python main.py --audio audio.json --skip-stage0
```

### 특정 스테이지만 실행

```bash
python main.py --only 1    # Stage 1만 실행
python main.py --only 3    # Stage 3만 실행
```

### Q&A 시스템 실행

```bash
python graph_qa.py -g ./output/knowledge_graph.json
```

Q&A 시스템 내 명령어:
- 자연어 질문 입력: 그래프 기반 답변
- `/explain <개념>`: 특정 개념 설명
- `/rel <개념>`: 개념의 관계 시각화
- `/q`: 종료

---

## MSE 임계값 가이드

| 값 | 적합한 상황 |
|----|-----------|
| 500 | 색상 변화가 적은 심플한 PPT, 세밀한 감지 필요 시 |
| 1000 | 일반 강의 (기본값) |
| 2000 | 애니메이션·영상 전환이 많은 복잡한 슬라이드 |

---

## requirements.txt

```
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
google-generativeai>=0.8.0
google-genai>=0.8.0
torch>=2.0.0
colpali-engine>=0.3.0
transformers>=4.40.0
pyvis>=0.3.2
python-dotenv>=1.0.0
```

---

## 지식그래프 관계 타입

Stage 3에서 추출되는 12가지 개념 간 관계:

| 관계 | 의미 |
|------|------|
| `is_a` | A는 B의 한 종류 |
| `part_of` | A는 B의 구성요소 |
| `implements` | A는 B를 구현 |
| `abstracts` | A는 B들을 추상화 |
| `prerequisite_of` | A를 알아야 B 이해 가능 |
| `uses` | A는 B를 사용 |
| `calls` | A가 B를 호출 |
| `compared_to` | A와 B 비교 |
| `extends` | A가 B를 확장 |
| `replaces` | A가 B를 대체 |
| `solves` | A가 B(문제)를 해결 |
| `optimizes` | A가 B를 최적화 |