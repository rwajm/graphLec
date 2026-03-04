# GraphLec

강의 영상을 입력으로 받아 전사 → 슬라이드 감지 → 텍스트 통합 → 지식그래프 생성까지 자동으로 처리하는 멀티모달 파이프라인입니다.

---

## 전체 파이프라인 구조

```
lecture.mp4  ──┐
lecture.pptx ──┤  (선택)
               ▼
        ┌──────────────┐
        │   main.py    │  전체 파이프라인 오케스트레이터
        └──────┬───────┘
               │
     ┌─────────▼──────────────────────────────────────────────────────┐
     │ Preprocessing                                                  │
     │   1. transcriber.py       Groq Whisper 음성 전사               │
     │   2. slide_detector.py    pHash 기반 슬라이드 감지 + 이미지 저장│
     │   3. make_merged.py       Gemini 전사 교정 + 슬라이드-전사 병합│
     ├─────────▼──────────────────────────────────────────────────────┤
     │ Stage 2  integrate_text.py  t1 + t2 → t3 + text_vector        │
     ├─────────▼──────────────────────────────────────────────────────┤
     │ Stage 3  multimodal_graph.py  t3 + 벡터 → 지식그래프           │
     └─────────▼──────────────────────────────────────────────────────┘
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
| `preprocessing/transcriber.py` | Groq Whisper로 영상 음성 전사 |
| `preprocessing/slide_detector.py` | pHash 기반 슬라이드 전환 감지 및 이미지 저장 |
| `preprocessing/make_merged_from_detector.py` | Gemini 전사 교정 + 슬라이드-전사 병합 JSON 생성 |
| `integrate_text.py` | 슬라이드(t1)와 오디오 전사(t2)를 타임스탬프 기준으로 통합(t3) + Gemini 임베딩 |
| `multimodal_graph.py` | t3 기반 개념/관계 추출 → 지식그래프 구축 및 시각화 |
| `graph_qa.py` | 지식그래프 기반 Q&A 시스템 |

---

## 입력 / 출력

### 입력

| 파일 | 설명 | 필수 여부 |
|------|------|-----------|
| `lecture.mp4` | 분석할 강의 영상 (mp4, avi 등) | 필수 |
| `lecture.pptx` | 강의 슬라이드 PPTX 파일 | 선택 (없으면 이미지에서 텍스트 추출) |

### 출력

모든 결과 파일은 `./output/` 폴더에 저장됩니다.

| 파일 | 생성 단계 | 설명 |
|------|-----------|------|
| `output/{stem}/slide_NNN_start.jpg` | Preprocessing | 감지된 슬라이드 시작 이미지 |
| `output/{stem}/slide_NNN_end.jpg` | Preprocessing | 감지된 슬라이드 종료 이미지 |
| `{stem}_transcribed.json` | Preprocessing | Groq Whisper 전사 결과 |
| `{stem}_merged.json` | Preprocessing | 교정된 전사 + 슬라이드 병합 JSON |
| `output/slide_extracted.json` | 포맷 변환 | t1 + 슬라이드 정보 |
| `output/slide_extracted_light.json` | 포맷 변환 | 벡터 제외 경량 버전 |
| `output/audio.json` | 포맷 변환 | 교정된 전사 세그먼트 |
| `output/integrated_text.json` | Stage 2 | t3 (통합 텍스트) + text_vector |
| `output/integrated_text_light.json` | Stage 2 | 벡터 제외 경량 버전 |
| `output/knowledge_graph.json` | Stage 3 | 개념 노드 + 관계 엣지 + 벡터 통합 그래프 |
| `output/knowledge_graph_light.json` | Stage 3 | 벡터 제외 경량 버전 |
| `output/knowledge_graph.html` | Stage 3 | PyVis 인터랙티브 그래프 시각화 |

---

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/your-repo/graphlec.git
cd graphlec
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

### 4. ffmpeg 설치

Preprocessing에서 오디오 추출 및 프레임 샘플링에 사용합니다.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 5. API 키 설정

```bash
# 환경변수로 설정 (권장)
export GOOGLE_API_KEY="your_google_api_key"
export GROQ_API_KEY="your_groq_api_key"

# 또는 .env 파일 생성
cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
EOF
```

- Google API 키: [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급
- Groq API 키: [Groq Console](https://console.groq.com)에서 발급

---

## 실행

### 전체 파이프라인 실행

```bash
python main.py --video lecture.mp4
```

### PPTX 파일이 있는 경우

```bash
python main.py --video lecture.mp4 --pptx lecture.pptx
```

### 주요 옵션

```bash
python main.py \
  --video lecture.mp4 \         # 입력 영상 (필수)
  --pptx lecture.pptx \         # PPTX 강의 자료 (선택)
  --output ./output             # 결과 저장 폴더 (기본값: ./output)
```

### 특정 스테이지만 실행

```bash
python main.py --video lecture.mp4 --only 2    # Stage 2만 실행 (텍스트 통합)
python main.py --video lecture.mp4 --only 3    # Stage 3만 실행 (지식그래프)
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

## pHash 슬라이드 감지 파라미터

`preprocessing/slide_detector.py`에서 조정 가능:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `SAMPLE_FPS` | 4 | 분석용 샘플링 FPS |
| `MIN_DURATION` | 1.5 | 최소 슬라이드 지속 시간 (초) |
| `TRANSITION_THR` | 8 | 슬라이드 전환 판정 해밍 거리 |
| `DUPLICATE_THR` | 6 | 중복 슬라이드 판정 해밍 거리 |

| 값 (TRANSITION_THR) | 적합한 상황 |
|---------------------|-------------|
| 6 | 슬라이드 변화가 미세한 경우, 세밀한 감지 |
| 8 | 일반 강의 (기본값) |
| 12 | 애니메이션·영상 전환이 많은 복잡한 슬라이드 |

---

## requirements.txt

```
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
imagehash>=4.3.0
google-generativeai>=0.8.0
google-genai>=0.8.0
groq>=0.4.0
pyvis>=0.3.2
python-dotenv>=1.0.0
```

선택 패키지:
```
python-pptx>=0.6.21    # PPTX 텍스트 추출 시 필요
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
