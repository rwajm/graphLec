## 강의 영상 전사·강조 분석 파이프라인

### 1. 기능 개요

- **강의 영상 전사**
  - Groq Whisper `whisper-large-v3-turbo`로 한국어 강의 음성을 전사합니다.
  - `metadata.json`의 슬라이드 구간에 맞춰 **슬라이드별로 오디오를 잘라서 전사**합니다.

- **텍스트 교정 (2단계)**
  1. 전문용어/오타 위주 최소 교정 → `text_corrected`
  2. 추임새 제거 + 자연스러운 문장화 → `text_natural`
  - 이때 `slide_textualized.json`의 슬라이드 텍스트·요약을 LLM 프롬프트에 넣어 **전문용어·맥락을 반영한 더 정확한 교정**을 수행합니다.

- **컨텍스트 그룹핑 + 강조 구간 감지**
  - 세그먼트를 문맥 단위 그룹으로 묶고(슬라이드별/맥락별),
  - **오디오 표준편차 + 가중치 키워드 + 주제 키워드 반복**으로 강조 구간을 탐지합니다.
  - 주제 키워드는 전사 전체 빈도 + LLM 필터로 추출하며, 최대 20개를 사용합니다.

- **결과물**
  - 오디오 품질 분석 JSON
  - 강조 구간 JSON (`*_emphasis_std_topic_v2.json`)
  - 강조에 사용된 주제 키워드 JSON (`*_emphasis_keywords_v2.json`)
  - 슬라이드-컨텍스트-세그먼트 구조 JSON (`*_by_slide_v2.json`, `*_by_slide_iterative_v2.json`)
  - 강의 노트 (`*_notes_v2.md`)

---

### 2. 폴더/파일 구조

- **루트**
  - `main.py` — 최종 파이프라인 진입점
  - `audio_analyzer.py` — 오디오 품질 분석
  - `transcriber.py` — Groq Whisper 전사
  - `text_processor.py` — 텍스트 2단계 교정 + 노트 생성
  - `segment_grouper.py` — 세그먼트 그룹핑(맥락/슬라이드별)
  - `emphasis_audio.py` — 오디오 신호 기반 강조 감지 (볼륨/피치 표준편차)
  - `emphasis_keyword.py` — 키워드 기반 강조 감지 (가중치 키워드 + 주제 키워드 반복)
  - `emphasis_combiner.py` — 강조 감지 결과 통합
  - `config.py`, `utils.py` — 공용 모듈
  - `output/` — 결과 파일 출력 폴더
  - `archive/` — 예전/미사용 스크립트 보관

- **입력용 폴더**
  - `src/`
    - 강의 영상 (`*.mp4`)
    - `metadata.json` 또는 `<video_stem>_metadata.json`
    - `slide_textualized.json` 또는 `<video_stem>_slide_textualized.json`

---

### 3. 설치

```bash
pip install -r requirements.txt
```

`config.py`에서 사용하는 환경변수:

- `GROQ_API_KEY`
- `GOOGLE_API_KEY` (또는 `GEMINI_API_KEY`)

`.env` 파일 등에 설정해 두면 자동으로 로드됩니다.

---

### 4. 사용법

루트 폴더에서:

```bash
python main.py <video.mp4> [metadata.json]
```

- **`<video.mp4>`**
  - `src/<video.mp4>`에 있으면 자동으로 찾습니다.
- **`metadata.json` (선택)**:
  - 인자로 주지 않아도, 다음 순서로 자동 탐색합니다.
    - `src/<video_stem>_metadata.json`
    - `src/metadata.json`
    - 루트 `metadata.json`
- **`slide_textualized.json` (선택)**:
  - 다음 순서로 자동 탐색합니다.
    - `src/<video_stem>_slide_textualized.json`
    - `src/slide_textualized.json`
    - `output/<video_stem>_slide_textualized.json`

---

### 5. 주요 출력 파일 (예: `1장_영상.mp4`)

`output/` 폴더에 생성됩니다.

- `1장_영상_audio_features_v2.json` — 오디오 피처
- `1장_영상_audio_quality_v2.json` — 오디오 품질 평가
- `1장_영상_emphasis_std_topic_v2.json` — 강조 구간 + 통계
- `1장_영상_emphasis_keywords_v2.json` — 사용된 주제 키워드 목록 (최대 20개)
- `1장_영상_by_slide_v2.json` — 슬라이드-컨텍스트-세그먼트 + 강조 상세
- `1장_영상_by_slide_iterative_v2.json` — 슬라이드별 iterative 병합 + 강조 요약
- `1장_영상_notes_v2.md` — 강의 정리 노트