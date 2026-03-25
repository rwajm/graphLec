통합된 GraphLec 프로젝트의 README.md 파일 내용입니다. 아래 내용을 그대로 복사하여 사용하시면 됩니다.
GraphLec: 통합 강의 분석 및 지식그래프 생성 파이프라인

GraphLec은 강의 영상에서 시각적 정보(슬라이드)와 청각적 정보(음성)를 결합하여, 고정밀 전사, 강조 구간 분석, 그리고 최종적인 지식그래프(Knowledge Graph)를 구축하는 멀티모달 파이프라인입니다.

1. 주요 기능
🎥 영상 및 슬라이드 분석

    슬라이드 변화 감지: MSE(Mean Squared Error) 기반으로 영상 내 슬라이드 전환 시점을 추적하여 이미지를 추출합니다.

    시각 텍스트 추출 (t1): Gemini Vision을 사용하여 각 슬라이드의 텍스트와 이미지 벡터를 추출합니다.

🎙️ 음성 전사 및 강조 분석

    슬라이드별 맞춤 전사: Groq Whisper(whisper-large-v3-turbo)를 사용하여 슬라이드 구간에 맞춰 오디오를 잘라 전사합니다.

    2단계 텍스트 교정:

        1단계: 전문용어 및 오타 위주의 최소 교정을 수행합니다.

        2단계: 슬라이드 컨텍스트를 반영하여 추임새를 제거하고 자연스러운 문장으로 변환합니다.

    강조 구간 탐지: 오디오 표준편차와 가중치 키워드 반복 빈도, LLM 필터를 결합하여 핵심 강의 구간을 탐지합니다.

🕸️ 지식그래프 및 Q&A

    멀티모달 통합 (t3): 슬라이드 텍스트(t1)와 오디오 전사(t2)를 타임스탬프 기준으로 병합하고 임베딩합니다.

    지식그래프 구축: 추출된 개념 간의 관계(is_a, uses, solves 등 12종)를 정의하고 PyVis로 시각화합니다.

    그래프 기반 Q&A: 구축된 그래프 데이터를 바탕으로 자연어 질의응답 시스템을 제공합니다.

2. 파이프라인 구조
코드 스니펫

graph TD
    Video[강의 영상 .mp4] --> Stage0[Stage 0: 슬라이드 추출 MSE]
    Video --> AudioPipe[음성 파이프라인: Whisper 전사]
    
    Stage0 --> Stage1[Stage 1: 슬라이드 텍스트 추출 t1]
    AudioPipe --> Emphasis[강조 구간 및 키워드 분석]
    
    Stage1 --> Stage2[Stage 2: 텍스트/오디오 통합 t3]
    Emphasis --> Stage2
    
    Stage2 --> Stage3[Stage 3: 지식그래프 생성]
    Stage3 --> QA[Graph Q&A 서비스]

3. 설치 및 설정
패키지 설치
Bash

pip install -r requirements.txt

환경 변수 설정

.env 파일에 다음 API 키를 설정해야 합니다:

    GROQ_API_KEY: 고속 Whisper 전사 서비스용.

    GOOGLE_API_KEY (또는 GEMINI_API_KEY): Gemini Vision 및 임베딩용.

4. 사용법
전체 실행
Bash

python main.py --input input/lecture.mp4

주요 옵션

    --threshold: MSE 감지 임계값 (기본 1000).

    --skip-stage0: 이미 추출된 슬라이드 이미지가 있는 경우 건너뛰기.

    --only <n>: 특정 스테이지만 실행 (예: 지식그래프만 다시 생성 시 --only 3).

5. 결과물 파일 안내 (output/)
파일명 (예시)	유형	설명
*_notes_v2.md	문서	교정된 전사를 바탕으로 생성된 강의 정리 노트
*_emphasis_std_topic_v2.json	분석	오디오 통계 및 키워드 기반 강조 데이터
integrated_text.json	데이터	슬라이드+음성 통합 텍스트 및 벡터 데이터 (t3)
knowledge_graph.html	시각화	인터랙티브 지식그래프 (브라우저 확인용)
slide_NNN_Xs.jpg	이미지	영상에서 추출된 개별 슬라이드 컷
6. 요구 사항 (requirements.txt)

    opencv-python, numpy, Pillow

    google-generativeai, google-genai

    torch, transformers

    pyvis, python-dotenv