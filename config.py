"""
모든 파이프라인 스테이지에서 공유하는 설정값을 관리합니다.
API 키는 환경변수로 관리하는 것을 권장합니다.

환경변수 설정 예시:
  export GOOGLE_API_KEY="your_api_key_here"
  export GROQ_API_KEY="your_groq_api_key_here"

또는 .env 파일 사용 (python-dotenv 설치 필요):
  GOOGLE_API_KEY=your_api_key_here
  GROQ_API_KEY=your_groq_api_key_here
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ─── python-dotenv 지원 (선택적) ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================ #
#  설치 필요 패키지 목록 (requirements.txt 참고)
# ============================================================================ #
#
# [Core]
#   opencv-python>=4.8.0          # 영상 처리
#   numpy>=1.24.0                 # 수치 연산
#   Pillow>=10.0.0                # 이미지 처리
#   imagehash>=4.3.0              # pHash 슬라이드 감지 (Preprocessing)
#
# [Gemini API]
#   google-generativeai>=0.8.0    # Gemini Vision / 텍스트 추출 (Stage 3)
#   google-genai>=0.8.0           # Gemini Embedding (Stage 2, Q&A, Preprocessing)
#
# [Groq Whisper]
#   groq>=0.4.0                   # Groq Whisper 전사 (Preprocessing)
#
# [Graph]
#   pyvis>=0.3.2                  # 지식그래프 HTML 시각화 (Stage 3)
#
# [Optional]
#   python-dotenv>=1.0.0          # .env 파일 지원
#   python-pptx>=0.6.21           # PPTX 텍스트 추출 (Preprocessing, 선택)
#
# ffmpeg 설치 필요 (Preprocessing: 오디오 추출 및 프레임 샘플링)
# ============================================================================ #


@dataclass
class PipelineConfig:
    """전체 파이프라인 통합 설정"""

    # ─── API 키 ──────────────────────────────────────────────────────────────
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )

    # ─── 경로 설정 ───────────────────────────────────────────────────────────
    video_path: str = ""                    # 입력 영상 파일 (예: lecture.mp4)
    pptx_path: str = ""                     # PPTX 강의 자료 (선택)
    output_dir: str = "./output"            # 결과 저장 폴더

    # ─── Stage 2: 텍스트 통합 및 임베딩 ─────────────────────────────────────
    embedding_model: str = "models/gemini-embedding-001"
    # 텍스트 벡터 생성에 사용할 Gemini Embedding 모델

    embedding_dim: int = 768
    # 임베딩 벡터 차원 수

    # ─── Stage 3: 지식그래프 생성 ────────────────────────────────────────────
    gemini_model: str = "models/gemini-2.5-flash"
    # 개념/관계 추출에 사용할 Gemini 모델

    def __post_init__(self):
        # API 키 경고
        if not self.google_api_key:
            print("⚠️  GOOGLE_API_KEY가 설정되지 않았습니다.")
            print("   환경변수를 설정하거나 .env 파일에 입력하세요.")

        # 출력 폴더 생성
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """설정 유효성 검사"""
        ok = True

        if not self.google_api_key:
            print("❌ GOOGLE_API_KEY 미설정")
            ok = False

        if self.video_path and not Path(self.video_path).exists():
            print(f"❌ 영상 파일 없음: {self.video_path}")
            ok = False

        if self.pptx_path and not Path(self.pptx_path).exists():
            print(f"❌ PPTX 파일 없음: {self.pptx_path}")
            ok = False

        return ok
