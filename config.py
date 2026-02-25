"""
모든 파이프라인 스테이지에서 공유하는 설정값을 관리합니다.
API 키는 환경변수로 관리하는 것을 권장합니다.

환경변수 설정 예시:
  export GOOGLE_API_KEY="your_api_key_here"

또는 .env 파일 사용 (python-dotenv 설치 필요):
  GOOGLE_API_KEY=your_api_key_here
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
#   opencv-python>=4.8.0          # 영상 처리 (Stage 0)
#   numpy>=1.24.0                 # 수치 연산
#   Pillow>=10.0.0                # 이미지 처리
#
# [Gemini API]
#   google-generativeai>=0.8.0    # Gemini Vision / 텍스트 추출 (Stage 1, 2, 3)
#   google-genai>=0.8.0           # Gemini Embedding (Stage 2, Q&A)
#
# [ColPali - 이미지 벡터화]
#   torch>=2.0.0                  # PyTorch (Stage 1)
#   colpali-engine>=0.3.0         # ColPali 모델 (Stage 1)
#   transformers>=4.40.0          # HuggingFace 모델 로더
#
# [Graph]
#   pyvis>=0.3.2                  # 지식그래프 HTML 시각화 (Stage 3)
#
# [Optional]
#   python-dotenv>=1.0.0          # .env 파일 지원
#
# GPU 사용 시 CUDA 버전에 맞는 PyTorch 설치 필요:
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
# ============================================================================ #


@dataclass
class PipelineConfig:
    """전체 파이프라인 통합 설정"""

    alpha: float = 0.4  # 타임스탬프 가중치 (1-alpha = 임베딩 가중치)

    # ─── API 키 ──────────────────────────────────────────────────────────────
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )

    # ─── 경로 설정 ───────────────────────────────────────────────────────────
    video_path: str = ""                    # 입력 영상 파일 (예: lecture.mp4)
    audio_json: str = "./audio.json"        # Whisper 전사 결과 JSON
    slides_dir: str = "./slides"            # Stage 0 → Stage 1 중간 폴더
                                            #   Stage 0: 슬라이드 이미지 저장
                                            #   Stage 1: 슬라이드 이미지 읽기
    output_dir: str = "./output"            # Stage 1~3 JSON/HTML 결과 저장 폴더

    # ─── Stage 0: MSE 슬라이드 감지 ─────────────────────────────────────────
    mse_threshold: int = 500
    # 슬라이드 변화 감지 임계값. 낮을수록 민감 (500~2000 권장)
    # - 500  : 미세한 변화도 감지 (과검출 가능)
    # - 1000 : 기본값 (일반 강의)
    # - 2000 : 큰 변화만 감지 (미검출 가능)

    mse_sample_rate: float = 0.5
    # 프레임 샘플링 간격 (초). 0.5 = 초당 2프레임 검사

    # ─── Stage 1: 슬라이드 텍스트/벡터 추출 ──────────────────────────────────
    gemini_model: str = "models/gemini-2.5-flash"
    # t1 추출 및 개념/관계 추출에 사용할 Gemini 모델

    colpali_model: str = "vidore/colpali-v1.2"
    # ColPali 이미지 벡터화 모델 (HuggingFace)

    device: str = "cuda"
    # ColPali 연산 디바이스. GPU 없으면 자동으로 "cpu"로 전환됨

    # ─── Stage 2: 텍스트 통합 및 임베딩 ─────────────────────────────────────
    embedding_model: str = "models/gemini-embedding-001"
    # 텍스트 벡터 생성에 사용할 Gemini Embedding 모델

    embedding_dim: int = 768
    # 임베딩 벡터 차원 수

    # ─── Stage 3: 지식그래프 생성 ────────────────────────────────────────────
    # gemini_model 동일 사용 (개념/관계 추출)
    # output: knowledge_graph.json, knowledge_graph.html

    def __post_init__(self):
        # API 키 경고
        if not self.google_api_key:
            print("⚠️  GOOGLE_API_KEY가 설정되지 않았습니다.")
            print("   환경변수를 설정하거나 config.py의 google_api_key를 직접 입력하세요.")

        # 출력 폴더 생성
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.slides_dir).mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """설정 유효성 검사"""
        ok = True

        if not self.google_api_key:
            print("❌ google_api_key 미설정")
            ok = False

        if self.video_path and not Path(self.video_path).exists():
            print(f"❌ 영상 파일 없음: {self.video_path}")
            ok = False

        if not Path(self.audio_json).exists():
            print(f"❌ 오디오 JSON 없음: {self.audio_json}")
            ok = False

        return ok