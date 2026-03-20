import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from google import genai

# .env 파일 로드
load_dotenv()

@dataclass
class PipelineConfig:
    """전체 파이프라인 통합 설정 및 API 클라이언트 관리"""

    # ─── API 키 설정 ──────────────────────────────────────────────────────────
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    # ─── 경로 설정 ─────────────────────────────────────────────────────────────
    video_path: str = ""                    # 입력 영상 파일 
    audio_json: str = "./audio.json"        # Whisper 전사 결과 JSON 
    slides_dir: str = "./slides"            # 슬라이드 이미지 저장/읽기 폴더 
    output_dir: str = "./output"            # 결과물 저장 폴더 

    # ─── Stage 0 & 1: 영상 처리 및 텍스트 추출 ────────────────────────────────
    mse_threshold: int = 1000               # 슬라이드 변화 감지 임계값 (500~2000) 
    mse_sample_rate: float = 0.5            # 프레임 샘플링 간격 (초) 
    gemini_model: str = "models/gemini-2.0-flash"  # 텍스트 및 개념 추출 모델 

    # ─── Stage 2 & 3: 통합 및 지식그래프 ──────────────────────────────────────
    embedding_model: str = "models/text-embedding-004"
    embedding_dim: int = 768
    match_threshold: float = 0.55           # 오디오-슬라이드 매칭 최소 점수 
    alpha: float = 0.4                      # 타임스탬프 가중치 
    synonyms_path: Optional[str] = None     # 동의어 사전 경로 

    def __post_init__(self):
        """초기화 및 디렉토리 생성"""
        self._validate_keys()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True) [cite: 1]
        Path(self.slides_dir).mkdir(parents=True, exist_ok=True) [cite: 1]

    def _validate_keys(self):
        """API 키 존재 여부 확인"""
        missing_keys = []
        if not self.google_api_key:
            missing_keys.append("GOOGLE_API_KEY/GEMINI_API_KEY") [cite: 2]
        if not self.groq_api_key:
            missing_keys.append("GROQ_API_KEY") [cite: 2]

        if missing_keys:
            print("❌ 필요한 API 키가 설정되지 않았습니다:")
            for k in missing_keys:
                print(f"   - {k}")
            sys.exit(1) [cite: 2]

    def get_gemini_client(self):
        """Gemini API 클라이언트 반환"""
        return genai.Client(api_key=self.google_api_key) [cite: 2]

    def get_groq_client(self):
        """Groq API 클라이언트 반환"""
        return Groq(api_key=self.groq_api_key) [cite: 2]

    def validate_paths(self) -> bool:
        """입력 파일 유효성 검사"""
        if self.video_path and not Path(self.video_path).exists():
            print(f"❌ 영상 파일 없음: {self.video_path}") [cite: 1]
            return False
        return True

# 기본 설정 인스턴스 생성
config = PipelineConfig()
gemini_client = config.get_gemini_client()
groq_client = config.get_groq_client()