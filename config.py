"""
config.py
=========
API 클라이언트 초기화 및 경로 상수 정의

경로 구조:
    input/
        lecture.mp4
    output_slides/
        metadata.json
        slide_0001.jpg
        ...
    output/
        {stem}_slide_textualized.json
        {stem}_annotation.json
        {stem}_segments.json
        {stem}_silences.json
        {stem}_deictics.json
        {stem}_deictics_ambiguous.json
        {stem}_audio_features.json
        {stem}_audio_quality.json
        {stem}_emphasis.json
        {stem}_by_slide.json
        {stem}_by_slide_iterative.json
        {stem}_slide_classified.json
        {stem}_fused.json
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

# ──────────────────────────────────────────────────────────────
# API 키
# ──────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

missing_keys: list[str] = []
if not GEMINI_API_KEY:
    missing_keys.append("GOOGLE_API_KEY / GEMINI_API_KEY")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if missing_keys:
    print("❌ 필요한 API 키를 환경 변수로 설정해주세요:")
    for k in missing_keys:
        print(f"   - {k}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# API 클라이언트
# ──────────────────────────────────────────────────────────────

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────────────────────
# 기본 경로 상수 (CLI 인자로 override 가능)
# ──────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR  = Path("input")
DEFAULT_SLIDES_DIR = Path("output_slides")   # metadata.json + 슬라이드 이미지
DEFAULT_OUTPUT_DIR = Path("output")          # 모든 분석 결과

# ──────────────────────────────────────────────────────────────
# 파일명 헬퍼
# ──────────────────────────────────────────────────────────────

def output_paths(stem: str, output_dir: Path, slides_dir: Path) -> dict[str, Path]:
    """
    영상 stem과 디렉토리로 모든 출력 경로를 한 번에 반환.

    사용 예:
        paths = output_paths("lecture", Path("output"), Path("output_slides"))
        paths["segments"]   # output/lecture_segments.json
        paths["metadata"]   # output_slides/metadata.json
    """
    return {
        # slides_dir
        "metadata":            slides_dir / "metadata.json",
        # output_dir
        "textualized":         output_dir / f"{stem}_slide_textualized.json",
        "annotation":          output_dir / f"{stem}_annotation.json",
        "segments":            output_dir / f"{stem}_segments.json",
        "silences":            output_dir / f"{stem}_silences.json",
        "deictics":            output_dir / f"{stem}_deictics.json",
        "deictics_ambiguous":  output_dir / f"{stem}_deictics_ambiguous.json",
        "audio_features":      output_dir / f"{stem}_audio_features.json",
        "audio_quality":       output_dir / f"{stem}_audio_quality.json",
        "emphasis":            output_dir / f"{stem}_emphasis.json",
        "by_slide":            output_dir / f"{stem}_by_slide.json",
        "by_slide_iterative":  output_dir / f"{stem}_by_slide_iterative.json",
        "classified":          output_dir / f"{stem}_slide_classified.json",
        "fused":               output_dir / f"{stem}_fused.json",
    }