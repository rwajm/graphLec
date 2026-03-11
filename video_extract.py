"""
슬라이드 데이터 추출 파이프라인 (Stage 1)

Input:
  - slides/ 폴더: 슬라이드 이미지

Output:
  - slide_extracted.json: t1 추출 결과

추출 항목:
  - t1: 슬라이드 원본 텍스트 (Gemini Vision)
  - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

※ 오디오 정제는 다음 단계에서 t1을 사용하여 진행
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

# Stage 2와 동일한 SDK 사용
from google import genai
from google.genai import types

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

@dataclass
class Config:
    google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
    slides_dir: Path = Path("./slides")
    output_dir: Path = Path("./output")
    gemini_model: str = "models/gemini-2.5-flash"
    max_retries: int = 3
    retry_delay: float = 5.0

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# t1 추출용 프롬프트
T1_EXTRACTION_PROMPT = """
이 슬라이드 이미지에서 보이는 모든 텍스트와 구조 정보를 추출하라.
설명 없이 JSON만 출력.

출력 형식:
{
  "title": "슬라이드 제목",
  "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
  "structure": "다이어그램/표/화살표 관계를 텍스트로 기술"
}

추출 규칙:
- 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
- 원문 그대로 추출 (요약하지 말 것)
- 불릿 포인트는 "- " 또는 "• "로 시작
- 줄바꿈은 \\n으로 표시
- structure 필드: 다이어그램이 없으면 빈 문자열
  예시) "사용자 → 응용소프트웨어 → 운영체제 → 컴퓨터 하드웨어 (위에서 아래 계층 구조)"
  예시) "운영체제 vs 응용소프트웨어 비교표: 목적(자원관리 vs 사용자목적), 개발언어(C/C++ vs 다양)"
"""


# ============================================================================ #
#  슬라이드 로더                                                                 #
# ============================================================================ #

class SlideLoader:
    def __init__(self, slides_dir: Path):
        self.slides_dir = Path(slides_dir)

    def load(self) -> List[Dict]:
        slides = []
        pattern = re.compile(r'slide_(\d+)_(\d+\.?\d*)s\.(jpg|png)')

        for file in sorted(self.slides_dir.iterdir()):
            match = pattern.match(file.name)
            if match:
                slides.append({
                    "slide_number": int(match.group(1)),
                    "timestamp": float(match.group(2)),
                    "image_path": str(file),
                    "image": Image.open(file).convert("RGB")
                })

        logger.info(f"✓ Loaded {len(slides)} slides")
        return slides


# ============================================================================ #
#  t1 추출기 (Gemini Vision)                                                    #
# ============================================================================ #

class T1Extractor:
    """슬라이드 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

    def __init__(self, config: Config):
        self.config = config
        # Stage 2와 동일한 방식으로 클라이언트 초기화
        self.client = genai.Client(api_key=config.google_api_key)
        logger.info("✓ Gemini initialized for t1 extraction")

    def _call_gemini(self, image: Image.Image) -> str:
        """재시도 로직 포함 Gemini Vision 호출"""
        last_exc = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=[T1_EXTRACTION_PROMPT, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                return response.text
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"  ⚠ Gemini call failed "
                    f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        raise last_exc

    def extract(self, slide: Dict) -> Dict:
        """단일 슬라이드에서 t1, t1_structure 추출"""
        # 실패 시 폴백 기본값을 미리 설정
        slide.setdefault("title", f"Slide {slide['slide_number']}")
        slide.setdefault("t1", "")
        slide.setdefault("t1_structure", "")

        try:
            raw_text = self._call_gemini(slide["image"])

            # 1. 코드펜스 제거
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0]
            raw_text = raw_text.strip()

            # 2. 문자열 값 내부의 제어문자 정제
            def clean_string_value(m):
                inner = m.group(1)
                inner = re.sub(r'[\n\r\t]', ' ', inner)
                inner = re.sub(r' {2,}', ' ', inner).strip()
                return f'"{inner}"'
            raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

            # 3. 파싱 시도 → json_repair fallback
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                if JSON_REPAIR_AVAILABLE:
                    result = json.loads(repair_json(raw_text))
                else:
                    raise

            slide["title"] = result.get("title", f"Slide {slide['slide_number']}")
            slide["t1"] = result.get("raw_text", "")
            # structure를 t1에 합치지 않고 별도 필드로 분리
            # → Stage 3 프롬프트에서 관계 추출 힌트로 독립적으로 활용 가능
            slide["t1_structure"] = result.get("structure", "")

        except Exception as e:
            # result가 정의되지 않은 상태일 수 있으므로 기본값(setdefault)으로만 처리
            logger.error(
                f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
            )

        return slide

    def extract_batch(self, slides: List[Dict]) -> List[Dict]:
        logger.info(f"Extracting t1 from {len(slides)} slides...")

        for i, slide in enumerate(slides):
            self.extract(slide)
            logger.info(
                f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']}: "
                f"t1={len(slide['t1'])} chars, "
                f"structure={len(slide['t1_structure'])} chars"
            )

        logger.info("✓ t1 extraction complete")
        return slides


# ============================================================================ #
#  파이프라인                                                                    #
# ============================================================================ #

class ExtractionPipeline:
    """t1 추출 파이프라인"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self) -> Dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🎓 슬라이드 데이터 추출 파이프라인 (Stage 1)")
        print("="*70)
        print(f"📁 Slides: {self.config.slides_dir}")
        print(f"📂 Output: {self.config.output_dir}")

        # Stage 1: 슬라이드 로드
        print("\n" + "-"*70)
        print("Stage 1: 슬라이드 로드")
        print("-"*70)

        slides = SlideLoader(self.config.slides_dir).load()

        # Stage 2: t1 추출 (슬라이드 텍스트 + 구조)
        print("\n" + "-"*70)
        print("Stage 2: t1 추출 (Gemini Vision)")
        print("-"*70)

        slides = T1Extractor(self.config).extract_batch(slides)

        # Stage 3: 결과 저장
        print("\n" + "-"*70)
        print("Stage 3: 결과 저장")
        print("-"*70)

        # 타임스탬프 포맷팅 및 slide_id 생성
        for slide in slides:
            ts = slide["timestamp"]
            mins, secs = divmod(ts, 60)
            hrs, mins = divmod(mins, 60)
            slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
            slide["slide_id"] = f"slide_{slide['slide_number']:03d}"

        result = {
            "metadata": {
                "slides_dir": str(self.config.slides_dir),
                "processing_time": time.time() - start_time,
                "total_slides": len(slides)
            },
            "slides": [
                {
                    "slide_id": s["slide_id"],
                    "slide_number": s["slide_number"],
                    "timestamp": s["timestamp"],
                    "timestamp_formatted": s["timestamp_formatted"],
                    "image_path": s["image_path"],
                    "title": s["title"],
                    "t1": s["t1"],
                    # raw_text와 분리 저장 — Stage 3에서 관계 추출 시 독립적으로 참조
                    "t1_structure": s["t1_structure"],
                }
                for s in slides
            ]
        }

        output_path = self.config.output_dir / "slide_extracted.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path}")

        # 완료 리포트
        total_time = time.time() - start_time
        structure_count = sum(1 for s in slides if s.get("t1_structure"))

        print("\n" + "="*70)
        print("✅ 추출 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 슬라이드: {len(slides)}개")
        print(f"  • t1 추출: {sum(1 for s in slides if s['t1'])}개")
        print(f"  • t1_structure 추출: {structure_count}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"\n⏱️ 처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="t1 추출")
    parser.add_argument("-s", "--slides", default="./slides")
    parser.add_argument("-o", "--output", default="./output")
    parser.add_argument("--retries", type=int, default=3,
                        help="Gemini API 재시도 횟수 (default: 3)")

    args = parser.parse_args()

    config = Config(
        slides_dir=Path(args.slides),
        output_dir=Path(args.output),
        max_retries=args.retries,
    )

    if not config.slides_dir.exists():
        print(f"❌ Slides folder not found: {config.slides_dir}")
        return

    pipeline = ExtractionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()