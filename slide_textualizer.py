"""
슬라이드 텍스트화 파이프라인 (Stage 1)

슬라이드 이미지 내의 텍스트, 다이어그램, 표 등 시각 요소를
Gemini Vision을 통해 텍스트로 변환합니다.

Input:
  - output_slides/ 폴더: slide_extractor.py 출력 디렉토리
    ├── slide_001_base.jpg         ← annot 없을 때 사용
    ├── slide_001_annot_01.jpg
    ├── slide_001_annot_NN.jpg     ← annot 있을 때 마지막 프레임 사용 (애니메이션 완전 전개)
    ├── metadata.json
    └── ...

  annot이 있는 슬라이드는 마지막 annot 프레임을 텍스트 추출 기준으로 사용.
  → PPT 애니메이션으로 나중에 나타나는 텍스트까지 포함하기 위함.
  → VLM 프롬프트에 손글씨 제외 지시를 포함해 필기가 t1에 섞이지 않도록 처리.

Output:
  - slide_textualized.json: 텍스트화 결과

추출 항목:
  - t1          : 슬라이드 원본 텍스트 (Gemini Vision)
  - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

※ annotation 이미지의 강조 분석은 annotation_analyzer.py에서 별도 처리
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

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

@dataclass
class Config:
    slides_dir: Path = Path("output_slides")     # slide_extractor.py 출력 디렉토리
    output_dir: Path = Path("output")
    output_filename: str = "slide_textualized.json"  # 저장 파일명 ({stem}_slide_textualized.json)
    gemini_model: str = "models/gemini-2.5-flash"
    max_retries: int = 3
    retry_delay: float = 5.0

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# annot 없는 슬라이드 (이미지 1장): t1 + slide_emphasis 동시 추출
T1_EXTRACTION_PROMPT = """
이 슬라이드 이미지에서 슬라이드 유형을 판별하고, 텍스트와 구조 정보, 시각적 강조 요소를 추출하라.
설명 없이 JSON만 출력.

출력 형식:
{
  "slide_type": "text" | "image_only" | "mixed",
  "title": "슬라이드 제목 (없으면 빈 문자열)",
  "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
  "structure": "다이어그램/표/화살표 관계를 텍스트로 기술",
  "slide_emphasis": [
    {
      "text": "강조된 텍스트 원문",
      "type": "color" | "bold" | "underline" | "box" | "highlight" | "italic" | "callout" | "other",
      "color": "red",
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    }
  ]
}

slide_type 판별 기준:
- "text"       : 인쇄된 텍스트가 주요 내용 (일반 강의 슬라이드)
- "image_only" : 사진/그림/삽화만 있고 인쇄 텍스트가 없거나 극히 적음
- "mixed"      : 텍스트와 이미지가 함께 있어 둘 다 중요한 내용을 전달

추출 규칙:
- 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
- 원문 그대로 추출 (요약하지 말 것)
- 불릿 포인트는 "- " 또는 "• "로 시작
- 줄바꿈은 \\n으로 표시
- structure 필드: 다이어그램이 없으면 빈 문자열
  예시) "사용자 → 응용소프트웨어 → 운영체제 → 컴퓨터 하드웨어 (위에서 아래 계층 구조)"
  예시) "운영체제 vs 응용소프트웨어 비교표: 목적(자원관리 vs 사용자목적), 개발언어(C/C++ vs 다양)"

image_only / mixed 슬라이드 처리:
- raw_text: 이미지 내에 인쇄된 캡션/레이블 텍스트만 기재 (없으면 빈 문자열)
- structure: 이미지가 보여주는 장면/상황을 한 문장으로 기술

손글씨/필기 처리:
- 슬라이드 위에 교수가 직접 쓴 손글씨, 밑줄, 동그라미, 화살표 등 필기는 모두 무시
- 인쇄된 슬라이드 원본 텍스트만 추출할 것

slide_emphasis 추출 규칙:
- 슬라이드 제작 시 의도적으로 삽입된 시각적 강조만 수집
  (교수 필기/손글씨는 포함하지 말 것)
- 대상 및 type 값:
    "color"     : 다른 텍스트와 색상이 다른 텍스트 (빨간색, 주황색 등)
    "bold"      : 굵게 처리된 텍스트
    "underline" : 슬라이드 디자인의 일부인 밑줄
    "box"       : 강조 박스/테두리로 감싸진 텍스트
    "highlight" : 형광펜 효과
    "italic"    : 기울임 처리된 텍스트
    "callout"   : 말풍선/풍선 도형(callout shape) 안에 담긴 텍스트.
                  특정 요소를 화살표로 가리키며 설명하는 줄글 형태도 포함.
                  (단순 강조 목적이 아닌 설명 대체 용도로 보이는 경우에도 "callout"으로 분류)
    "other"     : 위에 해당하지 않는 기타 강조
- 불릿 포인트/리스트 마커(■ ▪ □ • ▶ 등)의 색상은 텍스트 강조로 보지 말 것.
  마커 색이 주황·빨강이더라도, 그 옆 텍스트 자체가 같은 색이 아니라면 "color" 강조로 포함하지 말 것.
- 강조 요소가 없으면 빈 배열 []
- bbox: 정규화 좌표 (0.0~1.0, 좌상단 기준) {"x": float, "y": float, "w": float, "h": float}
- color: 텍스트 색상이 강조 이유일 때만 기재 (예: "red", "orange"), 아니면 null
"""

# annot 있는 슬라이드 (이미지 2장):
# Image 1 (BASE)       → slide_emphasis 추출 (교수 필기 없는 원본)
# Image 2 (LAST_ANNOT) → t1 텍스트 추출 (PPT 애니메이션 완전 전개 상태)
T1_EXTRACTION_PROMPT_WITH_ANNOT = """
두 장의 이미지가 제공됩니다:
  - Image 1 (BASE)      : 교수 필기 전 원본 슬라이드
  - Image 2 (LAST_ANNOT): 교수 필기가 추가된 최종 상태 (PPT 애니메이션 완전 전개)

설명 없이 JSON만 출력.

출력 형식:
{
  "slide_type": "text" | "image_only" | "mixed",
  "title": "슬라이드 제목 (없으면 빈 문자열)",
  "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
  "structure": "다이어그램/표/화살표 관계를 텍스트로 기술",
  "slide_emphasis": [
    {
      "text": "강조된 텍스트 원문",
      "type": "color" | "bold" | "underline" | "box" | "highlight" | "italic" | "callout" | "other",
      "color": "red",
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    }
  ]
}

추출 규칙:
[raw_text / structure / slide_type]
  - Image 2 (LAST_ANNOT) 기준으로 추출
  - PPT 애니메이션으로 나중에 나타난 텍스트까지 모두 포함
  - 교수가 직접 쓴 손글씨/필기는 무시하고 인쇄된 원본 텍스트만 추출

[slide_emphasis]
  - 반드시 Image 1 (BASE) 만을 기준으로 판단할 것.
    Image 2는 slide_emphasis 판단에 절대 사용하지 말 것.
  - Image 2에만 보이는 빨간 선, 동그라미, 밑줄, 화살표 등은
    교수가 강의 중에 그린 필기이므로 slide_emphasis에 포함하지 말 것.
  - Image 1과 Image 2를 비교했을 때 Image 2에서 새로 생긴 색상/강조는
    슬라이드 디자인이 아닌 교수 필기이므로 포함하지 말 것.
  - 슬라이드 제작 시 의도적으로 삽입된 시각적 강조만 수집:
    "color"     : Image 1에서 이미 다른 텍스트와 색상이 다른 텍스트 (빨간색, 주황색 등)
    "bold"      : 굵게 처리된 텍스트
    "underline" : 슬라이드 디자인의 일부인 밑줄 (Image 1에 이미 존재하는 것만)
    "box"       : 강조 박스/테두리로 감싸진 텍스트
    "highlight" : 형광펜 효과
    "italic"    : 기울임 처리된 텍스트
    "callout"   : 말풍선/풍선 도형(callout shape) 안에 담긴 텍스트.
                  특정 요소를 화살표로 가리키며 설명하는 줄글 형태도 포함.
                  (단순 강조 목적이 아닌 설명 대체 용도로 보이는 경우에도 "callout"으로 분류)
    "other"     : 위에 해당하지 않는 기타 강조
  - 불릿 포인트/리스트 마커(■ ▪ □ • ▶ 등)의 색상은 텍스트 강조로 보지 말 것.
    마커 색이 주황·빨강이더라도, 그 옆 텍스트 자체가 같은 색이 아니라면 "color" 강조로 포함하지 말 것.
  - 강조 요소가 없으면 빈 배열 []
  - bbox: 정규화 좌표 (0.0~1.0, 좌상단 기준) {"x": float, "y": float, "w": float, "h": float}
  - color: 텍스트 색상이 강조 이유일 때만 기재 (예: "red", "orange"), 아니면 null
"""


# ============================================================================ #
#  슬라이드 로더                                                                 #
# ============================================================================ #

class SlideLoader:
    """
    slide_extractor.py 출력 디렉토리에서 텍스트 추출 대상 이미지 로드.

    선택 전략:
      - annot 프레임이 있는 슬라이드 → 마지막 annot 프레임 사용
        (PPT 애니메이션이 완전히 전개된 상태이므로 텍스트가 더 완전함)
      - annot 프레임이 없는 슬라이드 → base 프레임 사용

    타임스탬프는 metadata.json에서 읽고, 없으면 파일명 패턴으로 폴백.
    """

    def __init__(self, slides_dir: Path):
        self.slides_dir = Path(slides_dir)

    def load(self) -> List[Dict]:
        metadata_path = self.slides_dir / "metadata.json"

        if metadata_path.exists():
            return self._load_from_metadata(metadata_path)
        else:
            logger.warning("metadata.json 없음 - 파일명 패턴으로 폴백")
            return self._load_from_filenames()

    def _load_from_metadata(self, metadata_path: Path) -> List[Dict]:
        """
        metadata.json 기준으로 슬라이드별 텍스트 추출 대상 이미지 결정.
        annot이 있으면 마지막 annot, 없으면 base 프레임 사용.
        """
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # slide_number 기준으로 base / annot 분류
        base_entries: Dict[int, dict] = {}
        annot_entries: Dict[int, List[dict]] = {}

        for entry in metadata:
            idx = entry.get("slide_index")
            if idx is None:
                continue
            if entry.get("capture_type") == "base":
                base_entries[idx] = entry
            elif entry.get("capture_type") in ("annot", "annotation"):
                annot_entries.setdefault(idx, []).append(entry)

        slides = []
        for slide_num in sorted(base_entries.keys()):
            base = base_entries[slide_num]
            annots = annot_entries.get(slide_num, [])

            if annots:
                # 마지막 annot 프레임 선택 (annot_index 기준)
                last_annot = max(annots, key=lambda x: x.get("annot_index", 0))
                target = last_annot
                source = f"last_annot (annot_index={last_annot.get('annot_index')})"
            else:
                target = base
                source = "base"

            image_path = self.slides_dir / target["filename"]
            if not image_path.exists():
                logger.warning(f"파일 없음: {image_path}, base로 재시도")
                image_path = self.slides_dir / base["filename"]
                if not image_path.exists():
                    logger.warning(f"base도 없음: {image_path}, 스킵")
                    continue
                source = "base (fallback)"

            base_image_path = self.slides_dir / base["filename"]

            slides.append({
                "slide_number": slide_num,
                "timestamp":    base.get("timestamp_sec", 0.0),  # 타임스탬프는 항상 base 기준
                "image_path":   str(self.slides_dir / base["filename"]),  # 저장 경로는 base 기록
                "image":        Image.open(image_path).convert("RGB"),
                "base_image":   Image.open(base_image_path).convert("RGB"),  # 슬라이드 강조 감지용
                "has_annot":    bool(annots),
                "text_source":  source,
            })

        slides.sort(key=lambda x: x["slide_number"])
        last_annot_count = sum(1 for s in slides if "annot" in s["text_source"])
        logger.info(
            f"✓ Loaded {len(slides)} slides from metadata.json "
            f"(last_annot: {last_annot_count}, base: {len(slides)-last_annot_count})"
        )
        return slides

    def _load_from_filenames(self) -> List[Dict]:
        """
        metadata 없을 때 파일명 패턴으로 폴백.
        annot 있으면 마지막 annot, 없으면 base 사용.
        """
        base_pattern  = re.compile(r'slide_(\d+)_base\.(jpg|png)', re.IGNORECASE)
        annot_pattern = re.compile(r'slide_(\d+)_annot_(\d+)\.(jpg|png)', re.IGNORECASE)

        base_files: Dict[int, Path] = {}
        annot_files: Dict[int, List[tuple]] = {}  # {slide_num: [(annot_idx, path), ...]}

        for file in self.slides_dir.iterdir():
            m = base_pattern.match(file.name)
            if m:
                base_files[int(m.group(1))] = file
                continue
            m = annot_pattern.match(file.name)
            if m:
                num, idx = int(m.group(1)), int(m.group(2))
                annot_files.setdefault(num, []).append((idx, file))

        slides = []
        for slide_num in sorted(base_files.keys()):
            base_path = base_files[slide_num]
            annots = annot_files.get(slide_num, [])

            if annots:
                last_annot_path = max(annots, key=lambda x: x[0])[1]
                target_path = last_annot_path
                source = f"last_annot"
            else:
                target_path = base_path
                source = "base"

            slides.append({
                "slide_number": slide_num,
                "timestamp":    0.0,
                "image_path":   str(base_path),
                "image":        Image.open(target_path).convert("RGB"),
                "base_image":   Image.open(base_path).convert("RGB"),
                "has_annot":    bool(annots),
                "text_source":  source,
            })

        last_annot_count = sum(1 for s in slides if "annot" in s["text_source"])
        logger.info(
            f"✓ Loaded {len(slides)} slides from filenames "
            f"(last_annot: {last_annot_count}, base: {len(slides)-last_annot_count})"
        )
        return slides


# ============================================================================ #
#  t1 추출기 (Gemini Vision)                                                    #
# ============================================================================ #

class T1Extractor:
    """슬라이드 base 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

    def __init__(self, config: Config):
        self.config = config
        from config import gemini_client
        self.client = gemini_client
        logger.info("✓ Gemini initialized for t1 extraction")

    @staticmethod
    def _filter_emphasis(emphasis_list: List[Dict]) -> List[Dict]:
        """
        slide_emphasis 후처리:
          1. 순번 단독 항목 제거
             - "1" / "2." / "3)" 처럼 숫자 + 구두점/공백만 남는 경우 제거
             - "1. 운영체제의 목적"(뒤에 내용 있음), "256GB"(단위 혼합)는 유지
          2. 동일 텍스트 중복 병합
             - 같은 text(정규화 기준)가 여러 항목으로 등장하면 하나로 합침
             - type은 리스트로 수집 후 중복 제거: ["bold", "box"]
             - bbox는 첫 번째 항목 기준 유지
          3. emphasis_weight 부여 (병합 후 적용)
             - callout 타입 포함 : 0.3
             - box 타입만 포함   : 0.5  (테두리 박스 — 레이아웃 요소일 수 있음)
             - 그 외             : 1.0
        """
        WEIGHT_CALLOUT = 0.3
        WEIGHT_BOX     = 0.5
        WEIGHT_NORMAL  = 1.0

        # 순번 단독 패턴: "1" / "2." / "3)" / "10. " 등
        _standalone_num = re.compile(r'^\d+[\.\)]*\s*$')

        # ── 1단계: 순번 제거 + 텍스트별 버킷으로 수집 ──────────────────────── #
        buckets: Dict[str, Dict] = {}
        for item in emphasis_list:
            text = item.get("text", "").strip()
            if not text:
                continue
            if _standalone_num.match(text):
                logger.debug(f"  [emphasis filter] 순번 단독 제거: '{text}'")
                continue

            key = " ".join(text.lower().split())
            t   = item.get("type", "other") or "other"

            if key not in buckets:
                buckets[key] = {
                    "text":   text,
                    "types":  [t],
                    "color":  item.get("color"),
                    "bbox":   item.get("bbox"),
                }
            else:
                if t not in buckets[key]["types"]:
                    buckets[key]["types"].append(t)
                # color가 있으면 채움 (첫 항목에 없을 수도 있으므로)
                if not buckets[key]["color"] and item.get("color"):
                    buckets[key]["color"] = item.get("color")

        # ── 2단계: weight 부여 후 최종 리스트 생성 ──────────────────────────── #
        filtered = []
        for bucket in buckets.values():
            types = bucket["types"]
            type_val = types[0] if len(types) == 1 else types  # 단일이면 str, 복수면 list

            if "callout" in types:
                weight = WEIGHT_CALLOUT
            elif types == ["box"]:
                weight = WEIGHT_BOX
            else:
                weight = WEIGHT_NORMAL

            filtered.append({
                "text":             bucket["text"],
                "type":             type_val,
                "color":            bucket["color"],
                "bbox":             bucket["bbox"],
                "emphasis_weight":  weight,
            })

        return filtered

    def _call_gemini(self, image: Image.Image, base_image: Image.Image = None) -> str:
        """재시도 로직 포함 Gemini Vision 호출.

        base_image가 주어지면 annot 있는 슬라이드 — 2장 프롬프트 사용:
          Image 1 (BASE)      → slide_emphasis 추출
          Image 2 (LAST_ANNOT)→ t1 텍스트 추출
        base_image가 없으면 단일 이미지 프롬프트 사용.
        """
        if base_image is not None:
            contents = [
                "Image 1 (BASE):", base_image,
                "Image 2 (LAST_ANNOT):", image,
                T1_EXTRACTION_PROMPT_WITH_ANNOT,
            ]
        else:
            contents = [T1_EXTRACTION_PROMPT, image]

        last_exc = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
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
        """단일 슬라이드에서 t1, t1_structure, slide_emphasis 추출"""
        slide.setdefault("title", f"Slide {slide['slide_number']}")
        slide.setdefault("t1", "")
        slide.setdefault("t1_structure", "")
        slide.setdefault("slide_emphasis", [])

        # annot 있는 슬라이드: base + last_annot 2장 전달
        # annot 없는 슬라이드: base만 전달 (이미지가 동일하므로 1장 프롬프트 사용)
        has_annot  = slide.get("has_annot", False)
        base_image = slide.get("base_image") if has_annot else None

        try:
            raw_text = self._call_gemini(slide["image"], base_image=base_image)

            # 코드펜스 제거
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0]
            raw_text = raw_text.strip()

            # 문자열 값 내부의 제어문자 정제
            def clean_string_value(m):
                inner = m.group(1)
                inner = re.sub(r'[\n\r\t]', ' ', inner)
                inner = re.sub(r' {2,}', ' ', inner).strip()
                return f'"{inner}"'
            raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

            # 파싱 → json_repair fallback
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                if JSON_REPAIR_AVAILABLE:
                    result = json.loads(repair_json(raw_text))
                else:
                    raise

            slide["title"]          = result.get("title", f"Slide {slide['slide_number']}")
            slide["t1"]             = result.get("raw_text", "")
            slide["t1_structure"]   = result.get("structure", "")
            slide["slide_type"]     = result.get("slide_type", "text")
            slide["slide_emphasis"] = self._filter_emphasis(
                result.get("slide_emphasis", [])
            )

        except Exception as e:
            logger.error(
                f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
            )

        return slide

    def extract_batch(self, slides: List[Dict]) -> List[Dict]:
        logger.info(f"Extracting t1 from {len(slides)} slides...")

        for i, slide in enumerate(slides):
            self.extract(slide)
            logger.info(
                f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']} "
                f"[{slide.get('text_source', 'base')}]: "
                f"t1={len(slide['t1'])} chars, "
                f"structure={len(slide['t1_structure'])} chars, "
                f"slide_emphasis={len(slide.get('slide_emphasis', []))}개"
            )

        logger.info("✓ t1 extraction complete")
        return slides


# ============================================================================ #
#  파이프라인                                                                    #
# ============================================================================ #

class TextualizationPipeline:
    """슬라이드 텍스트화 파이프라인"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self) -> Dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🎓 슬라이드 텍스트화 파이프라인 (Stage 1)")
        print("="*70)
        print(f"📁 Slides : {self.config.slides_dir}")
        print(f"📂 Output : {self.config.output_dir}")

        # Stage 1: base 슬라이드 로드
        print("\n" + "-"*70)
        print("Stage 1: 텍스트 추출 대상 슬라이드 로드 (annot 있으면 last_annot, 없으면 base)")
        print("-"*70)

        slides = SlideLoader(self.config.slides_dir).load()

        # Stage 2: t1 추출 (텍스트 + 구조)
        print("\n" + "-"*70)
        print("Stage 2: 텍스트화 (Gemini Vision)")
        print("-"*70)

        slides = T1Extractor(self.config).extract_batch(slides)

        # Stage 3: 결과 저장
        print("\n" + "-"*70)
        print("Stage 3: 결과 저장")
        print("-"*70)

        for slide in slides:
            ts = slide["timestamp"]
            mins, secs = divmod(ts, 60)
            hrs, mins = divmod(mins, 60)
            slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
            slide["slide_id"] = f"slide_{slide['slide_number']:03d}"

        total_time = time.time() - start_time

        result = {
            "metadata": {
                "slides_dir":      str(self.config.slides_dir),
                "processing_time": total_time,
                "total_slides":    len(slides),
            },
            "slides": [
                {
                    "slide_id":            s["slide_id"],
                    "slide_number":        s["slide_number"],
                    "timestamp":           s["timestamp"],
                    "timestamp_formatted": s["timestamp_formatted"],
                    "image_path":          s["image_path"],
                    "title":               s["title"],
                    "t1":                  s["t1"],
                    "t1_structure":        s["t1_structure"],
                    "slide_type":          s.get("slide_type", "text"),
                    "slide_emphasis":      s.get("slide_emphasis", []),
                    "text_source":         s.get("text_source", "base"),
                }
                for s in slides
            ]
        }

        output_path = self.config.output_dir / self.config.output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path}")

        structure_count = sum(1 for s in slides if s.get("t1_structure"))

        print("\n" + "="*70)
        print("✅ 텍스트화 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 슬라이드:      {len(slides)}개")
        image_only_count = sum(1 for s in slides if s.get("slide_type") == "image_only")
        mixed_count      = sum(1 for s in slides if s.get("slide_type") == "mixed")
        print(f"  • t1 추출:       {sum(1 for s in slides if s['t1'])}개")
        print(f"  • t1_structure:  {structure_count}개")
        print(f"  • image_only:    {image_only_count}개")
        print(f"  • mixed:         {mixed_count}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"\n⏱️  처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse
    from config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
    parser.add_argument("-s", "--slides", default=str(DEFAULT_SLIDES_DIR),
                        help=f"slide_extractor.py 출력 디렉토리 (default: {DEFAULT_SLIDES_DIR})")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"결과 저장 디렉토리 (default: {DEFAULT_OUTPUT_DIR})")
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

    TextualizationPipeline(config).run()


if __name__ == "__main__":
    main()