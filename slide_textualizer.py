# """
# 슬라이드 텍스트화 파이프라인 (Stage 1)

# 슬라이드 이미지 내의 텍스트, 다이어그램, 표 등 시각 요소를
# Gemini Vision을 통해 텍스트로 변환합니다.

# Input:
#   - output_slides/ 폴더: slide_extractor.py 출력 디렉토리
#     ├── slide_001_base.jpg   ← base 이미지만 대상
#     ├── slide_001_annot_01.jpg 
#     ├── metadata.json
#     └── ...

# Output:
#   - ./output/slide_textualized.json: 텍스트화 결과

# 추출 항목:
#   - t1          : 슬라이드 원본 텍스트 (Gemini Vision)
#   - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

# Usage:
#     # 기본값으로 실행
#     python slide_textualizer.py

#     # 경로 직접 지정
#     python slide_textualizer.py -s ./output_slides -o ./output

#     # 재시도 횟수 늘리기 (API 불안정할 때)
#     python slide_textualizer.py --retries 5
# """

# import os
# import re
# import json
# import logging
# import time
# from pathlib import Path
# from typing import Dict, List, Optional
# from dataclasses import dataclass
# from PIL import Image

# from google import genai
# from google.genai import types

# try:
#     from json_repair import repair_json
#     JSON_REPAIR_AVAILABLE = True
# except ImportError:
#     JSON_REPAIR_AVAILABLE = False

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================ #
# #  설정                                                                         #
# # ============================================================================ #

# @dataclass
# class Config:
#     google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
#     slides_dir: Path = Path("./output_slides")   # slide_extractor.py 출력 디렉토리
#     output_dir: Path = Path("./output")
#     gemini_model: str = "models/gemini-2.5-flash"
#     max_retries: int = 3
#     retry_delay: float = 5.0

#     def __post_init__(self):
#         self.output_dir.mkdir(parents=True, exist_ok=True)


# # t1 추출용 프롬프트
# T1_EXTRACTION_PROMPT = """
# 이 슬라이드 이미지에서 보이는 모든 텍스트와 구조 정보를 추출하라.
# 설명 없이 JSON만 출력.

# 출력 형식:
# {
#   "title": "슬라이드 제목",
#   "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
#   "structure": "다이어그램/표/화살표 관계를 텍스트로 기술"
# }

# 추출 규칙:
# - 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
# - 원문 그대로 추출 (요약하지 말 것)
# - 불릿 포인트는 "- " 또는 "• "로 시작
# - 줄바꿈은 \\n으로 표시
# - structure 필드: 다이어그램이 없으면 빈 문자열
#   예시) "사용자 → 응용소프트웨어 → 운영체제 → 컴퓨터 하드웨어 (위에서 아래 계층 구조)"
#   예시) "운영체제 vs 응용소프트웨어 비교표: 목적(자원관리 vs 사용자목적), 개발언어(C/C++ vs 다양)"
# """


# # ============================================================================ #
# #  슬라이드 로더                                                                 #
# # ============================================================================ #

# class SlideLoader:
#     """
#     slide_extractor.py 출력 디렉토리에서 base 이미지만 로드.
#     타임스탬프는 metadata.json에서 읽고, 없으면 파일명 패턴으로 폴백.
#     """

#     def __init__(self, slides_dir: Path):
#         self.slides_dir = Path(slides_dir)

#     def load(self) -> List[Dict]:
#         metadata_path = self.slides_dir / "metadata.json"

#         if metadata_path.exists():
#             return self._load_from_metadata(metadata_path)
#         else:
#             logger.warning("metadata.json 없음 - 파일명 패턴으로 폴백")
#             return self._load_from_filenames()

#     def _load_from_metadata(self, metadata_path: Path) -> List[Dict]:
#         """metadata.json 기준으로 base 이미지와 타임스탬프 로드"""
#         with open(metadata_path, encoding="utf-8") as f:
#             metadata = json.load(f)

#         slides = []
#         for entry in metadata:
#             if entry.get("capture_type") != "base":
#                 continue

#             image_path = self.slides_dir / entry["filename"]
#             if not image_path.exists():
#                 logger.warning(f"파일 없음: {image_path}")
#                 continue

#             slides.append({
#                 "slide_number": entry["slide_index"],
#                 "timestamp":    entry.get("timestamp_sec", 0.0),
#                 "image_path":   str(image_path),
#                 "image":        Image.open(image_path).convert("RGB"),
#             })

#         slides.sort(key=lambda x: x["slide_number"])
#         logger.info(f"✓ Loaded {len(slides)} base slides from metadata.json")
#         return slides

#     def _load_from_filenames(self) -> List[Dict]:
#         """metadata 없을 때 slide_NNN_base.jpg 패턴으로 폴백"""
#         pattern = re.compile(r'slide_(\d+)_base\.(jpg|png)', re.IGNORECASE)
#         slides = []

#         for file in sorted(self.slides_dir.iterdir()):
#             match = pattern.match(file.name)
#             if match:
#                 slides.append({
#                     "slide_number": int(match.group(1)),
#                     "timestamp":    0.0,
#                     "image_path":   str(file),
#                     "image":        Image.open(file).convert("RGB"),
#                 })

#         logger.info(f"✓ Loaded {len(slides)} base slides from filenames")
#         return slides


# # ============================================================================ #
# #  t1 추출기 (Gemini Vision)                                                    #
# # ============================================================================ #

# class T1Extractor:
#     """슬라이드 base 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

#     def __init__(self, config: Config):
#         self.config = config
#         self.client = genai.Client(api_key=config.google_api_key)
#         logger.info("✓ Gemini initialized for t1 extraction")

#     def _call_gemini(self, image: Image.Image) -> str:
#         """재시도 로직 포함 Gemini Vision 호출"""
#         last_exc = None
#         for attempt in range(self.config.max_retries):
#             try:
#                 response = self.client.models.generate_content(
#                     model=self.config.gemini_model,
#                     contents=[T1_EXTRACTION_PROMPT, image],
#                     config=types.GenerateContentConfig(
#                         response_mime_type="application/json"
#                     )
#                 )
#                 return response.text
#             except Exception as e:
#                 last_exc = e
#                 logger.warning(
#                     f"  ⚠ Gemini call failed "
#                     f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
#                 )
#                 if attempt < self.config.max_retries - 1:
#                     time.sleep(self.config.retry_delay * (attempt + 1))
#         raise last_exc

#     def extract(self, slide: Dict) -> Dict:
#         """단일 슬라이드에서 t1, t1_structure 추출"""
#         slide.setdefault("title", f"Slide {slide['slide_number']}")
#         slide.setdefault("t1", "")
#         slide.setdefault("t1_structure", "")

#         try:
#             raw_text = self._call_gemini(slide["image"])

#             # 코드펜스 제거
#             if "```json" in raw_text:
#                 raw_text = raw_text.split("```json")[1].split("```")[0]
#             elif "```" in raw_text:
#                 raw_text = raw_text.split("```")[1].split("```")[0]
#             raw_text = raw_text.strip()

#             # 문자열 값 내부의 제어문자 정제
#             def clean_string_value(m):
#                 inner = m.group(1)
#                 inner = re.sub(r'[\n\r\t]', ' ', inner)
#                 inner = re.sub(r' {2,}', ' ', inner).strip()
#                 return f'"{inner}"'
#             raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

#             # 파싱 → json_repair fallback
#             try:
#                 result = json.loads(raw_text)
#             except json.JSONDecodeError:
#                 if JSON_REPAIR_AVAILABLE:
#                     result = json.loads(repair_json(raw_text))
#                 else:
#                     raise

#             slide["title"]        = result.get("title", f"Slide {slide['slide_number']}")
#             slide["t1"]           = result.get("raw_text", "")
#             slide["t1_structure"] = result.get("structure", "")

#         except Exception as e:
#             logger.error(
#                 f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
#             )

#         return slide

#     def extract_batch(self, slides: List[Dict]) -> List[Dict]:
#         logger.info(f"Extracting t1 from {len(slides)} slides...")

#         for i, slide in enumerate(slides):
#             self.extract(slide)
#             logger.info(
#                 f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']}: "
#                 f"t1={len(slide['t1'])} chars, "
#                 f"structure={len(slide['t1_structure'])} chars"
#             )

#         logger.info("✓ t1 extraction complete")
#         return slides


# # ============================================================================ #
# #  파이프라인                                                                    #
# # ============================================================================ #

# class TextualizationPipeline:
#     """슬라이드 텍스트화 파이프라인"""

#     def __init__(self, config: Config = None):
#         self.config = config or Config()

#     def run(self) -> Dict:
#         start_time = time.time()

#         print("\n" + "="*70)
#         print("🎓 슬라이드 텍스트화 파이프라인 (Stage 1)")
#         print("="*70)
#         print(f"📁 Slides : {self.config.slides_dir}")
#         print(f"📂 Output : {self.config.output_dir}")

#         # Stage 1: base 슬라이드 로드
#         print("\n" + "-"*70)
#         print("Stage 1: base 슬라이드 로드")
#         print("-"*70)

#         slides = SlideLoader(self.config.slides_dir).load()

#         # Stage 2: t1 추출 (텍스트 + 구조)
#         print("\n" + "-"*70)
#         print("Stage 2: 텍스트화 (Gemini Vision)")
#         print("-"*70)

#         slides = T1Extractor(self.config).extract_batch(slides)

#         # Stage 3: 결과 저장
#         print("\n" + "-"*70)
#         print("Stage 3: 결과 저장")
#         print("-"*70)

#         for slide in slides:
#             ts = slide["timestamp"]
#             mins, secs = divmod(ts, 60)
#             hrs, mins = divmod(mins, 60)
#             slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
#             slide["slide_id"] = f"slide_{slide['slide_number']:03d}"

#         total_time = time.time() - start_time

#         result = {
#             "metadata": {
#                 "slides_dir":      str(self.config.slides_dir),
#                 "processing_time": total_time,
#                 "total_slides":    len(slides),
#             },
#             "slides": [
#                 {
#                     "slide_id":            s["slide_id"],
#                     "slide_number":        s["slide_number"],
#                     "timestamp":           s["timestamp"],
#                     "timestamp_formatted": s["timestamp_formatted"],
#                     "image_path":          s["image_path"],
#                     "title":               s["title"],
#                     "t1":                  s["t1"],
#                     "t1_structure":        s["t1_structure"],
#                 }
#                 for s in slides
#             ]
#         }

#         output_path = self.config.output_dir / "slide_textualized.json"
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)
#         logger.info(f"✓ Saved: {output_path}")

#         structure_count = sum(1 for s in slides if s.get("t1_structure"))

#         print("\n" + "="*70)
#         print("✅ 텍스트화 완료!")
#         print("="*70)
#         print(f"\n📊 결과:")
#         print(f"  • 슬라이드:      {len(slides)}개")
#         print(f"  • t1 추출:       {sum(1 for s in slides if s['t1'])}개")
#         print(f"  • t1_structure:  {structure_count}개")
#         print(f"\n📁 생성된 파일:")
#         print(f"  • {output_path}")
#         print(f"\n⏱️  처리 시간: {total_time:.2f}초")

#         return result


# # ============================================================================ #
# #  메인                                                                         #
# # ============================================================================ #

# def main():
#     import argparse

#     parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
#     parser.add_argument("-s", "--slides", default="./output_slides",
#                         help="slide_extractor.py 출력 디렉토리 (default: ./output_slides)")
#     parser.add_argument("-o", "--output", default="./output",
#                         help="결과 저장 디렉토리 (default: ./output)")
#     parser.add_argument("--retries", type=int, default=3,
#                         help="Gemini API 재시도 횟수 (default: 3)")

#     args = parser.parse_args()

#     config = Config(
#         slides_dir=Path(args.slides),
#         output_dir=Path(args.output),
#         max_retries=args.retries,
#     )

#     if not config.slides_dir.exists():
#         print(f"❌ Slides folder not found: {config.slides_dir}")
#         return

#     TextualizationPipeline(config).run()


# if __name__ == "__main__":
#     main()

# """
# 슬라이드 텍스트화 파이프라인 (Stage 1)

# 슬라이드 이미지 내의 텍스트, 다이어그램, 표 등 시각 요소를
# Gemini Vision을 통해 텍스트로 변환합니다.

# Input:
#   - output_slides/ 폴더: slide_extractor.py 출력 디렉토리
#     ├── slide_001_base.jpg   ← base 이미지만 대상
#     ├── slide_001_annot_01.jpg 
#     ├── metadata.json
#     └── ...

# Output:
#   - ./output/slide_textualized.json: 텍스트화 결과

# 추출 항목:
#   - t1          : 슬라이드 원본 텍스트 (Gemini Vision)
#   - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

# Usage:
#     # 기본값으로 실행
#     python slide_textualizer.py

#     # 경로 직접 지정
#     python slide_textualizer.py -s ./output_slides -o ./output

#     # 재시도 횟수 늘리기 (API 불안정할 때)
#     python slide_textualizer.py --retries 5
# """

# import os
# import re
# import json
# import logging
# import time
# from pathlib import Path
# from typing import Dict, List, Optional
# from dataclasses import dataclass
# from PIL import Image

# from google import genai
# from google.genai import types

# try:
#     from json_repair import repair_json
#     JSON_REPAIR_AVAILABLE = True
# except ImportError:
#     JSON_REPAIR_AVAILABLE = False

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================ #
# #  설정                                                                         #
# # ============================================================================ #

# @dataclass
# class Config:
#     google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
#     slides_dir: Path = Path("./output_slides")   # slide_extractor.py 출력 디렉토리
#     output_dir: Path = Path("./output")
#     gemini_model: str = "models/gemini-2.5-flash"
#     max_retries: int = 3
#     retry_delay: float = 5.0

#     def __post_init__(self):
#         self.output_dir.mkdir(parents=True, exist_ok=True)


# # t1 추출용 프롬프트
# T1_EXTRACTION_PROMPT = """
# 이 슬라이드 이미지에서 보이는 모든 텍스트와 구조 정보를 추출하라.
# 설명 없이 JSON만 출력.

# 출력 형식:
# {
#   "title": "슬라이드 제목",
#   "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
#   "structure": "다이어그램/표/화살표 관계를 텍스트로 기술"
# }

# 추출 규칙:
# - 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
# - 원문 그대로 추출 (요약하지 말 것)
# - 불릿 포인트는 "- " 또는 "• "로 시작
# - 줄바꿈은 \\n으로 표시
# - structure 필드: 다이어그램이 없으면 빈 문자열
#   예시) "사용자 → 응용소프트웨어 → 운영체제 → 컴퓨터 하드웨어 (위에서 아래 계층 구조)"
#   예시) "운영체제 vs 응용소프트웨어 비교표: 목적(자원관리 vs 사용자목적), 개발언어(C/C++ vs 다양)"
# """


# # ============================================================================ #
# #  슬라이드 로더                                                                 #
# # ============================================================================ #

# class SlideLoader:
#     """
#     slide_extractor.py 출력 디렉토리에서 base 이미지만 로드.
#     타임스탬프는 metadata.json에서 읽고, 없으면 파일명 패턴으로 폴백.
#     """

#     def __init__(self, slides_dir: Path):
#         self.slides_dir = Path(slides_dir)

#     def load(self) -> List[Dict]:
#         metadata_path = self.slides_dir / "metadata.json"

#         if metadata_path.exists():
#             return self._load_from_metadata(metadata_path)
#         else:
#             logger.warning("metadata.json 없음 - 파일명 패턴으로 폴백")
#             return self._load_from_filenames()

#     def _load_from_metadata(self, metadata_path: Path) -> List[Dict]:
#         """metadata.json 기준으로 base 이미지와 타임스탬프 로드"""
#         with open(metadata_path, encoding="utf-8") as f:
#             metadata = json.load(f)

#         slides = []
#         for entry in metadata:
#             if entry.get("capture_type") != "base":
#                 continue

#             image_path = self.slides_dir / entry["filename"]
#             if not image_path.exists():
#                 logger.warning(f"파일 없음: {image_path}")
#                 continue

#             slides.append({
#                 "slide_number": entry["slide_index"],
#                 "timestamp":    entry.get("timestamp_sec", 0.0),
#                 "image_path":   str(image_path),
#                 "image":        Image.open(image_path).convert("RGB"),
#             })

#         slides.sort(key=lambda x: x["slide_number"])
#         logger.info(f"✓ Loaded {len(slides)} base slides from metadata.json")
#         return slides

#     def _load_from_filenames(self) -> List[Dict]:
#         """metadata 없을 때 slide_NNN_base.jpg 패턴으로 폴백"""
#         pattern = re.compile(r'slide_(\d+)_base\.(jpg|png)', re.IGNORECASE)
#         slides = []

#         for file in sorted(self.slides_dir.iterdir()):
#             match = pattern.match(file.name)
#             if match:
#                 slides.append({
#                     "slide_number": int(match.group(1)),
#                     "timestamp":    0.0,
#                     "image_path":   str(file),
#                     "image":        Image.open(file).convert("RGB"),
#                 })

#         logger.info(f"✓ Loaded {len(slides)} base slides from filenames")
#         return slides


# # ============================================================================ #
# #  t1 추출기 (Gemini Vision)                                                    #
# # ============================================================================ #

# class T1Extractor:
#     """슬라이드 base 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

#     def __init__(self, config: Config):
#         self.config = config
#         self.client = genai.Client(api_key=config.google_api_key)
#         logger.info("✓ Gemini initialized for t1 extraction")

#     def _call_gemini(self, image: Image.Image) -> str:
#         """재시도 로직 포함 Gemini Vision 호출"""
#         last_exc = None
#         for attempt in range(self.config.max_retries):
#             try:
#                 response = self.client.models.generate_content(
#                     model=self.config.gemini_model,
#                     contents=[T1_EXTRACTION_PROMPT, image],
#                     config=types.GenerateContentConfig(
#                         response_mime_type="application/json"
#                     )
#                 )
#                 return response.text
#             except Exception as e:
#                 last_exc = e
#                 logger.warning(
#                     f"  ⚠ Gemini call failed "
#                     f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
#                 )
#                 if attempt < self.config.max_retries - 1:
#                     time.sleep(self.config.retry_delay * (attempt + 1))
#         raise last_exc

#     def extract(self, slide: Dict) -> Dict:
#         """단일 슬라이드에서 t1, t1_structure 추출"""
#         slide.setdefault("title", f"Slide {slide['slide_number']}")
#         slide.setdefault("t1", "")
#         slide.setdefault("t1_structure", "")

#         try:
#             raw_text = self._call_gemini(slide["image"])

#             # 코드펜스 제거
#             if "```json" in raw_text:
#                 raw_text = raw_text.split("```json")[1].split("```")[0]
#             elif "```" in raw_text:
#                 raw_text = raw_text.split("```")[1].split("```")[0]
#             raw_text = raw_text.strip()

#             # 문자열 값 내부의 제어문자 정제
#             def clean_string_value(m):
#                 inner = m.group(1)
#                 inner = re.sub(r'[\n\r\t]', ' ', inner)
#                 inner = re.sub(r' {2,}', ' ', inner).strip()
#                 return f'"{inner}"'
#             raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

#             # 파싱 → json_repair fallback
#             try:
#                 result = json.loads(raw_text)
#             except json.JSONDecodeError:
#                 if JSON_REPAIR_AVAILABLE:
#                     result = json.loads(repair_json(raw_text))
#                 else:
#                     raise

#             slide["title"]        = result.get("title", f"Slide {slide['slide_number']}")
#             slide["t1"]           = result.get("raw_text", "")
#             slide["t1_structure"] = result.get("structure", "")

#         except Exception as e:
#             logger.error(
#                 f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
#             )

#         return slide

#     def extract_batch(self, slides: List[Dict]) -> List[Dict]:
#         logger.info(f"Extracting t1 from {len(slides)} slides...")

#         for i, slide in enumerate(slides):
#             self.extract(slide)
#             logger.info(
#                 f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']}: "
#                 f"t1={len(slide['t1'])} chars, "
#                 f"structure={len(slide['t1_structure'])} chars"
#             )

#         logger.info("✓ t1 extraction complete")
#         return slides


# # ============================================================================ #
# #  파이프라인                                                                    #
# # ============================================================================ #

# class TextualizationPipeline:
#     """슬라이드 텍스트화 파이프라인"""

#     def __init__(self, config: Config = None):
#         self.config = config or Config()

#     def run(self) -> Dict:
#         start_time = time.time()

#         print("\n" + "="*70)
#         print("🎓 슬라이드 텍스트화 파이프라인 (Stage 1)")
#         print("="*70)
#         print(f"📁 Slides : {self.config.slides_dir}")
#         print(f"📂 Output : {self.config.output_dir}")

#         # Stage 1: base 슬라이드 로드
#         print("\n" + "-"*70)
#         print("Stage 1: base 슬라이드 로드")
#         print("-"*70)

#         slides = SlideLoader(self.config.slides_dir).load()

#         # Stage 2: t1 추출 (텍스트 + 구조)
#         print("\n" + "-"*70)
#         print("Stage 2: 텍스트화 (Gemini Vision)")
#         print("-"*70)

#         slides = T1Extractor(self.config).extract_batch(slides)

#         # Stage 3: 결과 저장
#         print("\n" + "-"*70)
#         print("Stage 3: 결과 저장")
#         print("-"*70)

#         for slide in slides:
#             ts = slide["timestamp"]
#             mins, secs = divmod(ts, 60)
#             hrs, mins = divmod(mins, 60)
#             slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
#             slide["slide_id"] = f"slide_{slide['slide_number']:03d}"

#         total_time = time.time() - start_time

#         result = {
#             "metadata": {
#                 "slides_dir":      str(self.config.slides_dir),
#                 "processing_time": total_time,
#                 "total_slides":    len(slides),
#             },
#             "slides": [
#                 {
#                     "slide_id":            s["slide_id"],
#                     "slide_number":        s["slide_number"],
#                     "timestamp":           s["timestamp"],
#                     "timestamp_formatted": s["timestamp_formatted"],
#                     "image_path":          s["image_path"],
#                     "title":               s["title"],
#                     "t1":                  s["t1"],
#                     "t1_structure":        s["t1_structure"],
#                 }
#                 for s in slides
#             ]
#         }

#         output_path = self.config.output_dir / "slide_textualized.json"
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)
#         logger.info(f"✓ Saved: {output_path}")

#         structure_count = sum(1 for s in slides if s.get("t1_structure"))

#         print("\n" + "="*70)
#         print("✅ 텍스트화 완료!")
#         print("="*70)
#         print(f"\n📊 결과:")
#         print(f"  • 슬라이드:      {len(slides)}개")
#         print(f"  • t1 추출:       {sum(1 for s in slides if s['t1'])}개")
#         print(f"  • t1_structure:  {structure_count}개")
#         print(f"\n📁 생성된 파일:")
#         print(f"  • {output_path}")
#         print(f"\n⏱️  처리 시간: {total_time:.2f}초")

#         return result


# # ============================================================================ #
# #  메인                                                                         #
# # ============================================================================ #

# def main():
#     import argparse

#     parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
#     parser.add_argument("-s", "--slides", default="./output_slides",
#                         help="slide_extractor.py 출력 디렉토리 (default: ./output_slides)")
#     parser.add_argument("-o", "--output", default="./output",
#                         help="결과 저장 디렉토리 (default: ./output)")
#     parser.add_argument("--retries", type=int, default=3,
#                         help="Gemini API 재시도 횟수 (default: 3)")

#     args = parser.parse_args()

#     config = Config(
#         slides_dir=Path(args.slides),
#         output_dir=Path(args.output),
#         max_retries=args.retries,
#     )

#     if not config.slides_dir.exists():
#         print(f"❌ Slides folder not found: {config.slides_dir}")
#         return

#     TextualizationPipeline(config).run()


# if __name__ == "__main__":
#     main()

# """
# 슬라이드 텍스트화 파이프라인 (Stage 1)

# 슬라이드 이미지 내의 텍스트, 다이어그램, 표 등 시각 요소를
# Gemini Vision을 통해 텍스트로 변환합니다.

# Input:
#   - output_slides/ 폴더: slide_extractor.py 출력 디렉토리
#     ├── slide_001_base.jpg   ← base 이미지만 대상
#     ├── slide_001_annot_01.jpg 
#     ├── metadata.json
#     └── ...

# Output:
#   - ./output/slide_textualized.json: 텍스트화 결과

# 추출 항목:
#   - t1          : 슬라이드 원본 텍스트 (Gemini Vision)
#   - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

# Usage:
#     # 기본값으로 실행
#     python slide_textualizer.py

#     # 경로 직접 지정
#     python slide_textualizer.py -s ./output_slides -o ./output

#     # 재시도 횟수 늘리기 (API 불안정할 때)
#     python slide_textualizer.py --retries 5
# """

# import os
# import re
# import json
# import logging
# import time
# from pathlib import Path
# from typing import Dict, List, Optional
# from dataclasses import dataclass
# from PIL import Image

# from google import genai
# from google.genai import types

# try:
#     from json_repair import repair_json
#     JSON_REPAIR_AVAILABLE = True
# except ImportError:
#     JSON_REPAIR_AVAILABLE = False

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================ #
# #  설정                                                                         #
# # ============================================================================ #

# @dataclass
# class Config:
#     google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
#     slides_dir: Path = Path("./output_slides")   # slide_extractor.py 출력 디렉토리
#     output_dir: Path = Path("./output")
#     gemini_model: str = "models/gemini-2.5-flash"
#     max_retries: int = 3
#     retry_delay: float = 5.0

#     def __post_init__(self):
#         self.output_dir.mkdir(parents=True, exist_ok=True)


# # t1 추출용 프롬프트
# T1_EXTRACTION_PROMPT = """
# 이 슬라이드 이미지에서 보이는 모든 텍스트와 구조 정보를 추출하라.
# 설명 없이 JSON만 출력.

# 출력 형식:
# {
#   "title": "슬라이드 제목",
#   "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
#   "structure": "다이어그램/표/화살표 관계를 텍스트로 기술"
# }

# 추출 규칙:
# - 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
# - 원문 그대로 추출 (요약하지 말 것)
# - 불릿 포인트는 "- " 또는 "• "로 시작
# - 줄바꿈은 \\n으로 표시
# - structure 필드: 다이어그램이 없으면 빈 문자열
#   예시) "사용자 → 응용소프트웨어 → 운영체제 → 컴퓨터 하드웨어 (위에서 아래 계층 구조)"
#   예시) "운영체제 vs 응용소프트웨어 비교표: 목적(자원관리 vs 사용자목적), 개발언어(C/C++ vs 다양)"
# """


# # ============================================================================ #
# #  슬라이드 로더                                                                 #
# # ============================================================================ #

# class SlideLoader:
#     """
#     slide_extractor.py 출력 디렉토리에서 base 이미지만 로드.
#     타임스탬프는 metadata.json에서 읽고, 없으면 파일명 패턴으로 폴백.
#     """

#     def __init__(self, slides_dir: Path):
#         self.slides_dir = Path(slides_dir)

#     def load(self) -> List[Dict]:
#         metadata_path = self.slides_dir / "metadata.json"

#         if metadata_path.exists():
#             return self._load_from_metadata(metadata_path)
#         else:
#             logger.warning("metadata.json 없음 - 파일명 패턴으로 폴백")
#             return self._load_from_filenames()

#     def _load_from_metadata(self, metadata_path: Path) -> List[Dict]:
#         """metadata.json 기준으로 base 이미지와 타임스탬프 로드"""
#         with open(metadata_path, encoding="utf-8") as f:
#             metadata = json.load(f)

#         slides = []
#         for entry in metadata:
#             if entry.get("capture_type") != "base":
#                 continue

#             image_path = self.slides_dir / entry["filename"]
#             if not image_path.exists():
#                 logger.warning(f"파일 없음: {image_path}")
#                 continue

#             slides.append({
#                 "slide_number": entry["slide_index"],
#                 "timestamp":    entry.get("timestamp_sec", 0.0),
#                 "image_path":   str(image_path),
#                 "image":        Image.open(image_path).convert("RGB"),
#             })

#         slides.sort(key=lambda x: x["slide_number"])
#         logger.info(f"✓ Loaded {len(slides)} base slides from metadata.json")
#         return slides

#     def _load_from_filenames(self) -> List[Dict]:
#         """metadata 없을 때 slide_NNN_base.jpg 패턴으로 폴백"""
#         pattern = re.compile(r'slide_(\d+)_base\.(jpg|png)', re.IGNORECASE)
#         slides = []

#         for file in sorted(self.slides_dir.iterdir()):
#             match = pattern.match(file.name)
#             if match:
#                 slides.append({
#                     "slide_number": int(match.group(1)),
#                     "timestamp":    0.0,
#                     "image_path":   str(file),
#                     "image":        Image.open(file).convert("RGB"),
#                 })

#         logger.info(f"✓ Loaded {len(slides)} base slides from filenames")
#         return slides


# # ============================================================================ #
# #  t1 추출기 (Gemini Vision)                                                    #
# # ============================================================================ #

# class T1Extractor:
#     """슬라이드 base 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

#     def __init__(self, config: Config):
#         self.config = config
#         self.client = genai.Client(api_key=config.google_api_key)
#         logger.info("✓ Gemini initialized for t1 extraction")

#     def _call_gemini(self, image: Image.Image) -> str:
#         """재시도 로직 포함 Gemini Vision 호출"""
#         last_exc = None
#         for attempt in range(self.config.max_retries):
#             try:
#                 response = self.client.models.generate_content(
#                     model=self.config.gemini_model,
#                     contents=[T1_EXTRACTION_PROMPT, image],
#                     config=types.GenerateContentConfig(
#                         response_mime_type="application/json"
#                     )
#                 )
#                 return response.text
#             except Exception as e:
#                 last_exc = e
#                 logger.warning(
#                     f"  ⚠ Gemini call failed "
#                     f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
#                 )
#                 if attempt < self.config.max_retries - 1:
#                     time.sleep(self.config.retry_delay * (attempt + 1))
#         raise last_exc

#     def extract(self, slide: Dict) -> Dict:
#         """단일 슬라이드에서 t1, t1_structure 추출"""
#         slide.setdefault("title", f"Slide {slide['slide_number']}")
#         slide.setdefault("t1", "")
#         slide.setdefault("t1_structure", "")

#         try:
#             raw_text = self._call_gemini(slide["image"])

#             # 코드펜스 제거
#             if "```json" in raw_text:
#                 raw_text = raw_text.split("```json")[1].split("```")[0]
#             elif "```" in raw_text:
#                 raw_text = raw_text.split("```")[1].split("```")[0]
#             raw_text = raw_text.strip()

#             # 문자열 값 내부의 제어문자 정제
#             def clean_string_value(m):
#                 inner = m.group(1)
#                 inner = re.sub(r'[\n\r\t]', ' ', inner)
#                 inner = re.sub(r' {2,}', ' ', inner).strip()
#                 return f'"{inner}"'
#             raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

#             # 파싱 → json_repair fallback
#             try:
#                 result = json.loads(raw_text)
#             except json.JSONDecodeError:
#                 if JSON_REPAIR_AVAILABLE:
#                     result = json.loads(repair_json(raw_text))
#                 else:
#                     raise

#             slide["title"]        = result.get("title", f"Slide {slide['slide_number']}")
#             slide["t1"]           = result.get("raw_text", "")
#             slide["t1_structure"] = result.get("structure", "")

#         except Exception as e:
#             logger.error(
#                 f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
#             )

#         return slide

#     def extract_batch(self, slides: List[Dict]) -> List[Dict]:
#         logger.info(f"Extracting t1 from {len(slides)} slides...")

#         for i, slide in enumerate(slides):
#             self.extract(slide)
#             logger.info(
#                 f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']}: "
#                 f"t1={len(slide['t1'])} chars, "
#                 f"structure={len(slide['t1_structure'])} chars"
#             )

#         logger.info("✓ t1 extraction complete")
#         return slides


# # ============================================================================ #
# #  파이프라인                                                                    #
# # ============================================================================ #

# class TextualizationPipeline:
#     """슬라이드 텍스트화 파이프라인"""

#     def __init__(self, config: Config = None):
#         self.config = config or Config()

#     def run(self) -> Dict:
#         start_time = time.time()

#         print("\n" + "="*70)
#         print("🎓 슬라이드 텍스트화 파이프라인 (Stage 1)")
#         print("="*70)
#         print(f"📁 Slides : {self.config.slides_dir}")
#         print(f"📂 Output : {self.config.output_dir}")

#         # Stage 1: base 슬라이드 로드
#         print("\n" + "-"*70)
#         print("Stage 1: base 슬라이드 로드")
#         print("-"*70)

#         slides = SlideLoader(self.config.slides_dir).load()

#         # Stage 2: t1 추출 (텍스트 + 구조)
#         print("\n" + "-"*70)
#         print("Stage 2: 텍스트화 (Gemini Vision)")
#         print("-"*70)

#         slides = T1Extractor(self.config).extract_batch(slides)

#         # Stage 3: 결과 저장
#         print("\n" + "-"*70)
#         print("Stage 3: 결과 저장")
#         print("-"*70)

#         for slide in slides:
#             ts = slide["timestamp"]
#             mins, secs = divmod(ts, 60)
#             hrs, mins = divmod(mins, 60)
#             slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
#             slide["slide_id"] = f"slide_{slide['slide_number']:03d}"

#         total_time = time.time() - start_time

#         result = {
#             "metadata": {
#                 "slides_dir":      str(self.config.slides_dir),
#                 "processing_time": total_time,
#                 "total_slides":    len(slides),
#             },
#             "slides": [
#                 {
#                     "slide_id":            s["slide_id"],
#                     "slide_number":        s["slide_number"],
#                     "timestamp":           s["timestamp"],
#                     "timestamp_formatted": s["timestamp_formatted"],
#                     "image_path":          s["image_path"],
#                     "title":               s["title"],
#                     "t1":                  s["t1"],
#                     "t1_structure":        s["t1_structure"],
#                 }
#                 for s in slides
#             ]
#         }

#         output_path = self.config.output_dir / "slide_textualized.json"
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)
#         logger.info(f"✓ Saved: {output_path}")

#         structure_count = sum(1 for s in slides if s.get("t1_structure"))

#         print("\n" + "="*70)
#         print("✅ 텍스트화 완료!")
#         print("="*70)
#         print(f"\n📊 결과:")
#         print(f"  • 슬라이드:      {len(slides)}개")
#         print(f"  • t1 추출:       {sum(1 for s in slides if s['t1'])}개")
#         print(f"  • t1_structure:  {structure_count}개")
#         print(f"\n📁 생성된 파일:")
#         print(f"  • {output_path}")
#         print(f"\n⏱️  처리 시간: {total_time:.2f}초")

#         return result


# # ============================================================================ #
# #  메인                                                                         #
# # ============================================================================ #

# def main():
#     import argparse

#     parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
#     parser.add_argument("-s", "--slides", default="./output_slides",
#                         help="slide_extractor.py 출력 디렉토리 (default: ./output_slides)")
#     parser.add_argument("-o", "--output", default="./output",
#                         help="결과 저장 디렉토리 (default: ./output)")
#     parser.add_argument("--retries", type=int, default=3,
#                         help="Gemini API 재시도 횟수 (default: 3)")

#     args = parser.parse_args()

#     config = Config(
#         slides_dir=Path(args.slides),
#         output_dir=Path(args.output),
#         max_retries=args.retries,
#     )

#     if not config.slides_dir.exists():
#         print(f"❌ Slides folder not found: {config.slides_dir}")
#         return

#     TextualizationPipeline(config).run()


# if __name__ == "__main__":
#     main()

# """
# 슬라이드 텍스트화 파이프라인 (Stage 1)

# 슬라이드 이미지 내의 텍스트, 다이어그램, 표 등 시각 요소를
# Gemini Vision을 통해 텍스트로 변환합니다.

# Input:
#   - output_slides/ 폴더: slide_extractor.py 출력 디렉토리
#     ├── slide_001_base.jpg   ← base 이미지만 대상
#     ├── slide_001_annot_01.jpg 
#     ├── metadata.json
#     └── ...

# Output:
#   - ./output/slide_textualized.json: 텍스트화 결과

# 추출 항목:
#   - t1          : 슬라이드 원본 텍스트 (Gemini Vision)
#   - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

# Usage:
#     # 기본값으로 실행
#     python slide_textualizer.py

#     # 경로 직접 지정
#     python slide_textualizer.py -s ./output_slides -o ./output

#     # 재시도 횟수 늘리기 (API 불안정할 때)
#     python slide_textualizer.py --retries 5
# """

# import os
# import re
# import json
# import logging
# import time
# from pathlib import Path
# from typing import Dict, List, Optional
# from dataclasses import dataclass
# from PIL import Image

# from google import genai
# from google.genai import types

# try:
#     from json_repair import repair_json
#     JSON_REPAIR_AVAILABLE = True
# except ImportError:
#     JSON_REPAIR_AVAILABLE = False

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================ #
# #  설정                                                                         #
# # ============================================================================ #

# @dataclass
# class Config:
#     google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
#     slides_dir: Path = Path("./output_slides")   # slide_extractor.py 출력 디렉토리
#     output_dir: Path = Path("./output")
#     gemini_model: str = "models/gemini-2.5-flash"
#     max_retries: int = 3
#     retry_delay: float = 5.0

#     def __post_init__(self):
#         self.output_dir.mkdir(parents=True, exist_ok=True)


# # t1 추출용 프롬프트
# T1_EXTRACTION_PROMPT = """
# 이 슬라이드 이미지에서 보이는 모든 텍스트와 구조 정보를 추출하라.
# 설명 없이 JSON만 출력.

# 출력 형식:
# {
#   "title": "슬라이드 제목",
#   "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
#   "structure": "다이어그램/표/화살표 관계를 텍스트로 기술"
# }

# 추출 규칙:
# - 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
# - 원문 그대로 추출 (요약하지 말 것)
# - 불릿 포인트는 "- " 또는 "• "로 시작
# - 줄바꿈은 \\n으로 표시
# - structure 필드: 다이어그램이 없으면 빈 문자열
#   예시) "사용자 → 응용소프트웨어 → 운영체제 → 컴퓨터 하드웨어 (위에서 아래 계층 구조)"
#   예시) "운영체제 vs 응용소프트웨어 비교표: 목적(자원관리 vs 사용자목적), 개발언어(C/C++ vs 다양)"
# """


# # ============================================================================ #
# #  슬라이드 로더                                                                 #
# # ============================================================================ #

# class SlideLoader:
#     """
#     slide_extractor.py 출력 디렉토리에서 base 이미지만 로드.
#     타임스탬프는 metadata.json에서 읽고, 없으면 파일명 패턴으로 폴백.
#     """

#     def __init__(self, slides_dir: Path):
#         self.slides_dir = Path(slides_dir)

#     def load(self) -> List[Dict]:
#         metadata_path = self.slides_dir / "metadata.json"

#         if metadata_path.exists():
#             return self._load_from_metadata(metadata_path)
#         else:
#             logger.warning("metadata.json 없음 - 파일명 패턴으로 폴백")
#             return self._load_from_filenames()

#     def _load_from_metadata(self, metadata_path: Path) -> List[Dict]:
#         """metadata.json 기준으로 base 이미지와 타임스탬프 로드"""
#         with open(metadata_path, encoding="utf-8") as f:
#             metadata = json.load(f)

#         slides = []
#         for entry in metadata:
#             if entry.get("capture_type") != "base":
#                 continue

#             image_path = self.slides_dir / entry["filename"]
#             if not image_path.exists():
#                 logger.warning(f"파일 없음: {image_path}")
#                 continue

#             slides.append({
#                 "slide_number": entry["slide_index"],
#                 "timestamp":    entry.get("timestamp_sec", 0.0),
#                 "image_path":   str(image_path),
#                 "image":        Image.open(image_path).convert("RGB"),
#             })

#         slides.sort(key=lambda x: x["slide_number"])
#         logger.info(f"✓ Loaded {len(slides)} base slides from metadata.json")
#         return slides

#     def _load_from_filenames(self) -> List[Dict]:
#         """metadata 없을 때 slide_NNN_base.jpg 패턴으로 폴백"""
#         pattern = re.compile(r'slide_(\d+)_base\.(jpg|png)', re.IGNORECASE)
#         slides = []

#         for file in sorted(self.slides_dir.iterdir()):
#             match = pattern.match(file.name)
#             if match:
#                 slides.append({
#                     "slide_number": int(match.group(1)),
#                     "timestamp":    0.0,
#                     "image_path":   str(file),
#                     "image":        Image.open(file).convert("RGB"),
#                 })

#         logger.info(f"✓ Loaded {len(slides)} base slides from filenames")
#         return slides


# # ============================================================================ #
# #  t1 추출기 (Gemini Vision)                                                    #
# # ============================================================================ #

# class T1Extractor:
#     """슬라이드 base 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

#     def __init__(self, config: Config):
#         self.config = config
#         self.client = genai.Client(api_key=config.google_api_key)
#         logger.info("✓ Gemini initialized for t1 extraction")

#     def _call_gemini(self, image: Image.Image) -> str:
#         """재시도 로직 포함 Gemini Vision 호출"""
#         last_exc = None
#         for attempt in range(self.config.max_retries):
#             try:
#                 response = self.client.models.generate_content(
#                     model=self.config.gemini_model,
#                     contents=[T1_EXTRACTION_PROMPT, image],
#                     config=types.GenerateContentConfig(
#                         response_mime_type="application/json"
#                     )
#                 )
#                 return response.text
#             except Exception as e:
#                 last_exc = e
#                 logger.warning(
#                     f"  ⚠ Gemini call failed "
#                     f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
#                 )
#                 if attempt < self.config.max_retries - 1:
#                     time.sleep(self.config.retry_delay * (attempt + 1))
#         raise last_exc

#     def extract(self, slide: Dict) -> Dict:
#         """단일 슬라이드에서 t1, t1_structure 추출"""
#         slide.setdefault("title", f"Slide {slide['slide_number']}")
#         slide.setdefault("t1", "")
#         slide.setdefault("t1_structure", "")

#         try:
#             raw_text = self._call_gemini(slide["image"])

#             # 코드펜스 제거
#             if "```json" in raw_text:
#                 raw_text = raw_text.split("```json")[1].split("```")[0]
#             elif "```" in raw_text:
#                 raw_text = raw_text.split("```")[1].split("```")[0]
#             raw_text = raw_text.strip()

#             # 문자열 값 내부의 제어문자 정제
#             def clean_string_value(m):
#                 inner = m.group(1)
#                 inner = re.sub(r'[\n\r\t]', ' ', inner)
#                 inner = re.sub(r' {2,}', ' ', inner).strip()
#                 return f'"{inner}"'
#             raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

#             # 파싱 → json_repair fallback
#             try:
#                 result = json.loads(raw_text)
#             except json.JSONDecodeError:
#                 if JSON_REPAIR_AVAILABLE:
#                     result = json.loads(repair_json(raw_text))
#                 else:
#                     raise

#             slide["title"]        = result.get("title", f"Slide {slide['slide_number']}")
#             slide["t1"]           = result.get("raw_text", "")
#             slide["t1_structure"] = result.get("structure", "")

#         except Exception as e:
#             logger.error(
#                 f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
#             )

#         return slide

#     def extract_batch(self, slides: List[Dict]) -> List[Dict]:
#         logger.info(f"Extracting t1 from {len(slides)} slides...")

#         for i, slide in enumerate(slides):
#             self.extract(slide)
#             logger.info(
#                 f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']}: "
#                 f"t1={len(slide['t1'])} chars, "
#                 f"structure={len(slide['t1_structure'])} chars"
#             )

#         logger.info("✓ t1 extraction complete")
#         return slides


# # ============================================================================ #
# #  파이프라인                                                                    #
# # ============================================================================ #

# class TextualizationPipeline:
#     """슬라이드 텍스트화 파이프라인"""

#     def __init__(self, config: Config = None):
#         self.config = config or Config()

#     def run(self) -> Dict:
#         start_time = time.time()

#         print("\n" + "="*70)
#         print("🎓 슬라이드 텍스트화 파이프라인 (Stage 1)")
#         print("="*70)
#         print(f"📁 Slides : {self.config.slides_dir}")
#         print(f"📂 Output : {self.config.output_dir}")

#         # Stage 1: base 슬라이드 로드
#         print("\n" + "-"*70)
#         print("Stage 1: base 슬라이드 로드")
#         print("-"*70)

#         slides = SlideLoader(self.config.slides_dir).load()

#         # Stage 2: t1 추출 (텍스트 + 구조)
#         print("\n" + "-"*70)
#         print("Stage 2: 텍스트화 (Gemini Vision)")
#         print("-"*70)

#         slides = T1Extractor(self.config).extract_batch(slides)

#         # Stage 3: 결과 저장
#         print("\n" + "-"*70)
#         print("Stage 3: 결과 저장")
#         print("-"*70)

#         for slide in slides:
#             ts = slide["timestamp"]
#             mins, secs = divmod(ts, 60)
#             hrs, mins = divmod(mins, 60)
#             slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
#             slide["slide_id"] = f"slide_{slide['slide_number']:03d}"

#         total_time = time.time() - start_time

#         result = {
#             "metadata": {
#                 "slides_dir":      str(self.config.slides_dir),
#                 "processing_time": total_time,
#                 "total_slides":    len(slides),
#             },
#             "slides": [
#                 {
#                     "slide_id":            s["slide_id"],
#                     "slide_number":        s["slide_number"],
#                     "timestamp":           s["timestamp"],
#                     "timestamp_formatted": s["timestamp_formatted"],
#                     "image_path":          s["image_path"],
#                     "title":               s["title"],
#                     "t1":                  s["t1"],
#                     "t1_structure":        s["t1_structure"],
#                 }
#                 for s in slides
#             ]
#         }

#         output_path = self.config.output_dir / "slide_textualized.json"
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)
#         logger.info(f"✓ Saved: {output_path}")

#         structure_count = sum(1 for s in slides if s.get("t1_structure"))

#         print("\n" + "="*70)
#         print("✅ 텍스트화 완료!")
#         print("="*70)
#         print(f"\n📊 결과:")
#         print(f"  • 슬라이드:      {len(slides)}개")
#         print(f"  • t1 추출:       {sum(1 for s in slides if s['t1'])}개")
#         print(f"  • t1_structure:  {structure_count}개")
#         print(f"\n📁 생성된 파일:")
#         print(f"  • {output_path}")
#         print(f"\n⏱️  처리 시간: {total_time:.2f}초")

#         return result


# # ============================================================================ #
# #  메인                                                                         #
# # ============================================================================ #

# def main():
#     import argparse

#     parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
#     parser.add_argument("-s", "--slides", default="./output_slides",
#                         help="slide_extractor.py 출력 디렉토리 (default: ./output_slides)")
#     parser.add_argument("-o", "--output", default="./output",
#                         help="결과 저장 디렉토리 (default: ./output)")
#     parser.add_argument("--retries", type=int, default=3,
#                         help="Gemini API 재시도 횟수 (default: 3)")

#     args = parser.parse_args()

#     config = Config(
#         slides_dir=Path(args.slides),
#         output_dir=Path(args.output),
#         max_retries=args.retries,
#     )

#     if not config.slides_dir.exists():
#         print(f"❌ Slides folder not found: {config.slides_dir}")
#         return

#     TextualizationPipeline(config).run()


# if __name__ == "__main__":
#     main()

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
    slides_dir: Path = Path("./output_slides")   # slide_extractor.py 출력 디렉토리
    output_dir: Path = Path("./output")
    gemini_model: str = "models/gemini-2.5-flash"
    max_retries: int = 3
    retry_delay: float = 5.0

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# t1 추출용 프롬프트
T1_EXTRACTION_PROMPT = """
이 슬라이드 이미지에서 슬라이드 유형을 판별하고, 텍스트와 구조 정보를 추출하라.
설명 없이 JSON만 출력.

출력 형식:
{
  "slide_type": "text" | "image_only" | "mixed",
  "title": "슬라이드 제목 (없으면 빈 문자열)",
  "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
  "structure": "다이어그램/표/화살표 관계를 텍스트로 기술"
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
  예시) "19세기 군용 마차와 기병대가 진흙길을 이동하는 장면 (2장의 그림)"
  예시) "벨기에 파리-루베 구간의 돌길(pavé) 사진"

손글씨/필기 처리:
- 슬라이드 위에 교수가 직접 쓴 손글씨, 밑줄, 동그라미, 화살표 등 필기는 모두 무시
- 인쇄된 슬라이드 원본 텍스트만 추출할 것
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

            slides.append({
                "slide_number": slide_num,
                "timestamp":    base.get("timestamp_sec", 0.0),  # 타임스탬프는 항상 base 기준
                "image_path":   str(self.slides_dir / base["filename"]),  # 저장 경로는 base 기록
                "image":        Image.open(image_path).convert("RGB"),
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
        slide.setdefault("title", f"Slide {slide['slide_number']}")
        slide.setdefault("t1", "")
        slide.setdefault("t1_structure", "")

        try:
            raw_text = self._call_gemini(slide["image"])

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

            slide["title"]        = result.get("title", f"Slide {slide['slide_number']}")
            slide["t1"]           = result.get("raw_text", "")
            slide["t1_structure"] = result.get("structure", "")
            slide["slide_type"]   = result.get("slide_type", "text")

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
                f"structure={len(slide['t1_structure'])} chars"
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
                    "text_source":         s.get("text_source", "base"),
                }
                for s in slides
            ]
        }

        output_path = self.config.output_dir / "slide_textualized.json"
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

    parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
    parser.add_argument("-s", "--slides", default="./output_slides",
                        help="slide_extractor.py 출력 디렉토리 (default: ./output_slides)")
    parser.add_argument("-o", "--output", default="./output",
                        help="결과 저장 디렉토리 (default: ./output)")
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