"""
annotation_analyzer.py
=======================
slide_extractor.py의 출력(base + annot 프레임 쌍)을 받아
각 필기/강조가 어떤 콘텐츠를 타겟하는지 Gemini Vision으로 분석합니다.

Input : output_slides/ (slide_extractor.py 출력 디렉토리)
Output: output/annotation_analysis.json

분석 흐름:
  1. base + annot 이미지 쌍 구성
  2. pixel diff로 필기 영역 마스크 생성
  3. Gemini에게 (base, annot, mask) 세 장 전달 → 구조화 JSON 반환
  4. 결과를 knowledge graph Stage 1 (t1_structure) 형식과 호환되도록 저장

Usage:
    python annotation_analyzer.py --slides output_slides/ --output output/annotation_analysis.json
"""

import os
import re
import cv2
import json
import base64
import logging
import argparse
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 없으면 환경변수 직접 설정 필요

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


def _sanitize_raw(raw: str) -> str:
    """Gemini 응답에서 JSON 파싱을 방해하는 요소 제거"""
    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # JSON 문자열 내부의 unescaped 제어문자 제거
    raw = re.sub(r'(?<!\\)\n', ' ', raw)
    raw = re.sub(r'(?<!\\)\r', ' ', raw)
    raw = re.sub(r'(?<!\\)\t', ' ', raw)
    return raw


def _parse_json_safe(raw: str) -> dict:
    """
    1차: 표준 json.loads
    2차: json_repair 라이브러리 (잘린 JSON, 특수문자 복구)
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        result = json.loads(repair_json(raw))
        log.warning("  json_repair로 복구 성공")
        return result
    except Exception as e:
        raise json.JSONDecodeError(f"json_repair도 실패: {e}", raw, 0)


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
class Config:
    DIFF_PIXEL_THRESHOLD  = 20    # 픽셀 diff 절댓값 임계 (그레이스케일)
    DIFF_DILATE_KERNEL    = 15    # 마스크 팽창 커널 크기 (획 주변 여백 확보)
    MASK_MIN_AREA         = 50    # 이보다 작은 연결 컴포넌트는 노이즈로 제거


# ──────────────────────────────────────────────
# 이미지 쌍 로딩
# ──────────────────────────────────────────────
def load_slide_pairs(slides_dir: str) -> list[dict]:
    """
    output_slides/ 디렉토리에서 (base, [annot_01, annot_02, ...]) 쌍을 구성합니다.
    반환 형식:
        [
          {
            "slide_index": 1,
            "base_path": "...",
            "annot_paths": ["...", "..."]
          },
          ...
        ]
    """
    slides_path = Path(slides_dir)
    metadata_path = slides_path / "metadata.json"

    if metadata_path.exists():
        # metadata.json이 있으면 정확히 파싱
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        pairs: dict[int, dict] = {}
        for entry in metadata:
            idx = entry["slide_index"]
            if idx not in pairs:
                pairs[idx] = {"slide_index": idx, "base_path": None, "annot_paths": [], "annot_timestamps": []}

            full_path = str(slides_path / entry["filename"])
            if entry["capture_type"] == "base":
                pairs[idx]["base_path"] = full_path
            else:  # annotation or force_capture
                pairs[idx]["annot_paths"].append(full_path)
                pairs[idx]["annot_timestamps"].append(entry.get("timestamp_sec", 0.0))

        result = [v for v in pairs.values() if v["base_path"] is not None]
        result.sort(key=lambda x: x["slide_index"])
    else:
        # metadata 없을 경우 파일명 패턴으로 폴백
        log.warning("metadata.json 없음 - 파일명 패턴으로 쌍 구성")
        result = _load_pairs_by_filename(slides_path)

    log.info(f"슬라이드 {len(result)}개 로드 완료 (총 annot: {sum(len(p['annot_paths']) for p in result)}개)")
    return result


def _load_pairs_by_filename(slides_path: Path) -> list[dict]:
    bases = sorted(slides_path.glob("slide_*_base.jpg"))
    pairs = []
    for base in bases:
        idx = int(base.name.split("_")[1])
        annots = sorted(slides_path.glob(f"slide_{idx:03d}_annot_*.jpg"))
        pairs.append({
            "slide_index": idx,
            "base_path": str(base),
            "annot_paths": [str(a) for a in annots],
        })
    return pairs


# ──────────────────────────────────────────────
# Diff 마스크 생성
# ──────────────────────────────────────────────
def build_diff_mask(base_path: str, annot_path: str, cfg: Config) -> np.ndarray:
    """
    base와 annot 이미지의 픽셀 차이로 필기 영역 마스크를 만듭니다.

    - 노이즈 제거: 작은 연결 컴포넌트 제거
    - 팽창(dilate): 획 주변 여백을 확보해 타겟 텍스트가 마스크에 걸리도록

    반환: 3채널 BGR 마스크 이미지 (흰색=필기 영역, 검정=배경)
    """
    base  = cv2.imread(base_path)
    annot = cv2.imread(annot_path)

    # 크기 통일 (혹시 다를 경우)
    if base.shape != annot.shape:
        annot = cv2.resize(annot, (base.shape[1], base.shape[0]))

    # 그레이스케일 diff
    base_gray  = cv2.cvtColor(base,  cv2.COLOR_BGR2GRAY).astype(np.int16)
    annot_gray = cv2.cvtColor(annot, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff       = np.abs(base_gray - annot_gray).astype(np.uint8)

    # 임계값 이진화
    _, mask = cv2.threshold(diff, cfg.DIFF_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 노이즈 제거: 작은 컴포넌트 제거
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= cfg.MASK_MIN_AREA:
            clean_mask[labels == label] = 255

    # 팽창: 필기 획 주변으로 마스크 확장
    kernel = np.ones((cfg.DIFF_DILATE_KERNEL, cfg.DIFF_DILATE_KERNEL), np.uint8)
    dilated = cv2.dilate(clean_mask, kernel, iterations=1)

    # 3채널로 변환 (Gemini 전송용)
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


def encode_image_b64(image_path: str) -> str:
    """이미지 파일을 base64 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_ndarray_b64(img: np.ndarray) -> str:
    """numpy 배열을 JPEG base64 인코딩"""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ──────────────────────────────────────────────
# Gemini 분석
# ──────────────────────────────────────────────
ANALYSIS_PROMPT = """
당신은 강의 슬라이드 분석 전문가입니다.
세 장의 이미지가 제공됩니다:
  - Image 1 (BASE): 필기 전 원본 슬라이드
  - Image 2 (ANNOTATED): 교수가 필기/강조를 추가한 슬라이드
  - Image 3 (DIFF MASK): 흰색 영역이 필기가 추가된 위치를 나타내는 마스크

마스크의 흰색 영역을 기준으로, 각 필기 표시를 분석하여 아래 JSON 형식으로만 응답하세요.
JSON 외의 설명, 마크다운 코드블록을 절대 포함하지 마세요.

{
  "annotations": [
    {
      "id": 1,
      "type": "circle" | "underline" | "arrow" | "handwritten_text" | "bracket" | "cross" | "other",
      "target_type": "text" | "image" | "diagram_element" | "none",
      "target_content": "강조 대상이 되는 원본 슬라이드의 텍스트를 하나의 문자열로 그대로 기재 (리스트 불가, target_keywords 필드 사용 금지). 필기 텍스트라면 null",
      "handwritten_content": "필기 텍스트인 경우 읽히는 내용. 강조 표시라면 null",
      "emphasis_intent": "이 표시가 전달하는 의도를 한 문장으로 (예: 'CPU와 캐시를 하드웨어 자원으로 강조', '1.5GB 메모리 크기 추가 설명')",
      "position": "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center" | "top" | "bottom" | "left" | "right",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "slide_summary": "이 슬라이드에서 필기를 통해 교수가 강조한 전체적인 포인트를 2-3문장으로 요약"
}

분류 기준:
- circle: 텍스트나 요소를 동그라미로 감쌈
- underline: 텍스트 아래에 선을 그음
- arrow: 화살표로 특정 요소를 가리킴
- handwritten_text: 새로운 단어/숫자/문장을 직접 씀
- bracket: 중괄호나 대괄호 형태로 묶음
- cross: X 표시로 취소하거나 부정
- other: 위 유형에 해당하지 않는 경우

confidence 기준:
- high: 마스크와 타겟이 명확히 일치
- medium: 타겟 추정 가능하나 불확실
- low: 마스크가 불명확하거나 필기가 판독 불가

중요: target_content는 반드시 문자열(string)이어야 하며, 리스트나 배열 형태로 출력하지 말 것.
target_keywords 같은 임의 필드를 추가하지 말 것. 위 JSON 스키마를 정확히 따를 것.
"""


def analyze_annotation_pair(
    client: genai.Client,
    base_path: str,
    annot_path: str,
    diff_mask: np.ndarray,
    slide_index: int,
    annot_index: int,
    timestamp_sec: float = 0.0,
) -> dict:
    """
    (base, annot, diff_mask) 세 장을 Gemini에게 전달하여 분석 결과를 반환합니다.
    """
    log.info(f"  분석 중: slide_{slide_index:03d}_annot_{annot_index:02d}")

    base_b64  = encode_image_b64(base_path)
    annot_b64 = encode_image_b64(annot_path)
    mask_b64  = encode_ndarray_b64(diff_mask)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(parts=[
                    types.Part(text="Image 1 (BASE):"),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64.b64decode(base_b64))),
                    types.Part(text="Image 2 (ANNOTATED):"),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64.b64decode(annot_b64))),
                    types.Part(text="Image 3 (DIFF MASK):"),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64.b64decode(mask_b64))),
                    types.Part(text=ANALYSIS_PROMPT),
                ])
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,   # 2048→8192: annotation 많은 슬라이드 대응
            )
        )

        raw = _sanitize_raw(response.text.strip())
        result = _parse_json_safe(raw)
        result["slide_index"]   = slide_index
        result["annot_index"]   = annot_index
        result["timestamp_sec"] = timestamp_sec
        result["base_path"]     = base_path
        result["annot_path"]    = annot_path
        return result

    except json.JSONDecodeError as e:
        log.warning(f"  JSON 파싱 실패 (slide {slide_index}, annot {annot_index}): {e}")
        return {
            "slide_index":   slide_index,
            "annot_index":   annot_index,
            "timestamp_sec": timestamp_sec,
            "base_path":     base_path,
            "annot_path":    annot_path,
            "error":         f"JSON parse error: {e}",
            "raw_response": response.text if 'response' in dir() else "no response",
            "annotations":   [],
            "slide_summary": "",
        }
    except Exception as e:
        log.error(f"  Gemini API 오류 (slide {slide_index}, annot {annot_index}): {e}")
        return {
            "slide_index":   slide_index,
            "annot_index":   annot_index,
            "timestamp_sec": timestamp_sec,
            "error": str(e),
            "annotations":   [],
            "slide_summary": "",
        }


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def analyze_all(slides_dir: str, output_path: str, save_masks: bool = False):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

    # output 디렉토리 자동 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    cfg    = Config()

    pairs  = load_slide_pairs(slides_dir)
    results = []

    mask_dir = Path(slides_dir) / "diff_masks" if save_masks else None
    if mask_dir:
        mask_dir.mkdir(exist_ok=True)

    for pair in pairs:
        slide_idx = pair["slide_index"]
        base_path = pair["base_path"]

        if not pair["annot_paths"]:
            log.info(f"[슬라이드 {slide_idx}] 필기 없음 - 스킵")
            continue

        log.info(f"[슬라이드 {slide_idx}] {len(pair['annot_paths'])}개 annot 분석 시작")

        # annot이 여러 개면 직전 annot을 base로 사용 (누적 필기 처리)
        current_base = base_path
        annot_timestamps = pair.get("annot_timestamps", [0.0] * len(pair["annot_paths"]))
        for annot_idx, (annot_path, ts) in enumerate(zip(pair["annot_paths"], annot_timestamps), start=1):
            # diff 마스크 생성
            diff_mask = build_diff_mask(current_base, annot_path, cfg)

            if save_masks:
                mask_fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}_mask.jpg"
                cv2.imwrite(str(mask_dir / mask_fname), diff_mask)

            # Gemini 분석
            result = analyze_annotation_pair(
                client, current_base, annot_path, diff_mask,
                slide_idx, annot_idx, timestamp_sec=ts
            )
            results.append(result)

            # 다음 세션의 base = 현재 annot (누적 필기 지원)
            current_base = annot_path

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_annots = sum(len(r.get("annotations", [])) for r in results)
    log.info(f"\n완료: {len(results)}개 분석, 총 {total_annots}개 annotation 추출 → {output_path}")
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="슬라이드 필기 VLM 분석기")
    parser.add_argument("--slides",  "-s", required=True,                      help="slide_extractor.py 출력 디렉토리")
    parser.add_argument("--output",  "-o", default="./output/annotation_analysis.json", help="분석 결과 JSON 경로")
    parser.add_argument("--masks",         action="store_true",                help="diff 마스크 이미지 저장 (디버그용)")
    args = parser.parse_args()

    analyze_all(args.slides, args.output, save_masks=args.masks)