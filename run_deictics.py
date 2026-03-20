#!/usr/bin/env python3
"""
지시어 추출/애매 판정만 단독 실행하는 진입점.

사용법:
  # A) 기존 segments.json으로 실행
  python run_deictics.py output/lecture_segments.json

  # B) video로 바로 실행 (전사 + 교정 + 지시어 분석)
  python run_deictics.py input/lecture.mp4

출력:
  - output/{stem}_deictics.json
  - output/{stem}_deictics_ambiguous.json
"""

import json
import sys
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, DEFAULT_SLIDES_DIR, output_paths
from main import extract_deictics_from_segments, classify_ambiguous_deictics_with_llm
from text_processor import correct_segments_dual_with_slide_context
from segment_grouper import load_slide_ranges
from utils import get_video_duration


def _is_json_path(path: Path) -> bool:
    return path.suffix.lower() == ".json"


def _run_from_segments_json(segments_path: Path) -> tuple[list[dict], str, str]:
    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments") or []
    if not segments:
        raise ValueError("segments 배열이 비어 있습니다.")
    video_path = data.get("video_path") or ""
    video_stem = Path(video_path).stem if video_path else segments_path.stem.replace("_segments", "")
    has_words = any(
        isinstance(s, dict) and isinstance(s.get("words"), list) and s.get("words")
        for s in segments
    )
    if not has_words:
        print("ℹ️ 입력 segments에는 word timestamp(words)가 없어 deictic_time은 일부 추정값일 수 있습니다.")
    return segments, video_path, video_stem


def _run_from_video(video_path: Path) -> tuple[list[dict], str, str]:
    if not video_path.is_file():
        raise ValueError(f"영상을 찾을 수 없습니다: {video_path}")

    from main import _transcribe_by_slide
    from config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    stem = video_path.stem
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    duration = get_video_duration(str(video_path))
    meta_path = str(DEFAULT_SLIDES_DIR / "metadata.json")
    paths = output_paths(stem, output_dir, DEFAULT_SLIDES_DIR)

    # slide_textualized 로드
    slide_context_by_index: dict[int, dict] = {}
    tex_path = paths["textualized"]
    if tex_path.is_file():
        with open(tex_path, "r", encoding="utf-8") as f:
            tex_data = json.load(f)
        slide_context_by_index = {
            s["slide_number"]: s
            for s in tex_data.get("slides", [])
            if isinstance(s.get("slide_number"), int)
        }

    slide_ranges = load_slide_ranges(meta_path, duration) if Path(meta_path).is_file() else []

    print(f"📁 {video_path} ({duration/60:.1f}분)")
    print("[1/2] 전사(슬라이드별) + 교정 중...")
    segments_raw = _transcribe_by_slide(str(video_path), duration, meta_path, slide_ranges, output_dir)
    segments = correct_segments_dual_with_slide_context(segments_raw, slide_context_by_index)
    print(f"  ✓ 세그먼트 {len(segments)}개")
    return segments, str(video_path), stem


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python run_deictics.py <segments.json>")
        print("  python run_deictics.py <video.mp4>")
        sys.exit(1)

    arg1 = Path(sys.argv[1])
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    try:
        if _is_json_path(arg1):
            segments, video_path, stem = _run_from_segments_json(arg1)
        else:
            segments, video_path, stem = _run_from_video(arg1)
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)

    paths = output_paths(stem, output_dir, DEFAULT_SLIDES_DIR)

    # 지시어 추출
    deictics_report = extract_deictics_from_segments(segments, text_field="text_corrected")
    deictics_report["video_path"] = video_path
    with open(paths["deictics"], "w", encoding="utf-8") as f:
        json.dump(deictics_report, f, ensure_ascii=False, indent=2)
    print(f"✓ 지시어 추출 저장: {paths['deictics']}")

    # 애매 지시어 판정
    segments_clean = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    ambiguous_report = classify_ambiguous_deictics_with_llm(
        segments_clean, deictics_report, threshold=0.6, context_window=2
    )
    ambiguous_report["video_path"] = video_path
    with open(paths["deictics_ambiguous"], "w", encoding="utf-8") as f:
        json.dump(ambiguous_report, f, ensure_ascii=False, indent=2)
    print(f"✓ 애매 지시어 저장: {paths['deictics_ambiguous']}")


if __name__ == "__main__":
    main()