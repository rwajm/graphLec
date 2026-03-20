#!/usr/bin/env python3
"""
지시어 추출/애매 판정만 단독 실행하는 진입점.

사용법:
  # A) 기존 segments_v2.json으로 실행
  python run_deictics.py output/1장_영상_segments_v2.json

  # B) video로 바로 실행 (전사 + 교정 + 지시어 분석)
  python run_deictics.py src/1장_영상.mp4 [metadata.json]

출력:
  - output/<video_stem>_deictics_v2.json
  - output/<video_stem>_deictics_ambiguous_v2.json
"""

import json
import sys
from pathlib import Path

from main import (
    extract_deictics_from_segments,
    classify_ambiguous_deictics_with_llm,
    transcribe_by_slide,
    resolve_metadata_path,
    load_slide_textualized,
    _resolve_input,
)
from text_processor import correct_segments_dual_with_slide_context
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
    video_stem = Path(video_path).stem if video_path else segments_path.stem.replace("_segments_v2", "")
    has_words = any(isinstance(s, dict) and isinstance(s.get("words"), list) and s.get("words") for s in segments)
    if not has_words:
        print("ℹ️ 입력 segments에는 word timestamp(words)가 없어 deictic_time은 일부 추정값일 수 있습니다.")
    return segments, video_path, video_stem


def _run_from_video(video_path: Path, metadata_arg: str | None) -> tuple[list[dict], str, str]:
    if not video_path.is_file():
        raise ValueError(f"영상을 찾을 수 없습니다: {video_path}")

    duration = get_video_duration(str(video_path))
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    video_stem = video_path.stem

    metadata_path = None
    if metadata_arg:
        meta = _resolve_input(metadata_arg)
        if meta.is_file():
            metadata_path = str(meta)
    if not metadata_path:
        metadata_path = resolve_metadata_path(video_stem)

    print(f"📁 {video_path} ({duration/60:.1f}분)")
    print("[1/2] 전사(슬라이드별) + 교정 중...")
    segments_raw = transcribe_by_slide(str(video_path), duration, metadata_path, output_dir)
    slide_context = load_slide_textualized(video_stem)
    segments = correct_segments_dual_with_slide_context(segments_raw, slide_context)
    print(f"  ✓ 세그먼트 {len(segments)}개")
    return segments, str(video_path), video_stem


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python run_deictics.py <segments_v2.json>")
        print("  python run_deictics.py <video.mp4> [metadata.json]")
        sys.exit(1)

    arg1 = Path(sys.argv[1])
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        if _is_json_path(arg1):
            segments, video_path, video_stem = _run_from_segments_json(arg1)
        else:
            v = _resolve_input(sys.argv[1])
            metadata_arg = sys.argv[2] if len(sys.argv) >= 3 else None
            segments, video_path, video_stem = _run_from_video(v, metadata_arg)
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)

    deictics_report = extract_deictics_from_segments(segments, text_field="text_corrected")
    deictics_report["video_path"] = video_path
    deictics_path = output_dir / f"{video_stem}_deictics_v2.json"
    with open(deictics_path, "w", encoding="utf-8") as f:
        json.dump(deictics_report, f, ensure_ascii=False, indent=2)
    print(f"✓ 지시어 추출 저장: {deictics_path}")

    threshold = 0.6
    # 기존 포맷을 해치지 않도록 words는 내부 계산에서만 사용
    segments_clean = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    ambiguous_report = classify_ambiguous_deictics_with_llm(
        segments_clean, deictics_report, threshold=threshold, context_window=2
    )
    ambiguous_report["video_path"] = video_path
    ambiguous_path = output_dir / f"{video_stem}_deictics_ambiguous_v2.json"
    with open(ambiguous_path, "w", encoding="utf-8") as f:
        json.dump(ambiguous_report, f, ensure_ascii=False, indent=2)
    print(f"✓ 애매 지시어 저장: {ambiguous_path}")


if __name__ == "__main__":
    main()

