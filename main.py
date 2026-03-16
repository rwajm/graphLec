#!/usr/bin/env python3
"""
최종 파이프라인 (main):
- 전사: metadata.json 기준 슬라이드별 전사
- 교정: slide_textualized.json(전문용어/맥락) 반영 2단계 교정
- 이후: 컨텍스트 그룹핑 + 강조 감지 + by_slide/노트 출력

입력 파일(영상, metadata.json, slide_textualized.json)은 src/ 폴더에 둡니다.

사용법:
  python main.py <video.mp4> [metadata.json]
  (파일명만 주면 src/<파일명>에서 찾고, 없으면 현재 경로에서 찾음)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa

from utils import get_video_duration
from audio_analyzer import (
    extract_audio_from_video,
    analyze_audio_features,
    evaluate_audio_quality,
)
from transcriber import transcribe_video
from text_processor import (
    correct_segments_dual_with_slide_context,
    generate_lecture_notes,
)
from emphasis_detector_std import (
    detect_emphasis_by_std,
    detect_emphasis_by_keywords_weighted,
)
from emphasis_detector_std_topic import (
    detect_emphasis_by_topic_keyword_repetition,
    get_topic_keywords_filtered_v2,
)
from emphasis_comparator import combine_emphasis_simple
from segment_grouper import (
    group_segments_by_context,
    expand_group_annotations_to_segments,
    load_slide_ranges,
    group_segments_by_slide_and_context,
    group_segments_by_slide_and_context_iterative,
)

SRC_DIR = ROOT_DIR / "src"


def _resolve_input(path_or_name: str, must_exist: bool = False) -> Path:
    """입력 파일: src/ 우선, 없으면 인자 경로 그대로."""
    p = Path(path_or_name)
    if p.is_file():
        return p
    candidate = SRC_DIR / p.name
    if candidate.is_file():
        return candidate
    return p


def transcribe_slide_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    *,
    output_dir: Path,
    chunk_duration: float = 600.0,
) -> list[dict]:
    """영상 [start_sec, end_sec) 구간만 잘라서 Groq Whisper 전사."""
    from config import groq_client

    if end_sec <= start_sec:
        return []

    duration = end_sec - start_sec
    segments: list[dict] = []
    total_chunks = max(
        1, int(duration / chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
    )

    for i in range(total_chunks):
        rel_start = i * chunk_duration
        chunk_start = start_sec + rel_start
        if chunk_start >= end_sec:
            break
        this_chunk_duration = min(chunk_duration, end_sec - chunk_start)
        chunk_path = str(output_dir / f"temp_slide_chunk_{start_sec:.0f}_{i}.wav")

        subprocess.run(
            [
                "ffmpeg", "-ss", str(chunk_start), "-t", str(this_chunk_duration),
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", chunk_path,
            ],
            capture_output=True,
        )

        p = Path(chunk_path)
        if not p.exists() or p.stat().st_size == 0:
            continue

        with open(chunk_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(chunk_path, f.read()),
                model="whisper-large-v3-turbo",
                language="ko",
                response_format="verbose_json",
            )

        for seg in transcription.segments:
            segments.append({
                "start": float(seg["start"]) + chunk_start,
                "end": float(seg["end"]) + chunk_start,
                "text": (seg["text"] or "").strip(),
            })
        p.unlink(missing_ok=True)

    return segments


def transcribe_by_slide(
    video_path: str,
    duration: float,
    metadata_path: str | None,
    output_dir: Path,
) -> list[dict]:
    """metadata.json 슬라이드 구간별 전사 후 세그먼트 병합. 각 세그먼트에 slide_index 부여."""
    if not metadata_path or not Path(metadata_path).is_file():
        print("  ℹ️ metadata.json 없음 → 전체 전사 방식 사용")
        return transcribe_video(video_path, duration, output_dir=output_dir)

    print(f"  ✓ 슬라이드 구간 전사 사용: {metadata_path}")
    slide_ranges = load_slide_ranges(metadata_path, duration)
    print(f"  ✓ 슬라이드 범위 {len(slide_ranges)}개")

    all_segments: list[dict] = []
    for r in slide_ranges:
        sidx = r["slide_index"]
        start_sec = float(r["start_sec"])
        end_sec = float(r["end_sec"])
        print(f"  ▶ 슬라이드 {sidx}: {start_sec:.1f}s ~ {end_sec:.1f}s 전사...")
        segs = transcribe_slide_range(video_path, start_sec, end_sec, output_dir=output_dir)
        print(f"    ✓ 슬라이드 {sidx}: {len(segs)}개 세그먼트")
        for seg in segs:
            s = seg.copy()
            s["slide_index"] = sidx
            all_segments.append(s)

    all_segments.sort(key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))
    print(f"  ✓ 병합 완료: {len(all_segments)}개 세그먼트")
    return all_segments


def load_slide_textualized(video_stem: str | None = None) -> dict[int, dict]:
    """slide_textualized.json은 src/에 둠. <stem>_slide_textualized.json 또는 slide_textualized.json."""
    candidates: list[Path] = []
    if video_stem:
        candidates.append(SRC_DIR / f"{video_stem}_slide_textualized.json")
        candidates.append(SRC_DIR / "slide_textualized.json")
        candidates.append(ROOT_DIR / "output" / f"{video_stem}_slide_textualized.json")
    else:
        candidates.append(SRC_DIR / "slide_textualized.json")

    path = None
    for c in candidates:
        if c.is_file():
            path = c
            break
    if path is None:
        print("  ℹ️ slide_textualized.json 없음 → 슬라이드 컨텍스트 없이 교정")
        return {}

    print(f"  ✓ slide_textualized 로드: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    slides = data.get("slides") or []
    return {s["slide_number"]: s for s in slides if isinstance(s.get("slide_number"), int)}


def resolve_metadata_path(video_stem: str | None = None) -> str | None:
    """
    인자로 주지 않아도 metadata.json을 찾음.
    우선순위: src/<stem>_metadata.json → src/metadata.json → 루트 metadata.json
    """
    candidates: list[Path] = []
    if video_stem:
        candidates.append(SRC_DIR / f"{video_stem}_metadata.json")
    candidates.append(SRC_DIR / "metadata.json")
    candidates.append(ROOT_DIR / "metadata.json")
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _format_emphasis_reason(ann: dict) -> dict:
    detail = ann.get("emphasis_detail")
    if isinstance(detail, dict):
        return detail
    return {
        "score": ann.get("emphasis_score"),
        "methods": ann.get("emphasis_methods", []),
        "keywords": ann.get("emphasis_keywords", []),
        "keywords_by_method": ann.get("emphasis_keywords_by_method", {}),
        "detection_count": ann.get("detection_count", 0),
    }


def main():
    if len(sys.argv) < 2:
        print("사용법: python main.py <video.mp4> [metadata.json]")
        print("  영상만 필수. metadata.json·slide_textualized.json은 인자 없이도 src/에서 자동 탐색.")
        sys.exit(1)

    video_arg = _resolve_input(sys.argv[1])
    if not video_arg.is_file():
        print(f"오류: 영상을 찾을 수 없습니다: {video_arg}")
        sys.exit(1)
    video_path = str(video_arg)

    output_name = Path(video_path).stem

    metadata_path = None
    if len(sys.argv) >= 3:
        meta_arg = _resolve_input(sys.argv[2])
        if meta_arg.is_file():
            metadata_path = str(meta_arg)
    if not metadata_path:
        metadata_path = resolve_metadata_path(output_name)
        if metadata_path:
            print(f"  ✓ metadata 자동 탐색: {metadata_path}")
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    start_time = time.time()
    duration = get_video_duration(video_path)
    print(f"📁 {video_path} ({duration/60:.1f}분)")

    # [0] 오디오 품질 분석
    print("\n[0/5] 🎵 오디오 품질 분석...")
    audio_path = str(output_dir / "temp_full_audio.wav")
    extract_audio_from_video(video_path, audio_path)
    try:
        audio_features = analyze_audio_features(audio_path)
        audio_quality = evaluate_audio_quality(audio_features)
        with open(output_dir / f"{output_name}_audio_features_v2.json", "w", encoding="utf-8") as f:
            json.dump(audio_features, f, ensure_ascii=False, indent=2)
        with open(output_dir / f"{output_name}_audio_quality_v2.json", "w", encoding="utf-8") as f:
            json.dump(audio_quality, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 품질 평가: {audio_quality['overall_score']}/100 ({audio_quality['overall_grade']})")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    # [1] 슬라이드별 전사
    print("\n[1/5] Groq Whisper 전사 (슬라이드별)...")
    segments_raw = transcribe_by_slide(video_path, duration, metadata_path, output_dir)
    print(f"  ✓ {len(segments_raw)}개 세그먼트")

    slide_context_by_index = load_slide_textualized(output_name)

    # [2] 전사 2단계 교정 (slide_textualized 컨텍스트 활용)
    print("\n[2/5] 전사 교정 (전문용어/오타 + 자연스러운 문장)...")
    segments = correct_segments_dual_with_slide_context(segments_raw, slide_context_by_index)
    print("  ✓ 교정 완료 (text_raw / text_corrected / text_natural)")

    # [3] 컨텍스트 그룹핑 + 강조 감지
    print("\n[3/5] 컨텍스트 그룹핑 + 강조 구간 감지...")
    slides_structure = None
    slide_ranges = None
    if metadata_path and Path(metadata_path).is_file():
        slide_ranges = load_slide_ranges(metadata_path, duration)
        groups, slides_structure = group_segments_by_slide_and_context(
            segments, slide_ranges, duration, use_llm_merge=True, use_pause_sentence=False
        )
        print(f"  ✓ 슬라이드-컨텍스트 그룹: {len(groups)}개")
    else:
        groups = group_segments_by_context(segments)
        print(f"  ✓ 맥락 그룹: {len(groups)}개")

    audio_path_temp = str(output_dir / "temp_analysis_audio_v2.wav")
    extract_audio_from_video(video_path, audio_path_temp)
    y, sr = librosa.load(audio_path_temp, sr=16000)

    try:
        topic_kw_set = get_topic_keywords_filtered_v2(
            groups, min_freq=5, max_keywords=20, max_segment_ratio=1.0,
            min_keyword_len=2, candidate_pool_size=80, use_llm_filter=True,
        )
        std_topic_audio = detect_emphasis_by_std(y, sr, groups)
        std_topic_keyword = detect_emphasis_by_keywords_weighted(groups)
        std_topic_repeat = detect_emphasis_by_topic_keyword_repetition(
            groups, window=2, min_keyword_len=2, max_segment_ratio=1.0,
            min_freq=5, max_keywords=20, use_llm_filter=False, min_keyword_count=1,
            _topic_keywords_override=topic_kw_set,
        )
        std_topic_annotated_groups, std_topic_emphasis = combine_emphasis_simple(
            std_topic_audio, std_topic_keyword + std_topic_repeat, groups,
        )
        annotated_segments = expand_group_annotations_to_segments(
            std_topic_annotated_groups, segments, groups
        )

        with open(output_dir / f"{output_name}_emphasis_std_topic_v2.json", "w", encoding="utf-8") as f:
            json.dump({
                "method": "std_topic_v2",
                "description": "표준편차 + 가중치 키워드 + 주제 키워드 반복",
                "statistics": {
                    "total_count": len(std_topic_emphasis),
                    "ratio": round(len(std_topic_emphasis) / len(groups), 3) if groups else 0,
                    "duration": round(sum(s["end"] - s["start"] for s in std_topic_emphasis), 2),
                },
                "emphasis_sections": std_topic_emphasis,
            }, f, ensure_ascii=False, indent=2)

        with open(output_dir / f"{output_name}_emphasis_keywords_v2.json", "w", encoding="utf-8") as f:
            kw_report = {
                "description": "강조 감지에 사용된 주제 키워드",
                "topic_keywords": {"keywords": sorted(topic_kw_set), "count": len(topic_kw_set)},
            }
            json.dump(kw_report, f, ensure_ascii=False, indent=2)

        annot_by_start = {g["start"]: g for g in std_topic_annotated_groups}

        if slides_structure is not None and slide_ranges is not None:
            slides_batch = json.loads(json.dumps(slides_structure))
            for slide in slides_batch:
                new_contexts = []
                for ctx in slide["contexts"]:
                    ann = annot_by_start.get(ctx["start"])
                    ordered_ctx = {
                        "context_index": ctx.get("context_index"),
                        "start": ctx.get("start"), "end": ctx.get("end"), "text": ctx.get("text"),
                    }
                    if ann and ann.get("emphasis") == "강조":
                        ordered_ctx["emphasis"] = {"state": "강조", "detected": True, "detail": _format_emphasis_reason(ann)}
                    else:
                        ordered_ctx["emphasis"] = {"state": None, "detected": False}
                    ordered_ctx["segment_indices"] = ctx.get("segment_indices", [])
                    ordered_ctx["segments"] = ctx.get("segments", [])
                    new_contexts.append(ordered_ctx)
                slide["contexts"] = new_contexts
            with open(output_dir / f"{output_name}_by_slide_v2.json", "w", encoding="utf-8") as f:
                json.dump({"slides": slides_batch}, f, ensure_ascii=False, indent=2)
            print(f"    📄 output/{output_name}_by_slide_v2.json")

            slides_iter = group_segments_by_slide_and_context_iterative(
                annotated_segments, slide_ranges, duration
            )
            for slide in slides_iter:
                for ctx in slide["contexts"]:
                    seg_indices = ctx.get("segment_indices", [])
                    segs_ctx = [annotated_segments[i] for i in seg_indices if 0 <= i < len(annotated_segments)]
                    emphasized = [s for s in segs_ctx if s.get("emphasis") == "강조"]
                    if not emphasized:
                        ctx["emphasis"] = {"state": None, "detected": False}
                        continue
                    best = max(emphasized, key=lambda s: s.get("emphasis_score", 0.0))
                    ctx["emphasis"] = {"state": "강조", "detected": True, "detail": _format_emphasis_reason(best)}
            with open(output_dir / f"{output_name}_by_slide_iterative_v2.json", "w", encoding="utf-8") as f:
                json.dump({"slides": slides_iter}, f, ensure_ascii=False, indent=2)
            print(f"    📄 output/{output_name}_by_slide_iterative_v2.json")
    finally:
        Path(audio_path_temp).unlink(missing_ok=True)

    # [4] 강의 노트
    print("\n[4/5] 강의 정리본 생성...")
    notes = generate_lecture_notes(segments)
    with open(output_dir / f"{output_name}_notes_v2.md", "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"  ✓ output/{output_name}_notes_v2.md")

    print(f"\n완료. 소요 시간: {(time.time() - start_time)/60:.1f}분")


if __name__ == "__main__":
    main()
