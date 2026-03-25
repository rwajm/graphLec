"""
main.py
=======
강의 영상 분석 통합 파이프라인

실행 흐름:
  [병렬] Stage 1A: slide_extractor     — 슬라이드 프레임 추출
         Stage 1B: audio_analyzer      — 오디오 품질 분석
  [직렬] Stage 2 : slide_textualizer   — 슬라이드 텍스트 + 강조 추출
  [병렬] Stage 3A: annotation_analyzer — 필기 강조 분석
         Stage 3B: 오디오 파이프라인
                     transcriber       — 슬라이드별 전사 (metadata 필요)
                     text_processor    — 2단계 교정 + 침묵/지시어 추출
                     emphasis          — 오디오 강조 감지
  [병렬] Stage 4A: slide_classifier    — 슬라이드 역할 분류
         Stage 4B: by_slide 구조 저장  — (3B 결과 기반)
  [직렬] Stage 5 : fusion              — 최종 통합

Usage:
    python main.py --input lecture.mp4
    python main.py --input lecture.mp4 --output output/ --slides output/slides/
    python main.py --input lecture.mp4 --skip-extract
    python main.py --input lecture.mp4 --debug --masks
    python main.py --input lecture.mp4 --force
"""

try:
    import lzma
except ImportError:
    from backports import lzma  # lzma가 없으면 backports.lzma를 사용

import json
import sys
import time
import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
from deictics import (
    classify_ambiguous_deictics_with_llm,
    extract_deictics_from_segments,
    normalize_word_items,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _done(label: str, elapsed: float):
    print(f"\n  ✓ {label}  ({elapsed:.1f}초)")
    print("─" * 70)


def _save_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _is_done(path: Path, label: str, force: bool) -> bool:
    """출력 파일이 이미 존재하면 True 반환 (force=True면 항상 False)."""
    if not force and path.exists() and path.stat().st_size > 0:
        print(f"\n  ⏭  {label} — 출력 파일 존재, 스킵")
        print(f"     {path}")
        print("─" * 70)
        return True
    return False


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


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


# ──────────────────────────────────────────────────────────────
# 전사 헬퍼
# ──────────────────────────────────────────────────────────────

def _transcribe_range(
    video_path: str, start_sec: float, end_sec: float, output_dir: Path, groq_client
) -> list[dict]:
    chunk_duration = 600.0
    if end_sec <= start_sec:
        return []
    segments: list[dict] = []
    total_chunks = max(
        1, int((end_sec - start_sec) / chunk_duration) + (1 if (end_sec - start_sec) % chunk_duration > 0 else 0)
    )
    for i in range(total_chunks):
        chunk_start = start_sec + i * chunk_duration
        if chunk_start >= end_sec:
            break
        this_dur = min(chunk_duration, end_sec - chunk_start)
        chunk_path = str(output_dir / f"temp_chunk_{start_sec:.0f}_{i}.wav")
        subprocess.run([
            "ffmpeg", "-ss", str(chunk_start), "-t", str(this_dur),
            "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", chunk_path,
        ], capture_output=True)
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
            words = normalize_word_items(seg.get("words"), chunk_start=chunk_start)
            segments.append({
                "start": float(seg["start"]) + chunk_start,
                "end": float(seg["end"]) + chunk_start,
                "text": (seg["text"] or "").strip(),
                "words": words,
            })
        p.unlink(missing_ok=True)
    return segments


def _transcribe_by_slide(
    video_path: str,
    duration: float,
    meta_path: str | None,
    slide_ranges: list[dict],
    output_dir: Path,
) -> list[dict]:
    from transcriber import transcribe_video
    from config import groq_client

    if not meta_path or not slide_ranges:
        print("  ℹ️ metadata 없음 → 전체 전사 방식 사용")
        return transcribe_video(video_path, duration, output_dir=output_dir)

    all_segments: list[dict] = []
    for r in slide_ranges:
        sidx = r["slide_index"]
        start_sec = float(r["start_sec"])
        end_sec = float(r["end_sec"])
        print(f"    ▶ 슬라이드 {sidx}: {start_sec:.1f}s ~ {end_sec:.1f}s 전사...")
        segs = _transcribe_range(video_path, start_sec, end_sec, output_dir, groq_client)
        for seg in segs:
            s = seg.copy()
            s["slide_index"] = sidx
            all_segments.append(s)

    all_segments.sort(key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))
    return all_segments


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지
# ──────────────────────────────────────────────────────────────

def stage1_extract(args, slides_dir: Path, output_dir: Path) -> dict:
    from slide_extractor import extract_slides

    stem = Path(args.input).stem
    meta_path = output_dir / f"{stem}_metadata.json"

    if _is_done(meta_path, "Stage 1A 슬라이드 추출", args.force):
        return {"meta_path": str(meta_path), "elapsed": 0.0}

    _banner("Stage 1  —  슬라이드 추출  (slide_extractor)")
    t0 = time.time()
    metadata = extract_slides(
        input_path=args.input,
        output_dir=str(slides_dir),
        debug=args.debug,
    )
    elapsed = time.time() - t0

    _save_json(meta_path, metadata)

    slide_count = len({m["slide_index"] for m in metadata})
    _done(f"슬라이드 {slide_count}개, 프레임 {len(metadata)}개 추출", elapsed)
    return {"meta_path": str(meta_path), "elapsed": elapsed}


def stage1b_audio_analyze(args, output_dir: Path) -> dict:
    """Stage 1B: 오디오 품질 분석 (slide_extractor와 병렬)"""
    from audio_analyzer import extract_audio_from_video, analyze_audio_features, evaluate_audio_quality
    from utils import get_video_duration

    stem = Path(args.input).stem
    audio_quality_path = output_dir / f"{stem}_audio_quality.json"

    if _is_done(audio_quality_path, "Stage 1B 오디오 품질 분석", args.force):
        # duration은 파일에서 복원
        duration = 0.0
        try:
            with open(audio_quality_path) as f:
                q = json.load(f)
            duration = _safe_float(q.get("duration_sec"), 0.0)
        except Exception:
            pass
        if duration == 0.0:
            duration = get_video_duration(args.input)
        return {"duration": duration, "elapsed": 0.0}

    video_path = args.input

    _banner("Stage 1B  —  오디오 품질 분석  (audio_analyzer)")
    t0 = time.time()
    duration = get_video_duration(video_path)
    audio_path = str(output_dir / "temp_full_audio.wav")
    extract_audio_from_video(video_path, audio_path)
    try:
        audio_features = analyze_audio_features(audio_path)
        audio_quality = evaluate_audio_quality(audio_features)
        _save_json(output_dir / f"{stem}_audio_features.json", audio_features)
        _save_json(audio_quality_path, audio_quality)
        print(f"  ✓ 품질: {audio_quality['overall_score']}/100 ({audio_quality['overall_grade']})")
    finally:
        Path(audio_path).unlink(missing_ok=True)
    elapsed = time.time() - t0
    _done("오디오 품질 분석", elapsed)
    return {"duration": duration, "elapsed": elapsed}


def stage2_textualize(args, slides_dir: Path, output_dir: Path) -> dict:
    from slide_textualizer import TextualizationPipeline, Config as TextConfig

    stem = Path(args.input).stem
    textualized_path = output_dir / f"{stem}_slide_textualized.json"

    if _is_done(textualized_path, "Stage 2 슬라이드 텍스트화", args.force):
        return {"textualized_path": str(textualized_path), "elapsed": 0.0}

    _banner("Stage 2  —  슬라이드 텍스트화  (slide_textualizer)")
    t0 = time.time()
    text_config = TextConfig(
        slides_dir=slides_dir,
        output_dir=output_dir,
        output_filename=textualized_path.name,
        max_retries=args.retries,
    )
    text_result = TextualizationPipeline(text_config).run()
    elapsed = time.time() - t0

    _done(f"슬라이드 {text_result['metadata']['total_slides']}개 텍스트화", elapsed)
    return {"textualized_path": str(textualized_path), "elapsed": elapsed}


def stage3a_annotation(args, slides_dir: Path, output_dir: Path) -> dict:
    from annotation_analyzer import analyze_all

    stem = Path(args.input).stem
    annotation_path = output_dir / f"{stem}_annotation.json"

    if _is_done(annotation_path, "Stage 3A annotation", args.force):
        return {"annotation_path": str(annotation_path), "elapsed": 0.0}

    _banner("Stage 3A  —  필기 강조 분석  (annotation_analyzer)")
    t0 = time.time()
    annot_results = analyze_all(
        slides_dir=str(slides_dir),
        output_path=str(annotation_path),
        save_masks=args.masks,
    )
    elapsed = time.time() - t0

    n_annots = sum(len(r.get("annotations", [])) for r in annot_results)
    _done(f"{len(annot_results)}개 분석, 총 {n_annots}개 강조 추출", elapsed)
    return {"annotation_path": str(annotation_path), "elapsed": elapsed}


def stage3b_audio(args, meta_path: str, textualized_path: str, duration: float, output_dir: Path) -> dict:
    from text_processor import correct_segments_dual_with_slide_context
    from segment_grouper import (
        load_slide_ranges,
        group_segments_by_context,
        group_segments_by_slide_and_context,
        expand_group_annotations_to_segments,
    )
    from audio_analyzer import extract_audio_from_video
    from emphasis_audio import detect_emphasis_by_std
    from emphasis_keyword import (
        detect_emphasis_by_keywords_weighted,
        detect_emphasis_by_topic_keyword_repetition,
        get_topic_keywords_filtered_v2,
    )
    from emphasis_combiner import combine_emphasis_simple

    stem = Path(args.input).stem
    segments_path = output_dir / f"{stem}_segments.json"
    silences_path = output_dir / f"{stem}_silences.json"
    emphasis_path = output_dir / f"{stem}_emphasis.json"

    if _is_done(segments_path, "Stage 3B 오디오 파이프라인", args.force):
        # in-memory 데이터를 저장된 파일에서 복원
        by_slide_path = output_dir / f"{stem}_by_slide.json"
        slides_structure = None
        if by_slide_path.exists():
            try:
                with open(by_slide_path) as f:
                    slides_structure = json.load(f).get("slides")
            except Exception:
                pass

        annotated_segments: list[dict] = []
        try:
            with open(segments_path) as f:
                annotated_segments = json.load(f).get("segments", [])
        except Exception:
            pass

        slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []

        return {
            "segments_path": str(segments_path),
            "silences_path": str(silences_path),
            "deictics_path": str(output_dir / f"{stem}_deictics.json"),
            "emphasis_path": str(emphasis_path),
            "annotated_segments": annotated_segments,
            "annotated_groups": [],
            "slides_structure": slides_structure,
            "slide_ranges": slide_ranges,
            "duration": duration,
        }

    video_path = args.input

    _banner("Stage 3B  —  오디오 파이프라인")
    slide_context_by_index: dict[int, dict] = {}
    if textualized_path and Path(textualized_path).is_file():
        with open(textualized_path, "r", encoding="utf-8") as f:
            tex_data = json.load(f)
        slide_context_by_index = {
            s["slide_number"]: s
            for s in tex_data.get("slides", [])
            if isinstance(s.get("slide_number"), int)
        }

    # [3B-1] 슬라이드별 전사
    print("  [3B-1] 슬라이드별 전사...")
    t0 = time.time()
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    segments_raw = _transcribe_by_slide(video_path, duration, meta_path, slide_ranges, output_dir)
    print(f"    ✓ {len(segments_raw)}개 세그먼트  ({time.time()-t0:.1f}초)")

    # [3B-2] 텍스트 2단계 교정
    print("  [3B-2] 텍스트 교정...")
    t0 = time.time()
    segments = correct_segments_dual_with_slide_context(segments_raw, slide_context_by_index)
    segments_clean = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    _save_json(segments_path, {
        "video_path": video_path,
        "segment_count": len(segments_clean),
        "segments": segments_clean,
    })
    print(f"    ✓ 교정 완료  ({time.time()-t0:.1f}초)")

    # [3B-3] 침묵 구간 추출
    print("  [3B-3] 침묵 구간 추출...")
    MIN_SILENCE_SEC = 0.6
    silences: list[dict] = []
    for i in range(len(segments_clean) - 1):
        cur, nxt = segments_clean[i], segments_clean[i + 1]
        gap = float(nxt.get("start", 0.0)) - float(cur.get("end", 0.0))
        if gap >= MIN_SILENCE_SEC:
            silences.append({
                "index": len(silences),
                "start": float(cur.get("end", 0.0)),
                "end": float(nxt.get("start", 0.0)),
                "duration": gap,
                "prev_segment_index": i,
                "next_segment_index": i + 1,
            })
    _save_json(silences_path, {
        "video_path": video_path,
        "total_duration_sec": duration,
        "segment_count": len(segments_clean),
        "min_silence_sec": MIN_SILENCE_SEC,
        "silence_count": len(silences),
        "total_silence_duration_sec": sum(s["duration"] for s in silences),
        "silences": silences,
    })
    print(f"    ✓ 침묵 {len(silences)}개")

    # [3B-4] 지시어 추출
    print("  [3B-4] 지시어 추출...")
    deictics_report = extract_deictics_from_segments(segments, text_field="text_corrected")
    deictics_report["video_path"] = video_path
    _save_json(output_dir / f"{stem}_deictics.json", deictics_report)
    ambiguous_report = classify_ambiguous_deictics_with_llm(
        segments_clean, deictics_report, threshold=0.6, context_window=2
    )
    ambiguous_report["video_path"] = video_path
    _save_json(output_dir / f"{stem}_deictics_ambiguous.json", ambiguous_report)
    print(f"    ✓ 지시어 {deictics_report['deictic_total_count']}개, 애매 {ambiguous_report['ambiguous_count']}개")

    # [3B-5] 오디오 강조 감지
    print("  [3B-5] 오디오 강조 감지...")
    t0 = time.time()
    audio_path_temp = str(output_dir / "temp_analysis_audio.wav")
    extract_audio_from_video(video_path, audio_path_temp)
    annotated_segments: list[dict] = []
    annotated_groups: list[dict] = []
    slides_structure = None
    emphasis_sections: list[dict] = []
    topic_kw_set: set[str] = set()
    try:
        y, sr = librosa.load(audio_path_temp, sr=16000)
        if slide_ranges:
            groups, slides_structure = group_segments_by_slide_and_context(
                segments_clean, slide_ranges, duration, use_pause_sentence=False, use_llm_merge=True
            )
        else:
            groups = group_segments_by_context(segments_clean)
            slides_structure = None

        topic_kw_set = get_topic_keywords_filtered_v2(
            groups, min_freq=5, max_keywords=20, max_segment_ratio=1.0,
            min_keyword_len=2, candidate_pool_size=80, use_llm_filter=True,
        )
        audio_emphasis = detect_emphasis_by_std(y, sr, groups)
        keyword_emphasis = detect_emphasis_by_keywords_weighted(groups)
        topic_emphasis = detect_emphasis_by_topic_keyword_repetition(
            groups, window=2, min_keyword_len=2, max_segment_ratio=1.0,
            min_freq=5, max_keywords=20, use_llm_filter=False, min_keyword_count=1,
            _topic_keywords_override=topic_kw_set,
        )
        annotated_groups, emphasis_sections = combine_emphasis_simple(
            audio_emphasis, keyword_emphasis + topic_emphasis, groups,
        )
        annotated_segments = expand_group_annotations_to_segments(
            annotated_groups, segments_clean, groups
        )
        _save_json(emphasis_path, {
            "method": "std_topic_v2",
            "description": "표준편차 + 가중치 키워드 + 주제 키워드 반복",
            "keyword_report": {
                "description": "강조 감지에 사용된 주제 키워드",
                "topic_keywords": {
                    "keywords": sorted(topic_kw_set),
                    "count": len(topic_kw_set),
                },
            },
            "statistics": {
                "total_count": len(emphasis_sections),
                "ratio": round(len(emphasis_sections) / len(groups), 3) if groups else 0,
                "duration": round(sum(s["end"] - s["start"] for s in emphasis_sections), 2),
            },
            "topic_keywords": sorted(topic_kw_set),
            "emphasis_sections": emphasis_sections,
        })
        print(f"    ✓ 강조 {len(emphasis_sections)}개  ({time.time()-t0:.1f}초)")
    finally:
        Path(audio_path_temp).unlink(missing_ok=True)

    _done("오디오 파이프라인", 0.0)
    return {
        "segments_path": str(segments_path),
        "silences_path": str(silences_path),
        "deictics_path": str(output_dir / f"{stem}_deictics.json"),
        "emphasis_path": str(emphasis_path),
        "annotated_segments": annotated_segments,
        "annotated_groups": annotated_groups,
        "slides_structure": slides_structure,
        "slide_ranges": slide_ranges,
        "duration": duration,
    }


def stage4a_classify(
    args, textualized_path: str, meta_path: str, silences_path: str, output_dir: Path
) -> dict:
    from slide_classifier import classify_slides

    stem = Path(args.input).stem
    classified_path = output_dir / f"{stem}_slide_classified.json"

    if _is_done(classified_path, "Stage 4A 슬라이드 분류", args.force):
        return {"classified_path": str(classified_path), "elapsed": 0.0}

    _banner("Stage 4A  —  슬라이드 분류  (slide_classifier)")
    t0 = time.time()
    classified = classify_slides(
        textualized_path=textualized_path,
        metadata_path=meta_path,
        silences_path=silences_path,
        output_path=str(classified_path),
    )
    elapsed = time.time() - t0
    _done(f"슬라이드 {len(classified)}개 분류", elapsed)
    return {"classified_path": str(classified_path), "elapsed": elapsed}


def stage4b_save_by_slide(args, audio_result: dict, output_dir: Path) -> dict:
    stem = Path(args.input).stem
    by_slide_path = output_dir / f"{stem}_by_slide.json"

    slides_structure = audio_result.get("slides_structure")
    annotated_segments = audio_result.get("annotated_segments", [])
    annotated_groups = audio_result.get("annotated_groups", [])
    slide_ranges = audio_result.get("slide_ranges", [])
    duration = audio_result.get("duration", 0.0)

    if not slides_structure or not slide_ranges:
        return {}

    if _is_done(by_slide_path, "Stage 4B by_slide 저장", args.force):
        return {
            "by_slide_path": str(by_slide_path),
            "elapsed": 0.0,
        }

    _banner("Stage 4B  —  by_slide 구조 저장")
    t0 = time.time()

    slides_with_emphasis = json.loads(json.dumps(slides_structure))
    if annotated_groups:
        annot_by_start = {g.get("start"): g for g in annotated_groups}
        for slide in slides_with_emphasis:
            new_contexts = []
            for ctx in slide.get("contexts", []):
                ann = annot_by_start.get(ctx.get("start"))
                ordered_ctx = {
                    "context_index": ctx.get("context_index"),
                    "start": ctx.get("start"),
                    "end": ctx.get("end"),
                    "text": ctx.get("text"),
                }
                if ann and ann.get("emphasis") == "강조":
                    ordered_ctx["emphasis"] = {
                        "state": "강조",
                        "detected": True,
                        "detail": _format_emphasis_reason(ann),
                    }
                else:
                    ordered_ctx["emphasis"] = {"state": None, "detected": False}
                ordered_ctx["segment_indices"] = ctx.get("segment_indices", [])
                ordered_ctx["segments"] = ctx.get("segments", [])
                new_contexts.append(ordered_ctx)
            slide["contexts"] = new_contexts
    _save_json(by_slide_path, {"slides": slides_with_emphasis})

    elapsed = time.time() - t0
    _done("by_slide 구조 저장", elapsed)
    return {
        "by_slide_path": str(by_slide_path),
        "elapsed": elapsed,
    }


def stage5_fusion(
    args,
    textualized_path: str,
    annotation_path: str,
    audio_result: dict,
    output_dir: Path,
) -> dict:
    from fusion import Config as FusionConfig, run_fusion

    stem = Path(args.input).stem
    fused_path = output_dir / f"{stem}_fused.json"

    if _is_done(fused_path, "Stage 5 퓨전", args.force):
        return {"fused_path": str(fused_path), "elapsed": 0.0}

    _banner("Stage 5  —  퓨전  (fusion)")
    t0 = time.time()

    cfg = FusionConfig(
        stem=stem,
        output_dir=output_dir,
        slides_dir=Path(args.slides),
        audio_path=output_dir / f"{stem}_by_slide.json",
        classified_path=output_dir / f"{stem}_slide_classified.json",
        annotation_path=Path(annotation_path),
        output_path=fused_path,
    )
    fused_output = run_fusion(cfg)

    # run_fusion 반환 스키마를 메인 파이프라인 출력 형식에 맞게 감싼다.
    _save_json(
        fused_path,
        {
            "video_path": args.input,
            "description": "영상(slide+annotation) + 오디오 퓨전 결과",
            "slide_count": len(fused_output.get("slides", [])),
            "fusion_metadata": fused_output.get("metadata", {}),
            "slides": fused_output.get("slides", []),
        },
    )
    elapsed = time.time() - t0
    _done(f"슬라이드 {len(fused_output.get('slides', []))}개 퓨전", elapsed)
    return {"fused_path": str(fused_path), "elapsed": elapsed}


def stage6_graph_triples(args, output_dir: Path, slides_dir: Path) -> dict:
    from json_to_graph_triples import Config as TripleConfig, GraphPipeline

    stem = Path(args.input).stem
    csv_path = output_dir / f"{stem}_graph_triples.csv"

    if _is_done(csv_path, "Stage 6 그래프 트리플 생성", args.force):
        return {"csv_path": str(csv_path), "elapsed": 0.0}

    _banner("Stage 6  —  그래프 트리플 생성  (json_to_graph_triples)")
    t0 = time.time()

    cfg = TripleConfig(
        stem=stem,
        output_dir=output_dir,
        slides_dir=slides_dir,
    )
    GraphPipeline(cfg).run()

    elapsed = time.time() - t0
    _done("그래프 트리플 생성", elapsed)
    return {"csv_path": str(csv_path), "elapsed": elapsed}


def stage7_graph_import(args, csv_path: str, output_dir: Path, slides_dir: Path) -> dict:
    from import_graph import import_graph, resolve_csv_path

    stem = Path(args.input).stem
    resolved_csv = Path(csv_path) if csv_path else resolve_csv_path(stem, output_dir, slides_dir)

    if not resolved_csv.exists():
        raise FileNotFoundError(f"그래프 CSV 파일 없음: {resolved_csv}")

    _banner("Stage 7  —  그래프 DB 적재  (import_graph)")
    t0 = time.time()
    import_graph(resolved_csv, reset=args.neo4j_reset)
    elapsed = time.time() - t0
    _done("그래프 DB 적재", elapsed)
    return {"csv_path": str(resolved_csv), "elapsed": elapsed}


# ──────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────

def run_pipeline(args):
    total_start = time.time()
    timings: dict[str, float] = {}

    from config import output_paths, DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    stem = Path(args.input).stem
    slides_dir = Path(args.slides)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    paths = output_paths(stem, output_dir, slides_dir)

    print("\n" + "═" * 70)
    print("  강의 영상 분석 통합 파이프라인")
    print("═" * 70)
    print(f"  입력 영상 : {args.input}")
    print(f"  슬라이드  : {slides_dir}")
    print(f"  출력      : {output_dir}")
    if args.force:
        print("  ⚠️  --force: 모든 단계 강제 재실행")

    try:
        # ── Stage 1 (병렬 A/B) ──
        _banner("Stage 1  —  병렬 실행 (슬라이드 추출 + 오디오 품질 분석)")
        t_parallel = time.time()
        audio_analyze_result: dict = {}

        if args.skip_extract:
            log.info("Stage 1A 건너뜀 (--skip-extract)")
            meta_path = str(paths["metadata"])
            timings["Stage 1A 슬라이드 추출"] = 0.0
            audio_analyze_result = stage1b_audio_analyze(args, output_dir)
            timings["Stage 1B 오디오 품질 분석"] = audio_analyze_result["elapsed"]
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_1a = executor.submit(stage1_extract, args, slides_dir, output_dir)
                future_1b = executor.submit(stage1b_audio_analyze, args, output_dir)
                for future in as_completed([future_1a, future_1b]):
                    if future is future_1a:
                        r1 = future.result()
                        meta_path = r1["meta_path"]
                        timings["Stage 1A 슬라이드 추출"] = r1["elapsed"]
                    else:
                        audio_analyze_result = future.result()
                        timings["Stage 1B 오디오 품질 분석"] = audio_analyze_result["elapsed"]

        duration = audio_analyze_result.get("duration", 0.0)
        timings["Stage 1 병렬 총"] = time.time() - t_parallel
        print(f"\n  ✓ Stage 1 완료  ({timings['Stage 1 병렬 총']:.1f}초)")
        print("─" * 70)

        # ── Stage 2 (직렬) ──
        r2 = stage2_textualize(args, slides_dir, output_dir)
        textualized_path = r2["textualized_path"]
        timings["Stage 2 슬라이드 텍스트화"] = r2["elapsed"]

        # ── Stage 3 (병렬 A/B) ──
        _banner("Stage 3  —  병렬 실행 (annotation + 오디오)")
        t_parallel = time.time()
        audio_result: dict = {}
        annotation_result: dict = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(stage3a_annotation, args, slides_dir, output_dir)
            future_b = executor.submit(stage3b_audio, args, meta_path, textualized_path, duration, output_dir)
            for future in as_completed([future_a, future_b]):
                if future is future_a:
                    annotation_result = future.result()
                    timings["Stage 3A annotation"] = annotation_result["elapsed"]
                else:
                    audio_result = future.result()

        timings["Stage 3 병렬 총"] = time.time() - t_parallel
        print(f"\n  ✓ Stage 3 완료  ({timings['Stage 3 병렬 총']:.1f}초)")
        print("─" * 70)

        # ── Stage 4 (병렬 C/D) ──
        _banner("Stage 4  —  병렬 실행 (classifier + by_slide 저장)")
        t_parallel = time.time()
        classified_result: dict = {}
        by_slide_result: dict = {}

        silences_path = audio_result.get("silences_path", str(paths["silences"]))
        annotation_path = annotation_result.get("annotation_path", str(paths["annotation"]))

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_c = executor.submit(
                stage4a_classify, args, textualized_path, meta_path, silences_path, output_dir
            )
            future_d = executor.submit(stage4b_save_by_slide, args, audio_result, output_dir)
            for future in as_completed([future_c, future_d]):
                if future is future_c:
                    classified_result = future.result()
                    timings["Stage 4A 분류"] = classified_result.get("elapsed", 0.0)
                else:
                    by_slide_result = future.result()
                    timings["Stage 4B by_slide 저장"] = by_slide_result.get("elapsed", 0.0)

        timings["Stage 4 병렬 총"] = time.time() - t_parallel
        print(f"\n  ✓ Stage 4 완료  ({timings['Stage 4 병렬 총']:.1f}초)")
        print("─" * 70)

        # ── Stage 5 (직렬) ──
        r5 = stage5_fusion(
            args,
            textualized_path=textualized_path,
            annotation_path=annotation_path,
            audio_result=audio_result,
            output_dir=output_dir,
        )
        timings["Stage 5 퓨전"] = r5["elapsed"]

        graph_csv_path = ""
        if args.skip_graph_triples:
            print("\n  ⏭  Stage 6 그래프 트리플 생성 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 6 그래프 트리플"] = 0.0
        else:
            r6 = stage6_graph_triples(args, output_dir, slides_dir)
            graph_csv_path = r6["csv_path"]
            timings["Stage 6 그래프 트리플"] = r6["elapsed"]

        if args.skip_graph_import:
            print("\n  ⏭  Stage 7 그래프 DB 적재 — 사용자 옵션으로 스킵")
            print("─" * 70)
            timings["Stage 7 그래프 적재"] = 0.0
        else:
            r7 = stage7_graph_import(args, graph_csv_path, output_dir, slides_dir)
            timings["Stage 7 그래프 적재"] = r7["elapsed"]

        # ── 생성된 파일 목록 ──
        print("\n  생성된 파일:")
        output_files = [
            audio_result.get("segments_path", ""),
            audio_result.get("silences_path", ""),
            audio_result.get("deictics_path", ""),
            audio_result.get("emphasis_path", ""),
            annotation_path,
            textualized_path,
            classified_result.get("classified_path", ""),
            by_slide_result.get("by_slide_path", ""),
            r5.get("fused_path", ""),
            graph_csv_path,
        ]
        for path_str in output_files:
            if not path_str:
                continue
            p = Path(path_str)
            print(f"    {'✓' if p.exists() else '✗'}  {p}")
        print()

    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise

    finally:
        # 성공/실패 무관하게 항상 타이밍 출력
        total_elapsed = time.time() - total_start
        print("\n" + "═" * 70)
        print("  단계별 소요 시간 (현재까지)")
        print("═" * 70)
        for stage, t in timings.items():
            label = "  (스킵)" if t == 0.0 else f"  {t:>7.1f}초"
            print(f"    {stage:<30} {label}")
        print(f"\n    {'총 소요 시간':<30}  {total_elapsed:>7.1f}초")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    from config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="강의 영상 분석 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --input input/lecture.mp4
  python main.py --input input/lecture.mp4 --output output/ --slides output_slides/
  python main.py --input input/lecture.mp4 --skip-extract
  python main.py --input input/lecture.mp4 --debug --masks
  python main.py --input input/lecture.mp4 --force
  python main.py --input input/lecture.mp4 --skip-graph-import
        """,
    )
    parser.add_argument("--input",  "-i", default="input/lecture.mp4", help="입력 강의 영상 경로 (.mp4)")
    parser.add_argument("--slides", "-s", default=str(DEFAULT_SLIDES_DIR),
                        help=f"슬라이드 프레임 저장 디렉토리 (default: {DEFAULT_SLIDES_DIR})")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"분석 결과 저장 디렉토리 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Stage 1A 건너뜀 (이미 슬라이드가 추출된 경우)")
    parser.add_argument("--force", action="store_true",
                        help="출력 파일이 있어도 모든 단계 강제 재실행")
    parser.add_argument("--retries", type=int, default=3,
                        help="Gemini API 재시도 횟수 (default: 3)")
    parser.add_argument("--debug", action="store_true", help="Stage 1 디버그 로그 출력")
    parser.add_argument("--masks", action="store_true", help="Stage 3A diff 마스크 이미지 저장")
    parser.add_argument("--skip-graph-triples", action="store_true",
                        help="Stage 6 그래프 트리플 CSV 생성 스킵")
    parser.add_argument("--skip-graph-import", action="store_true",
                        help="Stage 7 Neo4j 그래프 적재 스킵")
    parser.add_argument("--neo4j-reset", action="store_true",
                        help="Stage 7 실행 시 기존 그래프 삭제 후 재적재")

    args = parser.parse_args()

    if not args.skip_extract and not Path(args.input).exists():
        print(f"❌ 입력 영상 없음: {args.input}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()