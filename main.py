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
import re
from collections import Counter
from typing import Any

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
from utils import api_call_with_retry

SRC_DIR = ROOT_DIR / "src"

# 지시 대명사/관형사 + 대상 지시(1글자 포함)
_DEICTICS = {
    # 1글자 지시어(오탐 가능성 있음: 토큰 단위로만 매칭)
    "이", "그", "저",
    # 관형사/형용사형
    "이런", "그런", "저런", "이러한", "그러한", "저러한",
    # 대상 지시
    "이것", "그것", "저것", "이거", "그거", "저거",
    "여기", "거기", "저기",
    "이쪽", "그쪽", "저쪽",
    "이러한것", "그러한것", "저러한것",
}

# 지시 대상(그림/표/수식/코드 등) 후보 — "이 그림", "저 표" 같은 지시구 추출용
_DEICTIC_OBJECTS = {
    "그림", "표", "도표", "그래프", "차트",
    "수식", "식", "공식",
    "코드", "프로그램",
    "사진", "이미지", "화면", "슬라이드", "페이지",
    "부분", "내용", "예시", "예",
}

# 시간/순서 지시어(애매 지시어 후보에서 제외)
_TEMPORAL_DEICTICS = {
    "다음", "다음에", "다음은", "다음으로",
    "이제", "지금", "방금", "아까", "나중", "나중에",
    "이번", "지난", "이전", "이후", "그다음", "그다음에",
}

# 담화 연결(말 이어주기) 표현: "그 다음에"처럼 지시어로 취급하지 않음
_DISCOURSE_NEXT_TOKENS = {
    "다음", "다음에", "그다음", "그다음에",
    "이후", "이후에", "후", "후에",
    "이어서", "연이어", "계속", "이제",
    "때", "때는", "때에", "때문", "때문에",
}

# 조사/어미 일부를 간단히 제거 (지시어 탐지용)
_PARTICLE_SUFFIXES = (
    "으로는", "로는", "으로도", "로도",
    "으로", "로",
    "에서", "에게", "한테", "께",
    "까지", "부터",
    "이나", "나", "이나요", "나요",
    "이랑", "랑", "하고", "과", "와",
    "에는", "에선", "에서", "에",
    "을", "를", "은", "는", "이", "가", "도", "만",
    "요",
)


def _strip_particles(token: str) -> str:
    """
    '여기를' -> '여기', '이거는' -> '이거' 처럼 지시어/대상 뒤에 붙는
    흔한 조사를 보수적으로 제거.
    """
    t = (token or "").strip()
    if len(t) <= 1:
        return t
    for suf in _PARTICLE_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            return t[: -len(suf)]
    return t


def _tokenize_for_deictics(text: str) -> list[str]:
    """
    지시어 추출용 간단 토큰화.
    - 한글/영문/숫자/밑줄 이외 문자는 공백 처리
    - 공백 기준 split
    """
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"[^\w\s\u3131-\uD7A3]", " ", t)
    return [x for x in t.split() if x]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _normalize_word_items(words: Any, chunk_start: float = 0.0) -> list[dict]:
    """
    Whisper word-level 정보를 표준 형태로 정규화.
    반환 항목: {text, normalized, start, end}
    """
    out: list[dict] = []
    if not isinstance(words, list):
        return out
    for w in words:
        if not isinstance(w, dict):
            continue
        raw = str(w.get("word") or w.get("text") or "").strip()
        if not raw:
            continue
        start = _safe_float(w.get("start"), 0.0) + chunk_start
        end = _safe_float(w.get("end"), start) + chunk_start
        out.append(
            {
                "text": raw,
                "normalized": _strip_particles(raw),
                "start": start,
                "end": end,
            }
        )
    return out


def _is_discourse_continuation_deictic(deictic: str, next_token: str | None) -> bool:
    """
    '그 다음에', '이 다음', '저 이후'처럼
    문장 연결/순서 전환 용법이면 지시어 후보에서 제외.
    """
    if deictic not in {"이", "그", "저"}:
        return False
    n = (next_token or "").strip()
    return n in _DISCOURSE_NEXT_TOKENS


def extract_deictics_from_segments(
    segments: list[dict],
    *,
    text_field: str = "text_corrected",
) -> dict:
    """
    세그먼트에서 지시어만 추출해 결과 dict 반환.
    - 1글자 지시어 포함
    - 토큰 단위로만 매칭 (문자열 부분매칭으로 인한 오탐 감소)
    """
    occurrences: list[dict] = []
    per_deictic = Counter()
    phrase_occurrences: list[dict] = []
    per_phrase = Counter()

    for seg_idx, seg in enumerate(segments):
        text = seg.get(text_field) or seg.get("text") or ""
        tokens = _tokenize_for_deictics(text)
        if not tokens:
            continue

        # 조사 제거(정규화) 후 세그먼트 내 지시어 카운트
        normalized = [_strip_particles(t) for t in tokens]
        seg_counter = Counter(t for t in normalized if t in _DEICTICS)
        word_items = _normalize_word_items(seg.get("words"), chunk_start=0.0)

        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)

        # (A) 지시어 occurrences (가능하면 단어 단위 타임스탬프 사용)
        if word_items:
            norm_words = [wi["normalized"] for wi in word_items]
            for wi_idx, wi in enumerate(word_items):
                d = wi["normalized"]
                if d not in _DEICTICS:
                    continue
                next_tok = norm_words[wi_idx + 1] if wi_idx + 1 < len(norm_words) else None
                if _is_discourse_continuation_deictic(d, next_tok):
                    continue
                per_deictic[d] += 1
                occurrences.append(
                    {
                        "index": len(occurrences),
                        "deictic": d,
                        "surface": wi["text"],
                        "segment_index": seg_idx,
                        "segment_start": start,
                        "segment_end": end,
                        "deictic_time": wi["start"],
                    }
                )
        elif seg_counter:
            # word timestamp가 없을 때 fallback: 세그먼트 내 균등 분할 근사
            deictic_tokens = [
                (idx, t) for idx, t in enumerate(normalized) if t in _DEICTICS
            ]
            n = len(deictic_tokens)
            if n <= 0:
                n = 1
            for rank, (orig_idx, d) in enumerate(deictic_tokens):
                next_tok = normalized[orig_idx + 1] if orig_idx + 1 < len(normalized) else None
                if _is_discourse_continuation_deictic(d, next_tok):
                    continue
                per_deictic[d] += 1
                t = start + ((rank + 1) / (n + 1)) * max(0.0, end - start)
                occurrences.append(
                    {
                        "index": len(occurrences),
                        "deictic": d,
                        "surface": d,
                        "segment_index": seg_idx,
                        "segment_start": start,
                        "segment_end": end,
                        "deictic_time": t,
                        "time_source": "estimated_from_segment",
                    }
                )

        # (B) 지시구(phrase): 지시어 + 대상명사
        # 예: ["이", "그림"] / ["저", "표"] / ["이런", "예시"] ...
        for i in range(len(normalized) - 1):
            d = normalized[i]
            obj = normalized[i + 1]
            if d in _DEICTICS and obj in _DEICTIC_OBJECTS:
                if _is_discourse_continuation_deictic(d, obj):
                    continue
                phrase = f"{d} {obj}"
                per_phrase[phrase] += 1
                if word_items and i + 1 < len(word_items):
                    phrase_time = word_items[i]["start"]
                else:
                    phrase_time = start
                phrase_occurrences.append(
                    {
                        "index": len(phrase_occurrences),
                        "phrase": phrase,
                        "deictic": d,
                        "object": obj,
                        "segment_index": seg_idx,
                        "start": start,
                        "end": end,
                        "deictic_time": phrase_time,
                    }
                )

    unique_deictics = [
        {"text": d, "count": int(c)}
        for d, c in per_deictic.most_common()
    ]
    unique_phrases = [
        {"text": p, "count": int(c)}
        for p, c in per_phrase.most_common()
    ]

    return {
        "description": "전사 세그먼트에서 지시어(지시 대명사/관형사 + 대상 지시) 추출 결과 (v2)",
        "source_text_field": text_field,
        "segment_count": len(segments),
        "deictic_total_count": int(sum(per_deictic.values())),
        "unique_deictics": unique_deictics,
        "occurrences": occurrences,
        "deictic_phrase_total_count": int(sum(per_phrase.values())),
        "unique_phrases": unique_phrases,
        "phrase_occurrences": phrase_occurrences,
    }


def classify_ambiguous_deictics_with_llm(
    segments: list[dict],
    deictics_report: dict,
    *,
    threshold: float = 0.6,
    context_window: int = 1,
) -> dict:
    """
    LLM으로 지시어 대상을 추론하고 confidence가 낮은 항목만 추출.
    - threshold 미만 또는 inferred_target=null 이면 ambiguous
    """
    from google.genai import types
    from config import gemini_client

    occurrences = deictics_report.get("occurrences") or []
    phrase_occ = deictics_report.get("phrase_occurrences") or []

    # 같은 segment/비슷한 시점에 phrase가 있으면 이미 명확한 경우로 간주
    phrase_keys = {
        (p.get("segment_index"), round(_safe_float(p.get("deictic_time"), 0.0), 2))
        for p in phrase_occ
    }

    checked = 0
    excluded_count = 0
    ambiguous_items: list[dict] = []

    for occ in occurrences:
        seg_idx = int(occ.get("segment_index", -1))
        if not (0 <= seg_idx < len(segments)):
            continue
        deictic = str(occ.get("deictic") or "").strip()
        deictic_time = _safe_float(occ.get("deictic_time"), _safe_float(occ.get("segment_start"), 0.0))
        key = (seg_idx, round(deictic_time, 2))
        if key in phrase_keys:
            # phrase가 포착된 경우는 기본적으로 대상 추론 가능성이 높아 제외
            excluded_count += 1
            continue

        # 1) 시간/순서 지시어는 제외
        if deictic in _TEMPORAL_DEICTICS:
            excluded_count += 1
            continue

        # 2) 주변 세그먼트(±context_window)에서 대상명사가 매칭되면 제외
        s0 = max(0, seg_idx - context_window)
        s1 = min(len(segments), seg_idx + context_window + 1)
        matched_object = None
        matched_segment_index = None
        for i in range(s0, s1):
            txt = (segments[i].get("text_corrected") or segments[i].get("text") or "").strip()
            toks = _tokenize_for_deictics(txt)
            norm_toks = {_strip_particles(t) for t in toks}
            obj_found = next((o for o in _DEICTIC_OBJECTS if o in norm_toks), None)
            if obj_found:
                matched_object = obj_found
                matched_segment_index = i
                break
        if matched_object is not None:
            excluded_count += 1
            continue

        # 3) 위 규칙으로 해결 안 된 경우만 LLM 판정
        checked += 1
        context_lines = []
        for i in range(s0, s1):
            marker = "CURRENT" if i == seg_idx else "NEAR"
            txt = (segments[i].get("text_corrected") or segments[i].get("text") or "").strip()
            context_lines.append(f"[{i}] ({marker}) {txt}")
        context_block = "\n".join(context_lines)

        prompt = f"""당신은 강의 전사에서 지시어가 무엇을 가리키는지 판정하는 분석가입니다.

다음 지시어가 문맥상 가리키는 대상이 명확한지 판단하세요.

- 지시어: {occ.get("deictic")}
- 표면형: {occ.get("surface")}
- 세그먼트 인덱스: {seg_idx}
- 시간: {deictic_time:.3f}초

주변 문맥:
{context_block}

출력은 JSON만:
{{
  "inferred_target": "그림|표|수식|코드|슬라이드|화면|내용|예시|기타|null",
  "confidence": 0.0,
  "reason": "한 줄 설명"
}}
confidence는 0~1 범위 실수.
대상을 특정하기 어렵다면 inferred_target은 null로 하세요."""

        def call_api():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=512),
            )

        inferred_target = None
        confidence = 0.0
        reason = "LLM 판정 실패"
        try:
            resp = api_call_with_retry(call_api)
            txt = (resp.text or "").strip()
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0].strip()
            elif "```" in txt and txt.count("```") >= 2:
                txt = txt.split("```")[1].split("```")[0].strip()
            data = json.loads(txt)
            inferred_target = data.get("inferred_target")
            confidence = _safe_float(data.get("confidence"), 0.0)
            reason = str(data.get("reason") or "").strip() or "사유 없음"
        except Exception:
            pass

        is_ambiguous = (inferred_target in (None, "null", "")) or (confidence < threshold)
        if is_ambiguous:
            ambiguous_items.append(
                {
                    "deictic": deictic,
                    "surface": occ.get("surface"),
                    "segment_index": seg_idx,
                    "segment_start": _safe_float(occ.get("segment_start"), 0.0),
                    "segment_end": _safe_float(occ.get("segment_end"), 0.0),
                    "deictic_time": deictic_time,
                    "excluded_by_rule_reason": None,
                    "context_match": None,
                    "llm_used": True,
                    "inferred_target": None if inferred_target in (None, "null", "") else inferred_target,
                    "confidence": confidence,
                    "reason": reason,
                }
            )

    return {
        "description": "문맥상 대상 추론 confidence가 낮은(애매한) 지시어 목록",
        "source_text_field": deictics_report.get("source_text_field", "text_corrected"),
        "ambiguity_threshold": threshold,
        "context_window": context_window,
        "excluded_by_rules_count": excluded_count,
        "total_checked": checked,
        "ambiguous_count": len(ambiguous_items),
        "ambiguous_items": ambiguous_items,
    }


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
            words = _normalize_word_items(seg.get("words"), chunk_start=chunk_start)
            segments.append({
                "start": float(seg["start"]) + chunk_start,
                "end": float(seg["end"]) + chunk_start,
                "text": (seg["text"] or "").strip(),
                "words": words,
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

    # 기존 결과 호환 유지:
    # - words(word-level timestamp)는 지시어 추출 내부 계산에만 사용
    # - 기존 산출물에는 포함하지 않음
    segments_clean = [{k: v for k, v in s.items() if k != "words"} for s in segments]

    # 교정 결과 전체를 별도 JSON으로 저장 (원본/교정 텍스트 확인용)
    segments_out_path = output_dir / f"{output_name}_segments_v2.json"
    with open(segments_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "전사 세그먼트 + 원본/교정 텍스트 (v2, 슬라이드별 전사 기반)",
                "video_path": video_path,
                "segment_count": len(segments_clean),
                "segments": segments_clean,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  ✓ 교정 세그먼트 저장: output/{output_name}_segments_v2.json")

    # 지시어(지시 대명사/관형사 + 대상 지시)만 추출한 JSON 저장
    deictics_report = extract_deictics_from_segments(segments, text_field="text_corrected")
    deictics_report["video_path"] = video_path
    deictics_out_path = output_dir / f"{output_name}_deictics_v2.json"
    with open(deictics_out_path, "w", encoding="utf-8") as f:
        json.dump(deictics_report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 지시어 추출 저장: output/{output_name}_deictics_v2.json")

    # LLM으로 대상 추론 confidence가 낮은(애매한) 지시어만 별도 저장
    DEICTIC_AMBIGUITY_THRESHOLD = 0.6
    ambiguous_report = classify_ambiguous_deictics_with_llm(
        segments_clean,
        deictics_report,
        threshold=DEICTIC_AMBIGUITY_THRESHOLD,
        context_window=2,
    )
    ambiguous_report["video_path"] = video_path
    ambiguous_out_path = output_dir / f"{output_name}_deictics_ambiguous_v2.json"
    with open(ambiguous_out_path, "w", encoding="utf-8") as f:
        json.dump(ambiguous_report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 애매 지시어 저장: output/{output_name}_deictics_ambiguous_v2.json")

    # 전사 세그먼트 기준 침묵 구간 JSON 추가 (앞뒤 텍스트 없이 시간 정보만)
    MIN_SILENCE_SEC = 0.6
    silences: list[dict] = []
    for i in range(len(segments_clean) - 1):
        cur = segments_clean[i]
        nxt = segments_clean[i + 1]
        start = float(cur.get("end", 0.0))
        end = float(nxt.get("start", start))
        gap = end - start
        if gap >= MIN_SILENCE_SEC:
            silences.append(
                {
                    "index": len(silences),
                    "start": start,
                    "end": end,
                    "duration": gap,
                    "prev_segment_index": i,
                    "next_segment_index": i + 1,
                }
            )
    total_silence_duration = sum(s["duration"] for s in silences) if silences else 0.0
    silences_out_path = output_dir / f"{output_name}_silences_v2.json"
    with open(silences_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "전사 세그먼트 사이 침묵(무음) 구간 정보 (v2, 슬라이드별 전사 기반)",
                "video_path": video_path,
                "total_duration_sec": duration,
                "segment_count": len(segments_clean),
                "min_silence_sec": MIN_SILENCE_SEC,
                "silence_count": len(silences),
                "total_silence_duration_sec": total_silence_duration,
                "silences": silences,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  ✓ 침묵 구간 저장: output/{output_name}_silences_v2.json")

    # [3] 컨텍스트 그룹핑 + 강조 감지
    print("\n[3/5] 컨텍스트 그룹핑 + 강조 구간 감지...")
    slides_structure = None
    slide_ranges = None
    if metadata_path and Path(metadata_path).is_file():
        slide_ranges = load_slide_ranges(metadata_path, duration)
        groups, slides_structure = group_segments_by_slide_and_context(
            segments_clean, slide_ranges, duration, use_pause_sentence=False, use_llm_merge=True
        )
        print(f"  ✓ 슬라이드-컨텍스트 그룹: {len(groups)}개")
    else:
        groups = group_segments_by_context(segments_clean)
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
            std_topic_annotated_groups, segments_clean, groups
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

    # [4] 강의 노트는 현재 비활성화 (코드는 text_processor.py에 유지)
    print("\n[4/5] 강의 정리본 생성 스킵 (비활성화)")

    print(f"\n완료. 소요 시간: {(time.time() - start_time)/60:.1f}분")


if __name__ == "__main__":
    main()
