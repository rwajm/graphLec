import json
import re
from collections import Counter
from typing import Any, Optional

from google.genai import types

from config import gemini_client
from utils import api_call_with_retry

_DEICTICS = {
    "이", "그", "저",
    "이런", "그런", "저런", "이러한", "그러한", "저러한",
    "이것", "그것", "저것", "이거", "그거", "저거",
    "여기", "거기", "저기",
    "이쪽", "그쪽", "저쪽",
    "이러한것", "그러한것", "저러한것",
}

_DEICTIC_OBJECTS = {
    "그림", "표", "도표", "그래프", "차트",
    "수식", "식", "공식",
    "코드", "프로그램",
    "사진", "이미지", "화면", "슬라이드", "페이지",
    "부분", "내용", "예시", "예",
}

_TEMPORAL_DEICTICS = {
    "다음", "다음에", "다음은", "다음으로",
    "이제", "지금", "방금", "아까", "나중", "나중에",
    "이번", "지난", "이전", "이후", "그다음", "그다음에",
}

_DISCOURSE_NEXT_TOKENS = {
    "다음", "다음에", "그다음", "그다음에",
    "이후", "이후에", "후", "후에",
    "이어서", "연이어", "계속", "이제",
    "때", "때는", "때에", "때문", "때문에",
}

_PARTICLE_SUFFIXES = (
    "으로는", "로는", "으로도", "로도", "으로", "로",
    "에서", "에게", "한테", "께",
    "까지", "부터",
    "이나", "나", "이나요", "나요",
    "이랑", "랑", "하고", "과", "와",
    "에는", "에선", "에서", "에",
    "을", "를", "은", "는", "이", "가", "도", "만", "요",
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _strip_particles(token: str) -> str:
    t = (token or "").strip()
    if len(t) <= 1:
        return t
    for suf in _PARTICLE_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            return t[: -len(suf)]
    return t


def _tokenize_for_deictics(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"[^\w\s\u3131-\uD7A3]", " ", t)
    return [x for x in t.split() if x]


def normalize_word_items(words: Any, chunk_start: float = 0.0) -> list[dict]:
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
        out.append({"text": raw, "normalized": _strip_particles(raw), "start": start, "end": end})
    return out


def _is_discourse_continuation_deictic(deictic: str, next_token: Optional[str]) -> bool:
    if deictic not in {"이", "그", "저"}:
        return False
    return (next_token or "").strip() in _DISCOURSE_NEXT_TOKENS


def extract_deictics_from_segments(
    segments: list[dict], *, text_field: str = "text_corrected"
) -> dict:
    occurrences: list[dict] = []
    per_deictic: Counter = Counter()
    phrase_occurrences: list[dict] = []
    per_phrase: Counter = Counter()

    for seg_idx, seg in enumerate(segments):
        text = seg.get(text_field) or seg.get("text") or ""
        tokens = _tokenize_for_deictics(text)
        if not tokens:
            continue
        normalized = [_strip_particles(t) for t in tokens]
        seg_counter = Counter(t for t in normalized if t in _DEICTICS)
        word_items = normalize_word_items(seg.get("words"), chunk_start=0.0)
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)

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
                occurrences.append({
                    "index": len(occurrences), "deictic": d, "surface": wi["text"],
                    "segment_index": seg_idx, "segment_start": start, "segment_end": end,
                    "deictic_time": wi["start"],
                })
        elif seg_counter:
            deictic_tokens = [(idx, t) for idx, t in enumerate(normalized) if t in _DEICTICS]
            n = max(len(deictic_tokens), 1)
            for rank, (orig_idx, d) in enumerate(deictic_tokens):
                next_tok = normalized[orig_idx + 1] if orig_idx + 1 < len(normalized) else None
                if _is_discourse_continuation_deictic(d, next_tok):
                    continue
                per_deictic[d] += 1
                t = start + ((rank + 1) / (n + 1)) * max(0.0, end - start)
                occurrences.append({
                    "index": len(occurrences), "deictic": d, "surface": d,
                    "segment_index": seg_idx, "segment_start": start, "segment_end": end,
                    "deictic_time": t, "time_source": "estimated_from_segment",
                })

        for i in range(len(normalized) - 1):
            d = normalized[i]
            obj = normalized[i + 1]
            if d in _DEICTICS and obj in _DEICTIC_OBJECTS:
                if _is_discourse_continuation_deictic(d, obj):
                    continue
                phrase = f"{d} {obj}"
                per_phrase[phrase] += 1
                phrase_time = word_items[i]["start"] if word_items and i < len(word_items) else start
                phrase_occurrences.append({
                    "index": len(phrase_occurrences), "phrase": phrase, "deictic": d, "object": obj,
                    "segment_index": seg_idx, "start": start, "end": end, "deictic_time": phrase_time,
                })

    return {
        "description": "전사 세그먼트에서 지시어 추출 결과",
        "source_text_field": text_field,
        "segment_count": len(segments),
        "deictic_total_count": int(sum(per_deictic.values())),
        "unique_deictics": [{"text": d, "count": int(c)} for d, c in per_deictic.most_common()],
        "occurrences": occurrences,
        "deictic_phrase_total_count": int(sum(per_phrase.values())),
        "unique_phrases": [{"text": p, "count": int(c)} for p, c in per_phrase.most_common()],
        "phrase_occurrences": phrase_occurrences,
    }


def classify_ambiguous_deictics_with_llm(
    segments: list[dict],
    deictics_report: dict,
    *,
    threshold: float = 0.6,
    context_window: int = 1,
) -> dict:
    occurrences = deictics_report.get("occurrences") or []
    phrase_occ = deictics_report.get("phrase_occurrences") or []
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
        if (seg_idx, round(deictic_time, 2)) in phrase_keys:
            excluded_count += 1
            continue
        if deictic in _TEMPORAL_DEICTICS:
            excluded_count += 1
            continue

        s0 = max(0, seg_idx - context_window)
        s1 = min(len(segments), seg_idx + context_window + 1)
        matched_object = None
        for i in range(s0, s1):
            txt = (segments[i].get("text_corrected") or segments[i].get("text") or "").strip()
            toks = _tokenize_for_deictics(txt)
            norm_toks = {_strip_particles(t) for t in toks}
            obj_found = next((o for o in _DEICTIC_OBJECTS if o in norm_toks), None)
            if obj_found:
                matched_object = obj_found
                break
        if matched_object is not None:
            excluded_count += 1
            continue

        checked += 1
        context_lines = []
        for i in range(s0, s1):
            marker = "CURRENT" if i == seg_idx else "NEAR"
            txt = (segments[i].get("text_corrected") or segments[i].get("text") or "").strip()
            context_lines.append(f"[{i}] ({marker}) {txt}")

        prompt = f"""강의 전사에서 지시어가 무엇을 가리키는지 판정하세요.

지시어: {occ.get("deictic")}  표면형: {occ.get("surface")}
세그먼트: {seg_idx} / 시간: {deictic_time:.3f}초

주변 문맥:
{chr(10).join(context_lines)}

출력 JSON만:
{{"inferred_target": "그림|표|수식|코드|슬라이드|화면|내용|예시|기타|null", "confidence": 0.0, "reason": "한 줄 설명"}}"""

        def call_api():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=512),
            )

        inferred_target, confidence, reason = None, 0.0, "LLM 판정 실패"
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

        if (inferred_target in (None, "null", "")) or (confidence < threshold):
            ambiguous_items.append({
                "deictic": deictic, "surface": occ.get("surface"),
                "segment_index": seg_idx,
                "segment_start": _safe_float(occ.get("segment_start"), 0.0),
                "segment_end": _safe_float(occ.get("segment_end"), 0.0),
                "deictic_time": deictic_time,
                "llm_used": True,
                "inferred_target": None if inferred_target in (None, "null", "") else inferred_target,
                "confidence": confidence,
                "reason": reason,
            })

    return {
        "description": "대상 추론 confidence가 낮은 지시어 목록",
        "source_text_field": deictics_report.get("source_text_field", "text_corrected"),
        "ambiguity_threshold": threshold,
        "context_window": context_window,
        "excluded_by_rules_count": excluded_count,
        "total_checked": checked,
        "ambiguous_count": len(ambiguous_items),
        "ambiguous_items": ambiguous_items,
    }
