"""
전사 세그먼트를 맥락/문장·문단 단위로 묶기

1차: 시간/문장부호 기반으로 "호흡 단위" 그룹 생성
2차: LLM(Gemini)을 이용해 인접 그룹의 의미/주제가 이어지면 다시 병합

- 쉼/침묵: 인접 세그먼트 간 간격이 짧으면 같은 호흡으로 묶음
- 문장 끝: 마침표·물음표·느낌표 등에서 문장 경계로 분리
- 문단: 긴 침묵에서 문단 경계로 분리
- 최대 길이: 한 묶음이 너무 길어지지 않도록 상한
- 의미 기반 병합: LLM이 "같은 주제/맥락의 설명"이라고 판단하면 인접 그룹 병합
"""

import json
import re

from google.genai import types

from config import gemini_client_2
from utils import api_call_with_retry


# 문장 종결로 볼 문장부호 (한국어·영어·공백 제거 후 끝)
SENTENCE_END_PATTERN = re.compile(r".*[.!?\u3002\uFF01\uFF1F]\s*$")

# 그룹 상한: 이 시간(초)을 넘으면 새 그룹 시작
MAX_GROUP_DURATION = 30.0
# 이 간격(초) 이상이면 새 그룹 (문단/호흡 끊김)
PAUSE_PARAGRAPH = 1.2
# 이 간격(초) 이상이면 문장 경계 후보 (같은 문장이 아닐 가능성)
PAUSE_SENTENCE = 0.6
# 문장 끝 부호가 있으면 이 간격 이하여도 새 문장으로 분리 가능
PAUSE_AFTER_SENTENCE_END = 0.4


def _build_group_list_for_prompt(groups: list[dict], max_chars: int = 220) -> str:
    """LLM 프롬프트용 그룹 요약 문자열 생성"""
    lines = []
    for i, g in enumerate(groups):
        text = (g.get("text") or "").replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars].rstrip() + "..."
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def decide_semantic_merges(groups: list[dict]) -> set[int]:
    """
    인접 그룹이 같은 주제/맥락인지 LLM으로 판단하여
    병합할 인덱스(i, i+1 병합)를 반환.
    """
    if len(groups) <= 1:
        return set()

    group_list = _build_group_list_for_prompt(groups)

    prompt = f"""당신은 대학 강의 전사를 분석하는 도우미입니다.
아래는 시간과 문장부호 기준으로 1차로 나눈 '맥락 그룹' 목록입니다.
각 그룹은 동일한 강의의 연속된 구간입니다.

목표: 인접한 두 그룹이 사실상 같은 주제/맥락의 설명이라면 하나로 다시 합치고 싶습니다.

특히, 다음과 같은 경우는 합쳐야 합니다.
- 같은 개념을 계속 설명하는데 시간/문장부호 때문에 잘린 경우
- 어떤 개념을 설명한 뒤 바로 그 예시/응용을 설명하는 경우

다음과 같은 경우는 합치지 마세요.
- 전혀 다른 개념/챕터/문단으로 넘어가는 경우
- 문제 풀이에서 완전히 다른 문제로 넘어가는 경우

아래 형식의 목록을 보고, i번째 그룹과 i+1번째 그룹을 합칠지 여부만 판단하세요.

그룹 목록:
{group_list}

출력은 다음 JSON 형식으로만 작성하세요:

```json
{{ "merge_after": [0, 1, 5] }}
```

여기서 merge_after의 각 숫자 i는 "그룹 i와 그룹 i+1을 병합하라"는 의미입니다.
설명은 쓰지 말고 JSON만 출력하세요.
"""

    def call_api():
        return gemini_client_2.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

    try:
        response = api_call_with_retry(call_api)
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        merge_after = data.get("merge_after", [])
        return {int(i) for i in merge_after if 0 <= int(i) < len(groups) - 1}
    except Exception:
        # 실패 시 의미 기반 병합 없이 진행
        return set()


def decide_merge_pair(current_text: str, next_text: str, max_chars: int = 400) -> bool:
    """
    두 인접 구간(current_text, next_text)이 같은 맥락/설명의 연속인지 LLM으로 판단.
    - True: 같은 맥락으로 보고 병합
    - False: 다른 맥락으로 보고 분리
    """
    cur = (current_text or "").replace("\n", " ").strip()
    nxt = (next_text or "").replace("\n", " ").strip()
    if not cur or not nxt:
        return False
    if len(cur) > max_chars:
        cur = cur[:max_chars].rstrip() + "..."
    if len(nxt) > max_chars:
        nxt = nxt[:max_chars].rstrip() + "..."

    prompt = f"""당신은 대학 강의 전사를 분석하는 도우미입니다.

아래에는 같은 슬라이드 안에서 연속된 두 구간의 전사 텍스트가 주어집니다.

[이전 구간] (이미 하나의 맥락으로 묶여 있는 설명 전체)
---
{cur}
---

[다음 구간] (바로 이어지는 설명)
---
{nxt}
---

판단 기준:

1. 다음과 같은 경우라면 "같은 맥락의 설명"으로 보고 합쳐야 합니다.
   - 이전 구간에서 소개한 개념/정의/기능에 대한 설명이 다음 구간에서 계속 이어지는 경우
   - 다음 구간이 이전 구간의 내용을 다시 말하거나 정리하는 경우
     (예: "다시 말해서", "다시 말하면", "정리하면", "즉", "한마디로" 등으로 시작)
   - 다음 구간이 이전 구간에서 소개한 개념의 예시/응용/비유를 설명하는 경우
     (예: "예를 들어", "예를 들면", "하나의 예로" 등)
   - 이전 구간이 "~에 대해 알아보겠습니다", "~의 정의는" 처럼 앞으로 설명할 것을 예고하고,
     다음 구간이 그 실제 설명인 경우

2. 다음과 같은 경우라면 "다른 맥락"으로 보고 합치지 말아야 합니다.
   - 이전 구간과 완전히 다른 개념/챕터/슬라이드의 내용을 시작하는 경우
   - 강의의 주요 주제(예: 운영체제, 프로세스, 메모리, 파일, 자원, 사용자 등)와 관계없는
     짧은 잡담, 수업 운영 멘트(출석, 쉬는 시간 안내, 농담 등)만으로 이루어진 경우
   - 이전 구간의 설명이 사실상 끝났고, 새로운 주제를 소개하는 경우
     (예: "다음으로", "이제 다른 주제로", "이제 ~를 보겠습니다" 등)

출력 형식:
- 두 구간이 같은 맥락의 하나의 설명으로 이어져 있다고 판단되면
  MERGE
  라고만 출력하세요.
- 서로 다른 맥락이라고 판단되면
  SPLIT
  라고만 출력하세요.

추가 설명이나 다른 문장은 쓰지 말고, "MERGE" 또는 "SPLIT" 중 하나만 출력하세요."""

    def call_api():
        return gemini_client_2.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )

    try:
        response = api_call_with_retry(call_api)
        text = (response.text or "").strip().upper()
        if "MERGE" in text and "SPLIT" not in text:
            return True
        if "SPLIT" in text and "MERGE" not in text:
            return False
        # 애매하면 보수적으로 분리
        return False
    except Exception:
        # 실패 시 보수적으로 분리
        return False


def group_segments_by_context(
    segments: list[dict],
    *,
    max_group_duration: float = MAX_GROUP_DURATION,
    pause_paragraph: float = PAUSE_PARAGRAPH,
    pause_sentence: float = PAUSE_SENTENCE,
    pause_after_sentence_end: float = PAUSE_AFTER_SENTENCE_END,
    use_semantic_merge: bool = True,
) -> list[dict]:
    """
    인접 세그먼트를 맥락/문장·문단 단위로 묶는다.

    각 세그먼트는 { 'start', 'end', 'text' } 형태.
    반환되는 각 그룹은:
      - start, end: 구간 시간
      - text: 묶인 텍스트 (공백으로 연결)
      - segment_indices: 원본 segments 인덱스 리스트
    """
    if not segments:
        return []

    # 1차: 시간/문장부호 기반 그룹화
    groups: list[dict] = []
    current_indices = [0]
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    current_texts = [segments[0]["text"].strip()]

    for i in range(1, len(segments)):
        seg = segments[i]
        prev_end = segments[i - 1]["end"]
        gap = seg["start"] - prev_end
        prev_text = (segments[i - 1]["text"] or "").strip()
        prev_ends_sentence = bool(SENTENCE_END_PATTERN.match(prev_text))

        # 문단 수준 침묵 -> 무조건 새 그룹
        if gap >= pause_paragraph:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 그룹 길이 상한 초과 -> 새 그룹
        if (seg["end"] - current_start) > max_group_duration:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 문장 끝 + (짧은 침묵이거나 다음으로 이어짐) -> 문장 경계로 새 그룹
        if prev_ends_sentence and gap >= pause_after_sentence_end:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 문장 수준 침묵 (문장 끝 부호 없음) -> 새 그룹
        if gap >= pause_sentence:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 같은 호흡/문맥으로 묶음 (1차 기준)
        current_indices.append(i)
        current_end = seg["end"]
        current_texts.append(seg["text"].strip())

    if current_indices:
        text = " ".join(t for t in current_texts if t)
        groups.append({
            "start": current_start,
            "end": current_end,
            "text": text,
            "segment_indices": current_indices.copy(),
        })

    # 2차: 의미/주제 기반 병합 (LLM)
    if not use_semantic_merge or len(groups) <= 1:
        return groups

    merge_after = decide_semantic_merges(groups)
    if not merge_after:
        return groups

    merged: list[dict] = []
    current = {
        "start": groups[0]["start"],
        "end": groups[0]["end"],
        "text": groups[0]["text"],
        "segment_indices": list(groups[0]["segment_indices"]),
    }

    for i in range(len(groups) - 1):
        g_next = groups[i + 1]

        if i in merge_after:
            # 의미적으로 같은 맥락 -> 병합
            current["end"] = g_next["end"]
            current["text"] = f"{current['text']} {g_next['text']}".strip()
            current["segment_indices"].extend(g_next["segment_indices"])
        else:
            merged.append(current)
            current = {
                "start": g_next["start"],
                "end": g_next["end"],
                "text": g_next["text"],
                "segment_indices": list(g_next["segment_indices"]),
            }

    if current:
        merged.append(current)

    return merged


def expand_group_annotations_to_segments(
    annotated_groups: list[dict],
    segments: list[dict],
    groups: list[dict],
) -> list[dict]:
    """
    그룹 단위 강조 결과를 원본 세그먼트 단위로 펼친다.

    - annotated_groups: 그룹별 강조 정보가 붙은 리스트 (각 항목에 'start'로 그룹 식별)
    - segments: 원본 전사 세그먼트 리스트
    - groups: group_segments_by_context() 반환값 (각 그룹에 segment_indices 있음)

    반환: segments와 같은 길이·순서의 리스트. 각 세그먼트에 해당 그룹의
    emphasis, emphasis_score 등이 복사됨.
    """
    group_by_start = {g["start"]: g for g in groups}
    annotated_by_start = {s["start"]: s for s in annotated_groups}

    # segment index -> annotated group (that contains this segment)
    index_to_annotated = {}
    for g in groups:
        start = g["start"]
        ann = annotated_by_start.get(start)
        if ann is None:
            continue
        for idx in g["segment_indices"]:
            index_to_annotated[idx] = ann

    result = []
    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        ann = index_to_annotated.get(i)
        if ann:
            for key in (
                "emphasis",
                "emphasis_detected",
                "emphasis_score",
                "emphasis_methods",
                "emphasis_reasons",
                "detection_count",
                "confidence",
                "emphasis_signals",
                "emphasis_keywords",
                "emphasis_keywords_by_method",
                "emphasis_detail",
            ):
                if key in ann:
                    seg_copy[key] = ann[key]
        else:
            seg_copy["emphasis"] = None
            seg_copy["emphasis_detected"] = False
            seg_copy["emphasis_score"] = 0
        result.append(seg_copy)

    return result


# ---------------------------------------------------------------------------
# 슬라이드 기반 그룹화 (metadata.json 사용)
# ---------------------------------------------------------------------------

def load_slide_ranges(metadata_path: str, duration_sec: float) -> list[dict]:
    """
    metadata.json에서 슬라이드별 시간 구간 계산.
    같은 slide_index의 base(annot_index=0)가 한 슬라이드 시작.
    반환: [ {"slide_index": 1, "start_sec": 0.07, "end_sec": 46.73}, ... ]
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    bases = [x for x in items if x.get("annot_index") == 0 or x.get("capture_type") == "base"]
    bases = sorted(bases, key=lambda x: (x["slide_index"], x["timestamp_sec"]))
    seen = set()
    unique_bases = []
    for b in bases:
        if b["slide_index"] in seen:
            continue
        seen.add(b["slide_index"])
        unique_bases.append({"slide_index": b["slide_index"], "timestamp_sec": b["timestamp_sec"]})
    unique_bases.sort(key=lambda x: x["timestamp_sec"])
    ranges = []
    for i, b in enumerate(unique_bases):
        start = b["timestamp_sec"]
        end = unique_bases[i + 1]["timestamp_sec"] if i + 1 < len(unique_bases) else duration_sec
        ranges.append({"slide_index": b["slide_index"], "start_sec": start, "end_sec": end})
    return ranges


def group_segments_by_slide_and_context(
    segments: list[dict],
    slide_ranges: list[dict],
    duration_sec: float,
    *,
    use_llm_merge: bool = True,
    use_pause_sentence: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    먼저 슬라이드별로 세그먼트를 나누고, 각 슬라이드 내부에서 LLM 의미 병합으로 컨텍스트 구성.
    use_pause_sentence: True면 침묵/문장끝 기준 분할 추가 (나중에 사용할 옵션).

    반환: (groups_flat, slides_structure)
    - groups_flat: 강조 분석용 그룹 리스트 (start, end, text, segment_indices, slide_index, context_index_in_slide)
    - slides_structure: 최종 JSON용 [ { slide_index, start_sec, end_sec, text, contexts: [ { context_index, start, end, text, segment_indices, segments: [...] } ] } ]
    """
    if not segments or not slide_ranges:
        return [], []

    seg_to_slide = []
    for i, seg in enumerate(segments):
        t = seg["start"]
        slide_idx = None
        for r in slide_ranges:
            if r["start_sec"] <= t < r["end_sec"]:
                slide_idx = r["slide_index"]
                break
        if slide_idx is None and slide_ranges:
            if t < slide_ranges[0]["start_sec"]:
                slide_idx = slide_ranges[0]["slide_index"]
            else:
                slide_idx = slide_ranges[-1]["slide_index"]
        seg_to_slide.append(slide_idx)

    slides_structure = []
    groups_flat = []

    for r in slide_ranges:
        sidx = r["slide_index"]
        start_sec = r["start_sec"]
        end_sec = r["end_sec"]
        indices_in_slide = [i for i in range(len(segments)) if seg_to_slide[i] == sidx]
        if not indices_in_slide:
            slides_structure.append({
                "slide_index": sidx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": "",
                "contexts": [],
            })
            continue

        slide_segments = [segments[i] for i in indices_in_slide]
        initial_groups = []
        for k, i in enumerate(indices_in_slide):
            seg = segments[i]
            initial_groups.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": (seg.get("text") or "").strip(),
                "segment_indices": [i],
            })

        if use_pause_sentence:
            pass

        if use_llm_merge and len(initial_groups) > 1:
            merge_after = decide_semantic_merges(initial_groups)
            merged = []
            cur = {
                "start": initial_groups[0]["start"],
                "end": initial_groups[0]["end"],
                "text": initial_groups[0]["text"],
                "segment_indices": list(initial_groups[0]["segment_indices"]),
            }
            for j in range(len(initial_groups) - 1):
                g_next = initial_groups[j + 1]
                if j in merge_after:
                    cur["end"] = g_next["end"]
                    cur["text"] = f"{cur['text']} {g_next['text']}".strip()
                    cur["segment_indices"].extend(g_next["segment_indices"])
                else:
                    merged.append(cur)
                    cur = {
                        "start": g_next["start"],
                        "end": g_next["end"],
                        "text": g_next["text"],
                        "segment_indices": list(g_next["segment_indices"]),
                    }
            merged.append(cur)
            context_groups = merged
        else:
            context_groups = initial_groups

        slide_text = " ".join((g["text"] for g in context_groups if g["text"])).strip()
        contexts_for_slide = []
        for cix, g in enumerate(context_groups):
            g_flat = {
                "start": g["start"],
                "end": g["end"],
                "text": g["text"],
                "segment_indices": g["segment_indices"],
                "slide_index": sidx,
                "context_index_in_slide": cix,
            }
            groups_flat.append(g_flat)
            segs_in_context = [
                {"start": segments[i]["start"], "end": segments[i]["end"], "text": (segments[i].get("text") or "").strip()}
                for i in g["segment_indices"]
            ]
            contexts_for_slide.append({
                "context_index": cix,
                "start": g["start"],
                "end": g["end"],
                "text": g["text"],
                "segment_indices": g["segment_indices"],
                "segments": segs_in_context,
            })
        slides_structure.append({
            "slide_index": sidx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": slide_text,
            "contexts": contexts_for_slide,
        })

    return groups_flat, slides_structure


