"""
텍스트 교정 및 강의 정리본 생성
"""

import json

from google.genai import types

from config import gemini_client
from utils import api_call_with_retry


def correct_segments(segments: list[dict], batch_size: int = 100) -> list[dict]:
    """세그먼트 텍스트 교정 (배치 처리)"""
    all_corrections = {}
    total_batches = (len(segments) + batch_size - 1) // batch_size

    if total_batches > 1:
        print(f"  (세그먼트 {len(segments)}개, {total_batches}개 배치로 교정 중...)")

    for batch_idx in range(0, len(segments), batch_size):
        batch = segments[batch_idx:batch_idx + batch_size]
        seg_text = "\n".join([f"[{batch_idx + i}] {s['text']}" for i, s in enumerate(batch)])

        prompt = f"""강의 전사 교정

## 전사 (교정 대상)
{seg_text}

## 출력 (JSON만)
{{"corrections": [{{"index": 0, "text": "교정된 텍스트"}}, ...]}}

규칙:
- 전문용어 오타 수정 (예: 계략적->개략적, Sublet->Servlet)
- 추임새(자, 뭐, 어, 그) 제거
- 자연스러운 문장으로"""

        def call_api():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=16384)
            )

        response = api_call_with_retry(call_api)
        text = response.text.strip()

        # JSON 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
            for c in result.get("corrections", []):
                all_corrections[c["index"]] = c["text"]
        except:
            pass

    # 교정 적용
    corrected = []
    for i, seg in enumerate(segments):
        s = seg.copy()
        if i in all_corrections and all_corrections[i]:
            s["text_original"] = seg["text"]
            s["text"] = all_corrections[i]
        corrected.append(s)

    return corrected


def correct_segments_dual(segments: list[dict], batch_size: int = 100) -> list[dict]:
    """
    세그먼트 텍스트 교정 (2단계):
    1) original_corrected: 전문용어 오타/잘못된 단어만 최소 수정 (구어체/추임새는 유지)
    2) natural: original_corrected를 바탕으로 추임새 제거 + 자연스러운 문장으로 리라이팅

    반환 세그먼트에는 다음 필드가 추가/수정됨:
    - text_raw: 원본 전사 텍스트
    - text_corrected: 1단계 교정 텍스트 (전문용어 오타 수정, 구어체 유지)
    - text_natural: 2단계 교정 텍스트 (추임새 제거 + 자연스러운 문장)
    - text: text_corrected (후속 파이프라인에서 사용할 기본 텍스트)
    """
    total = len(segments)
    if total == 0:
        return []

    # 1단계: 전문용어/오타/잘못된 단어만 최소 수정 (구어체/추임새는 그대로)
    corrections_stage1: dict[int, str] = {}
    total_batches = (total + batch_size - 1) // batch_size
    if total_batches > 1:
        print(f"  (세그먼트 {total}개, {total_batches}개 배치로 1단계 교정 중...)")

    for batch_idx in range(0, total, batch_size):
        batch = segments[batch_idx:batch_idx + batch_size]
        seg_text = "\n".join([f"[{batch_idx + i}] {s['text']}" for i, s in enumerate(batch)])

        prompt = f"""강의 전사 1단계 교정 (전문용어/오타만 최소 수정)

## 전사 (교정 대상)
{seg_text}

## 출력 형식 (JSON만)
{{"corrections": [{{"index": 0, "text": "1단계 교정된 텍스트"}}, ...]}}

규칙:
- 각 index에 해당하는 문장을 "가능한 한 그대로" 유지하면서,
  전문용어 오타, 잘못된 단어, 명백한 잘못된 표현만 자연스럽게 바꿔주세요.
- 문장의 구조, 길이, 어순은 최대한 유지하세요.
- 추임새(자, 뭐, 어, 그 등), 지시어, 구어체 표현은 제거하지 말고 그대로 두세요.
- 문장을 둘로 나누거나 합치지 말고, 한 문장은 한 문장으로 그대로 남겨두세요.
- corrections 배열에는 수정이 실제로 적용된 문장만 포함해도 됩니다.
"""

        def call_api_stage1():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=16384),
            )

        response = api_call_with_retry(call_api_stage1)
        text = (response.text or "").strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text and text.count("```") >= 2:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
            for c in result.get("corrections", []):
                idx = int(c.get("index"))
                new_text = str(c.get("text") or "").strip()
                if 0 <= idx < total and new_text:
                    corrections_stage1[idx] = new_text
        except Exception:
            # 파싱 실패 시 해당 배치는 건너뜀 (원문 유지)
            continue

    # 2단계: 1단계 교정 텍스트를 기반으로 추임새 제거 + 자연스러운 문장으로 리라이팅
    corrections_stage2: dict[int, str] = {}
    if total_batches > 1:
        print(f"  (세그먼트 {total}개, {total_batches}개 배치로 2단계 교정 중...)")

    for batch_idx in range(0, total, batch_size):
        batch = segments[batch_idx:batch_idx + batch_size]
        # 1단계 교정 결과를 입력으로 사용
        lines = []
        for i, s in enumerate(batch):
            global_idx = batch_idx + i
            base_text = corrections_stage1.get(global_idx, s["text"])
            lines.append(f"[{global_idx}] {base_text}")
        seg_text_stage2 = "\n".join(lines)

        prompt2 = f"""강의 전사 2단계 교정 (추임새 제거 + 자연스러운 문장)

## 전사 (2단계 교정 대상)
{seg_text_stage2}

## 출력 형식 (JSON만)
{{"corrections": [{{"index": 0, "text": "2단계 교정된 자연스러운 문장"}}, ...]}}

규칙:
- 입력으로 주어진 각 문장(1단계 교정 텍스트)을 기준으로,
  추임새(자, 뭐, 어, 그 등), 불필요한 구어체, 군더더기 표현을 제거하고
  자연스러운 문장(문어체에 가까운 형태)으로 정리해주세요.
- 전문용어 오타는 이미 1단계에서 수정되었다고 가정하고, 그대로 유지하되,
  문맥상 더 자연스러운 표현이 있다면 바꿔도 됩니다.
- 의미가 달라지지 않도록 주의하면서, 문장을 간결하고 명확하게 만들세요.
- 한 문장은 한 문장으로 유지하되, 너무 긴 경우 적절히 나누어도 괜찮습니다.
- corrections 배열에는 수정이 실제로 적용된 문장만 포함해도 됩니다.
"""

        def call_api_stage2():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt2)],
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=16384),
            )

        response2 = api_call_with_retry(call_api_stage2)
        text2 = (response2.text or "").strip()

        if "```json" in text2:
            text2 = text2.split("```json")[1].split("```")[0].strip()
        elif "```" in text2 and text2.count("```") >= 2:
            text2 = text2.split("```")[1].split("```")[0].strip()

        try:
            result2 = json.loads(text2)
            for c in result2.get("corrections", []):
                idx = int(c.get("index"))
                new_text = str(c.get("text") or "").strip()
                if 0 <= idx < total and new_text:
                    corrections_stage2[idx] = new_text
        except Exception:
            continue

    # 최종 세그먼트 구성: raw / corrected / natural 모두 포함
    corrected_segments: list[dict] = []
    for i, seg in enumerate(segments):
        s = seg.copy()
        raw_text = seg.get("text", "")
        text_corr = corrections_stage1.get(i, raw_text)
        text_nat = corrections_stage2.get(i, text_corr)

        s["text_raw"] = raw_text
        s["text_corrected"] = text_corr
        s["text_natural"] = text_nat
        # 후속 파이프라인(그룹/강조)은 교정된 텍스트를 기본으로 사용하도록 text를 교체
        s["text"] = text_corr

        corrected_segments.append(s)

    return corrected_segments


def _build_slide_context_block(slide_context_by_index: dict[int, dict], slide_indices: set[int], max_chars: int = 1200) -> str:
    """
    슬라이드 텍스트화 정보(slide_textualized.json)를 바탕으로,
    교정용 프롬프트 상단에 넣을 간단한 컨텍스트 블록 생성.
    """
    if not slide_context_by_index or not slide_indices:
        return ""
    parts: list[str] = []
    for sidx in sorted(slide_indices):
        info = slide_context_by_index.get(sidx)
        if not info:
            continue
        title = info.get("title") or ""
        t1 = (info.get("t1") or "").strip()
        t1_structure = (info.get("t1_structure") or "").strip()
        snippet_parts = []
        if t1:
            snippet_parts.append(t1)
        if t1_structure:
            snippet_parts.append(t1_structure)
        snippet = " ".join(snippet_parts)
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "..."
        line = f"- 슬라이드 {sidx}: {title}\n  내용: {snippet}"
        parts.append(line)
        context_text = "\n".join(parts)
        if len(context_text) >= max_chars:
            break
    return "\n".join(parts)


def correct_segments_dual_with_slide_context(
    segments: list[dict],
    slide_context_by_index: dict[int, dict],
    batch_size: int = 100,
) -> list[dict]:
    """
    slide_textualized 기반 슬라이드 컨텍스트를 활용한 2단계 교정 버전.

    segments 각 항목에 'slide_index'가 붙어 있다고 가정하고,
    같은 배치 안에 등장하는 슬라이드들의 제목/텍스트를 프롬프트 상단에 제공해
    전문용어/맥락 기반으로 더 정확한 교정을 유도한다.
    """
    total = len(segments)
    if total == 0:
        return []

    corrections_stage1: dict[int, str] = {}
    total_batches = (total + batch_size - 1) // batch_size
    if total_batches > 1:
        print(f"  (세그먼트 {total}개, {total_batches}개 배치로 1단계 교정 중... / 슬라이드 컨텍스트 포함)")

    # 1단계
    for batch_idx in range(0, total, batch_size):
        batch = segments[batch_idx : batch_idx + batch_size]
        seg_text_lines = []
        slide_indices_in_batch: set[int] = set()
        for i, s in enumerate(batch):
            global_idx = batch_idx + i
            seg_text_lines.append(f"[{global_idx}] {s.get('text', '')}")
            sidx = s.get("slide_index")
            if isinstance(sidx, int):
                slide_indices_in_batch.add(sidx)
        seg_text = "\n".join(seg_text_lines)
        slide_block = _build_slide_context_block(slide_context_by_index, slide_indices_in_batch)

        prompt = f"""강의 전사 1단계 교정 (전문용어/오타만 최소 수정)

## 관련 슬라이드 정보 (전문용어/맥락/배경지식)
{slide_block}

## 전사 (교정 대상)
{seg_text}

## 출력 형식 (JSON만)
{{"corrections": [{{"index": 0, "text": "1단계 교정된 텍스트"}}, ...]}}

규칙:
- 각 index에 해당하는 문장을 "가능한 한 그대로" 유지하면서,
  전문용어 오타, 잘못된 단어, 명백한 잘못된 표현만 자연스럽게 바꿔주세요.
- 문장의 구조, 길이, 어순은 최대한 유지하세요.
- 추임새(자, 뭐, 어, 그 등), 지시어, 구어체 표현은 제거하지 말고 그대로 두세요.
- 문장을 둘로 나누거나 합치지 말고, 한 문장은 한 문장으로 그대로 남겨두세요.
- corrections 배열에는 수정이 실제로 적용된 문장만 포함해도 됩니다.
"""

        def call_api_stage1():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=16384),
            )

        response = api_call_with_retry(call_api_stage1)
        text = (response.text or "").strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text and text.count("```") >= 2:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
            for c in result.get("corrections", []):
                idx = int(c.get("index"))
                new_text = str(c.get("text") or "").strip()
                if 0 <= idx < total and new_text:
                    corrections_stage1[idx] = new_text
        except Exception:
            continue

    # 2단계
    corrections_stage2: dict[int, str] = {}
    if total_batches > 1:
        print(f"  (세그먼트 {total}개, {total_batches}개 배치로 2단계 교정 중... / 슬라이드 컨텍스트 포함)")

    for batch_idx in range(0, total, batch_size):
        batch = segments[batch_idx : batch_idx + batch_size]
        lines = []
        slide_indices_in_batch: set[int] = set()
        for i, s in enumerate(batch):
            global_idx = batch_idx + i
            base_text = corrections_stage1.get(global_idx, s.get("text", ""))
            lines.append(f"[{global_idx}] {base_text}")
            sidx = s.get("slide_index")
            if isinstance(sidx, int):
                slide_indices_in_batch.add(sidx)
        seg_text_stage2 = "\n".join(lines)
        slide_block = _build_slide_context_block(slide_context_by_index, slide_indices_in_batch)

        prompt2 = f"""강의 전사 2단계 교정 (추임새 제거 + 자연스러운 문장)

## 관련 슬라이드 정보 (전문용어/맥락/배경지식)
{slide_block}

## 전사 (2단계 교정 대상)
{seg_text_stage2}

## 출력 형식 (JSON만)
{{"corrections": [{{"index": 0, "text": "2단계 교정된 자연스러운 문장"}}, ...]}}

규칙:
- 입력으로 주어진 각 문장(1단계 교정 텍스트)을 기준으로,
  추임새(자, 뭐, 어, 그 등), 불필요한 구어체, 군더더기 표현을 제거하고
  자연스러운 문장(문어체에 가까운 형태)으로 정리해주세요.
- 전문용어 오타는 이미 1단계에서 수정되었다고 가정하고, 그대로 유지하되,
  문맥상 더 자연스러운 표현이 있다면 바꿔도 됩니다.
- 의미가 달라지지 않도록 주의하면서, 문장을 간결하고 명확하게 만들세요.
- 한 문장은 한 문장으로 유지하되, 너무 긴 경우 적절히 나누어도 괜찮습니다.
- corrections 배열에는 수정이 실제로 적용된 문장만 포함해도 됩니다.
"""

        def call_api_stage2():
            return gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt2)],
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=16384),
            )

        response2 = api_call_with_retry(call_api_stage2)
        text2 = (response2.text or "").strip()

        if "```json" in text2:
            text2 = text2.split("```json")[1].split("```")[0].strip()
        elif "```" in text2 and text2.count("```") >= 2:
            text2 = text2.split("```")[1].split("```")[0].strip()

        try:
            result2 = json.loads(text2)
            for c in result2.get("corrections", []):
                idx = int(c.get("index"))
                new_text = str(c.get("text") or "").strip()
                if 0 <= idx < total and new_text:
                    corrections_stage2[idx] = new_text
        except Exception:
            continue

    corrected_segments: list[dict] = []
    for i, seg in enumerate(segments):
        s = seg.copy()
        raw_text = seg.get("text", "")
        text_corr = corrections_stage1.get(i, raw_text)
        text_nat = corrections_stage2.get(i, text_corr)

        s["text_raw"] = raw_text
        s["text_corrected"] = text_corr
        s["text_natural"] = text_nat
        s["text"] = text_corr

        corrected_segments.append(s)

    return corrected_segments


def generate_lecture_notes(segments: list[dict]) -> str:
    """강의 정리본 생성"""
    full_text = " ".join([s["text"] for s in segments[:500]])

    prompt = f"""당신은 대학 강의 내용을 체계적으로 정리하는 전문가입니다.
아래 강의 전사 내용을 바탕으로 **학습에 최적화된 상세한 강의 노트**를 작성하세요.

## 강의 전사 내용
{full_text[:40000]}

## 작성 지침

### 1. 구조
- **제목**: 강의의 핵심 주제를 반영한 명확한 제목
- **개요**: 강의에서 다루는 내용을 2-3문장으로 요약
- **목차**: 주요 섹션 나열

### 2. 본문 구성 (각 주제별로)
- **개념 정의**: 핵심 개념을 명확하게 정의
- **상세 설명**: 교수자가 설명한 내용을 풀어서 서술
- **예시/비유**: 강의에서 언급된 예시나 비유 포함
- **코드/수식**: 관련 코드나 수식이 있다면 코드 블록으로 표시
- **주의사항**: 교수자가 강조한 주의점이나 흔한 실수

### 3. 추가 요소
- **📌 핵심 포인트**: 각 섹션 끝에 핵심 내용 bullet point로 정리
- **💡 팁**: 실무적 조언이나 추가 팁
- **⚠️ 주의**: 흔히 하는 실수나 주의사항
- **🔗 연관 개념**: 관련된 다른 개념 언급

### 4. 마무리
- **📝 전체 요약**: 강의 내용 전체를 3-5개 핵심 포인트로 요약
- **❓ 복습 질문**: 학습 확인을 위한 질문 3-5개
- **📚 추가 학습**: 더 알아볼 만한 주제 제안

## 작성 규칙
- 마크다운 문법 사용
- 전문 용어는 처음 등장 시 간단한 설명 추가
- 논리적 흐름에 따라 내용 배치
- 불필요한 반복 제거, 핵심만 간결하게

이제 위 지침에 따라 상세하고 체계적인 강의 노트를 작성하세요."""

    def call_api():
        return gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=16384)
        )

    response = api_call_with_retry(call_api)
    return response.text.strip()
