"""
표준편차 + 주제/내용 키워드 반복 강조 모듈

진단: EMPHASIS_DEBUG=1 python main.py ... 로 실행하면 주제 키워드 진단 결과를 output/ 폴더에 저장.

- 표준편차 기반 오디오 강조 (기존 std)
- 가중치 키워드 강조 (기존 std)
- 주제/내용에 맞는 키워드가 **전사문 전체**에서 반복될 때 강조 신호 추가.

키워드: kiwipiepy 설치 시 형태소 분석으로 **명사(NNG, NNP)만** 추출. 미설치 시 어미 제거 방식으로 대체.
주제와 동떨어진 키워드 제거: LLM(Gemini)으로 강의 흐름·주제에 맞는 키워드만 남김.

max_keywords / min_freq 조정 (기본값 20/5):
  [1] emphasis_detector_std_topic.py
      - _get_topic_keywords(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
      - get_topic_keywords_list(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
      - detect_emphasis_by_topic_keyword_repetition(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
      - diagnose_topic_keywords 내부 _get_topic_keywords(..., min_freq=5, max_keywords=20)  호출 인자
  [2] main.py
      - get_topic_keywords_list(groups, min_freq=5, max_keywords=20, ...)  키워드 보고서용 호출
  위 네 군데를 같은 값으로 맞춰서 바꾸면 됨.
"""

import json
import os
import re
from collections import Counter

from emphasis_detector_std import (
    detect_emphasis_by_std,
    detect_emphasis_by_keywords_weighted,
)

# kiwipiepy 있으면 명사만 추출, 없으면 기존 스템 방식 사용
_KIWI = None

def _get_kiwi():
    global _KIWI
    if _KIWI is None:
        try:
            from kiwipiepy import Kiwi
            _KIWI = Kiwi()
        except ImportError:
            pass
    return _KIWI


# 주제 키워드에서 제외: 조사·대명사·지시어·말버릇·일반적 단어 (내용적으로 특정 주제를 지칭하지 않는 것)
_MINIMAL_STOP = {
    "그", "이", "저", "것", "수", "등", "및", "또는", "그리고",
    "있다", "하다", "되다", "이다", "없다", "않다",
    "위해", "통해", "대한", "있는", "하는", "되는", "라는",
    "이제", "그래서", "그러면", "그럼", "이렇게", "그렇게",
    "어떤", "무슨", "어디", "언제", "왜", "몇", "네", "예", "아니요",
    "이런", "저런", "그런", "여러", "바로", "우리", "다음", "이거", "그거", "저거",
    "이런것", "그런것", "저런것", "무엇", "어떤것",
    "문제", "경우", "방법", "이유", "결과", "사실", "정말", "대부분", "보통", "항상",
    "그런데", "그러나", "따라서", "즉시", "아마", "혹시", "좀", "더", "매우", "너무", "잘", "많이", "적게",
}

# 토큰 끝에서 벗길 어미·조사 (긴 것 우선). 벗긴 뒤 남은 부분이 내용어 후보.
_ENDINGS = (
    "했습니다", "됐습니다", "였습니다", "습니다", "ㅂ니다",
    "입니다", "합니다", "됩니다",
    "이에요", "이예요", "예요", "에요", "네요", "군요", "어요", "아요", "해요", "되요",
    "되는", "하는", "있는", "라는", "이라는", "된", "할", "하게", "하지",
    "는데", "니까", "거나", "지만", "에서", "으로", "고", "면", "며", "죠", "네",
    "부터", "까지", "처럼", "대로", "마저", "조차", "랑", "들",
    "가", "를", "을", "의", "로", "에", "와", "과", "는", "은", "도", "만", "이",
)


def _strip_endings(word: str) -> str:
    """'소프트웨어입니다' -> '소프트웨어' 처럼 어미만 제거. 남은 길이 2미만이면 원형 반환."""
    if not word or len(word) < 2:
        return word
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) > len(ending):
            stem = word[: -len(ending)]
            if len(stem) >= 2:
                return stem
    return word


def _tokenize(text: str) -> list[str]:
    """한국어·영어 혼합 텍스트에서 2글자 이상 의미 단위 추출 (공백·구두점 기준)"""
    text = (text or "").strip()
    text = re.sub(r"[^\w\s\u3131-\uD7A3]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2]


def _content_stems(tokens: list[str], min_length: int = 2) -> list[str]:
    """토큰 리스트에서 어미를 벗겨 내용어 후보만 반환 (중복 포함, 빈도 계산용). Kiwi 미사용 시."""
    stems = []
    for t in tokens:
        if t in _MINIMAL_STOP or len(t) < min_length:
            continue
        s = _strip_endings(t)
        if len(s) >= min_length and s not in _MINIMAL_STOP:
            stems.append(s)
    return stems


# 형태소 분석 시 주제 키워드 후보로 인정할 품사
# - NNG: 일반명사
# - NNP: 고유명사
# - SL : 외국어 (예: "file", "CPU" 등)
_NOUN_TAGS = ("NNG", "NNP", "SL")


def _select_representative_keywords(candidates: set[str]) -> set[str]:
    """
    서로 부분 문자열 관계인 키워드들 중, 가장 긴 형태만 대표로 선택.

    예:
      {'운영', '체제', '운영체제'} -> {'운영체제'}
      {'파일', '입출력', '파일입출력'} -> {'파일입출력'}

    너무 공격적으로 합치지 않기 위해, 단순 부분 문자열 기준만 사용.
    """
    if not candidates:
        return set()

    reps: list[str] = []
    # 길이가 긴 것부터 차례로 살펴보며, 이미 선택된 대표의 부분 문자열이면 버림
    for w in sorted(candidates, key=len, reverse=True):
        if any(w in r and w != r for r in reps):
            # 예: w='운영', r='운영체제' -> '운영'은 버리고 '운영체제'만 유지
            continue
        reps.append(w)
    return set(reps)


def _add_compound_nouns_from_seq(seq: list[str], out: list[str], min_length: int) -> None:
    """
    연속된 명사 시퀀스에서 복합명사 후보를 추가.
    예: ['운영', '체제'] -> '운영체제', ['파일', '입출력'] -> '파일입출력'
    """
    if len(seq) < 2:
        return
    # 최대 3개까지 붙여서 복합명사 후보 생성
    max_n = 3
    n = len(seq)
    for size in range(2, min(max_n, n) + 1):
        for i in range(0, n - size + 1):
            comp = "".join(seq[i : i + size])
            if len(comp) >= min_length and comp not in _MINIMAL_STOP:
                out.append(comp)


def _extract_content_words(text: str, min_length: int = 3) -> list[str]:
    """
    텍스트에서 주제 키워드 후보만 추출.
    - kiwipiepy 있음: 명사(NNG, NNP)만 추출.
    - 없음: 기존 어미 제거 스템 방식.
    """
    text = (text or "").strip()
    if not text:
        return []
    kiwi = _get_kiwi()
    if kiwi is not None:
        try:
            tokens = kiwi.tokenize(text)
            words: list[str] = []
            noun_seq: list[str] = []
            for t in tokens:
                tag = getattr(t, "tag", None)
                form = getattr(t, "form", None)
                if tag in _NOUN_TAGS and form and len(form) >= min_length and form not in _MINIMAL_STOP:
                    # 단일 명사 키워드
                    words.append(form)
                    # 복합명사 후보 시퀀스에 추가
                    noun_seq.append(form)
                else:
                    # 명사 시퀀스가 끊어지면, 지금까지의 시퀀스로 복합명사 생성
                    if noun_seq:
                        _add_compound_nouns_from_seq(noun_seq, words, min_length)
                        noun_seq = []
            # 마지막 시퀀스 처리
            if noun_seq:
                _add_compound_nouns_from_seq(noun_seq, words, min_length)
            return words
        except Exception:
            pass
    tokens = _tokenize(text)
    return _content_stems(tokens, min_length=min_length)


def _get_topic_keywords(
    segments: list[dict],
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_length: int = 2,
) -> set[str]:
    """
    전사문 전체에서 반복 등장하는 **내용어(어미 제거)** 중, 의미 있을 만한 것만 주제 키워드로 추출.

    - min_freq: **전체 전사문에서 그 단어가 나온 총 횟수**. 이 횟수 이상인 단어만 후보에 포함.
      (예: min_freq=2 이면 2번 이상 등장한 단어만 주제 키워드 후보.)
    - max_keywords: 최종적으로 남기는 주제 키워드 **최대 개수** (빈도 순으로 채움).
    - Kiwi 사용 시: 명사(NNG,NNP)만 추출. min_length=2면 '운영','체제' 같은 2글자 명사도 포함.
    - max_segment_ratio: 이 비율**초과**인 구간에 나오는 단어만 제외 (말버릇성 제거).
    """
    if not segments:
        return set()
    total_segments = len(segments)
    counter: Counter = Counter()
    segment_presence: Counter = Counter()

    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_length)
        seen_here = set()
        for s in words:
            counter[s] += 1
            seen_here.add(s)
        for s in seen_here:
            segment_presence[s] += 1

    # 1.0이면 비율 필터 미적용(모든 반복 명사 포함). 0.95면 95% 초과 구간에 나오는 단어만 제외.
    threshold_segments = total_segments if max_segment_ratio >= 1.0 else max(2, int(total_segments * max_segment_ratio))
    topic = set()
    for w, cnt in counter.most_common(max_keywords * 2):
        if cnt < min_freq:
            continue
        if segment_presence[w] > threshold_segments:
            continue
        topic.add(w)
        if len(topic) >= max_keywords:
            break
    # 같은 개념의 여러 표기가 섞여 있을 때 가장 긴 형태만 대표로 남김
    return _select_representative_keywords(topic)


def get_topic_keywords_filtered(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
    use_llm_filter: bool = True,
) -> set[str]:
    """
    주제 키워드를 추출하고 LLM 필터까지 적용한 최종 집합을 반환.
    여러 min_keyword_count 변형을 실행할 때, 키워드 추출/LLM 호출을 1회만 수행하기 위해 사용.
    """
    kw_set = _get_topic_keywords(
        segments,
        min_freq=min_freq,
        max_keywords=max_keywords,
        max_segment_ratio=max_segment_ratio,
        min_length=min_keyword_len,
    )
    if use_llm_filter and kw_set:
        kw_set = _filter_topic_keywords_by_llm(segments, kw_set)
    return kw_set


def get_topic_keywords_filtered_v2(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
    candidate_pool_size: int = 80,
    use_llm_filter: bool = True,
) -> set[str]:
    """
    v2: **전체 후보를 더 많이 모은 뒤** LLM으로 주제 무관만 제거하고,
    남은 것 중 **빈도 순 상위 max_keywords(20)개**를 반환.
    (기존은 상위 20개 먼저 뽑고 LLM 제거해서 20개 미만이 나올 수 있음.)
    """
    if not segments:
        return set()
    total_segments = len(segments)
    counter: Counter = Counter()
    segment_presence: Counter = Counter()
    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_keyword_len)
        seen_here = set()
        for s in words:
            counter[s] += 1
            seen_here.add(s)
        for s in seen_here:
            segment_presence[s] += 1

    threshold_segments = total_segments if max_segment_ratio >= 1.0 else max(2, int(total_segments * max_segment_ratio))
    candidate_pool: set[str] = set()
    for w, cnt in counter.most_common(candidate_pool_size * 3):
        if cnt < min_freq:
            continue
        if segment_presence[w] > threshold_segments:
            continue
        candidate_pool.add(w)
        if len(candidate_pool) >= candidate_pool_size:
            break

    if not candidate_pool:
        return set()

    if use_llm_filter:
        filtered = _filter_topic_keywords_by_llm(segments, candidate_pool)
    else:
        filtered = candidate_pool

    # 빈도 순 정렬 후 상위를 더 많이 넣어서 대표 선정 후에도 20개가 나오도록 함
    sorted_by_freq = sorted(filtered, key=lambda w: counter[w], reverse=True)
    top_expanded = sorted_by_freq[: max(max_keywords * 2, 40)]
    representative = _select_representative_keywords(set(top_expanded))
    # 대표 집합을 빈도 순으로 정렬해 상위 max_keywords개 반환
    repr_sorted = sorted(representative, key=lambda w: counter[w], reverse=True)[:max_keywords]
    return set(repr_sorted)


def get_topic_keywords_list(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
) -> list[str]:
    """
    주제 키워드 반복 감지에 사용되는 키워드 목록을 반환 (확인용).
    _get_topic_keywords와 동일한 조건으로 추출한 뒤 정렬된 리스트로 반환.
    """
    kw_set = _get_topic_keywords(
        segments,
        min_freq=min_freq,
        max_keywords=max_keywords,
        max_segment_ratio=max_segment_ratio,
        min_length=min_keyword_len,
    )
    return sorted(kw_set)


def diagnose_topic_keywords(segments: list[dict], check_words: tuple[str, ...] = ("운영체제", "운영", "체제"), min_length: int = 2, max_segment_ratio: float = 1.0) -> dict:
    """
    주제 키워드가 왜 잡히지 않는지 진단.
    - Kiwi가 "운영체제"를 어떻게 분석하는지
    - check_words 각각이 몇 구간/몇 번 나오는지, topic_keywords에 포함되는지
    """
    total = len(segments)
    kiwi = _get_kiwi()
    lines = []
    lines.append("=== 주제 키워드 진단 ===")
    lines.append(f"전체 구간 수: {total}")
    lines.append(f"min_length={min_length}, max_segment_ratio={max_segment_ratio}")
    lines.append("")

    # 1) Kiwi로 "운영체제" 분석
    if kiwi is not None:
        try:
            sample = "운영체제의 목적과 기능"
            tokens = kiwi.tokenize(sample)
            lines.append(f"[Kiwi 분석 예시] '{sample}'")
            for t in tokens:
                form = getattr(t, "form", "?")
                tag = getattr(t, "tag", "?")
                lines.append(f"  form='{form}' tag={tag} len(form)={len(form)} NNG/NNP? {tag in _NOUN_TAGS} min_length 통과? {len(form) >= min_length} stop? {form in _MINIMAL_STOP}")
            lines.append("")
        except Exception as e:
            lines.append(f"Kiwi 예시 분석 실패: {e}")
            lines.append("")
    else:
        lines.append("Kiwi 미설치 (명사 추출 불가, 스템 방식 사용)")
        lines.append("")

    # 2) 구간별 추출 단어에서 check_words 등장 횟수
    counter: Counter = Counter()
    segment_presence: Counter = Counter()
    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_length)
        seen = set(words)
        for s in words:
            counter[s] += 1
        for s in seen:
            segment_presence[s] += 1

    threshold = total if max_segment_ratio >= 1.0 else max(2, int(total * max_segment_ratio))
    lines.append(f"threshold_segments (이 값 초과 시 제외): {threshold}")
    lines.append("")

    topic = _get_topic_keywords(
        segments,
        min_freq=5,
        max_keywords=20,
        max_segment_ratio=max_segment_ratio,
        min_length=min_length,
    )
    lines.append(f"topic_keywords 개수: {len(topic)}")
    lines.append("")

    for w in check_words:
        cnt = counter.get(w, 0)
        pres = segment_presence.get(w, 0)
        in_topic = w in topic
        lines.append(f"[{w}]")
        lines.append(f"  전체 등장 횟수: {cnt}, 등장한 구간 수: {pres}")
        lines.append(f"  threshold 초과? {pres > threshold} -> 제외됨: {pres > threshold}")
        lines.append(f"  min_freq(5) 미달? {cnt < 5}")
        lines.append(f"  topic_keywords 포함: {in_topic}")
        if not in_topic and cnt > 0:
            reason = []
            if cnt < 5:
                reason.append("min_freq 미달")
            if pres > threshold:
                reason.append("구간 비율 초과로 제외")
            if w in _MINIMAL_STOP:
                reason.append("stopword")
            if len(w) < min_length:
                reason.append(f"길이 {len(w)} < min_length {min_length}")
            lines.append(f"  -> 추정 원인: {', '.join(reason) or '확인 필요 (Kiwi 품사/길이 등)'}")
        lines.append("")

    # 3) topic_keywords 샘플 (처음 30개)
    sample_list = sorted(topic)[:30]
    lines.append("topic_keywords 샘플 (처음 30개): " + ", ".join(sample_list))
    return {"lines": lines, "topic_count": len(topic), "in_topic": {w: w in topic for w in check_words}}


def _filter_topic_keywords_by_llm(segments: list[dict], candidate_keywords: set[str]) -> set[str]:
    """
    LLM(Gemini)으로 강의 전체 흐름·주제에 맞는 키워드만 남기고, 동떨어진 단어는 제거.
    실패 시 원본 후보 그대로 반환.
    """
    if not candidate_keywords:
        return candidate_keywords
    try:
        from google.genai import types
        from config import gemini_client
        from utils import api_call_with_retry
    except ImportError:
        return candidate_keywords

    # 강의 맥락: 전사 앞부분 + 중간 샘플 (최대 약 4000자)
    parts = []
    n = len(segments)
    step = max(1, n // 8) if n > 8 else 1
    for i in range(0, min(n, 50), step):
        parts.append((segments[i].get("text") or "").strip())
    context = " ".join(parts).strip()[:4000]
    keyword_list = sorted(candidate_keywords)

    prompt = f"""당신은 강의 전사문을 보고, 그 강의의 **주제와 흐름에 맞는 키워드**만 골라주는 도우미입니다.

아래는 이 강의 전사의 앞·중간 일부입니다 (맥락 파악용):
---
{context}
---

아래는 전사문에서 빈도 기반으로 뽑은 **키워드 후보** 목록입니다. 이 중에서:
- 이 강의의 주제·내용·흐름과 **관련 있는** 키워드만 남기고,
- 주제와 **동떨어진** 단어(다른 분야 용어, 말실수/오타로 나온 단어, 강의와 무관한 단어)는 제외해주세요.

키워드 후보 (쉼표로 구분):
{", ".join(keyword_list)}

출력은 반드시 아래 JSON 형식만 사용하세요. 설명 없이 JSON만 출력하세요.
```json
{{ "keywords": ["키워드1", "키워드2", ...] }}
```
선택한 키워드만 배열에 넣으면 됩니다. 반드시 위 후보 목록에 있던 단어만 포함하세요."""

    def call_api():
        return gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

    try:
        response = api_call_with_retry(call_api)
        text = (response.text or "").strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        filtered = [str(k).strip() for k in data.get("keywords", [])]
        result = set(filtered) & candidate_keywords
        if result:
            print(f"    -> LLM 필터: {len(candidate_keywords)}개 후보 -> {len(result)}개 (주제 관련만 유지)")
            return result
    except Exception:
        pass
    return candidate_keywords


def detect_emphasis_by_topic_keyword_repetition(
    segments: list[dict],
    *,
    window: int = 2,
    min_keyword_len: int = 2,
    max_segment_ratio: float = 1.0,
    min_freq: int = 5,
    max_keywords: int = 20,
    use_llm_filter: bool = True,
    min_keyword_count: int = 1,
    _topic_keywords_override: set[str] | None = None,
) -> list[dict]:
    """
    전사문 **전체**에서 반복되는 주제 키워드가 등장하는 구간을 강조로 표시.

    - min_freq: 단어가 전사문 전체에서 나온 **총 횟수**가 이 값 이상이어야 주제 키워드 후보.
    - max_keywords: 채울 주제 키워드 최대 개수.
    - use_llm_filter: True면 Gemini로 강의 흐름·주제와 동떨어진 키워드는 제거 후 사용.
    - min_keyword_count: 한 구간에 주제 키워드가 이 개수 이상 매칭돼야 강조로 판정.
    - _topic_keywords_override: 이미 추출된 주제 키워드 집합을 넘기면 재추출/LLM 필터를 건너뜀.
    """
    use_noun_only = _get_kiwi() is not None
    label = f"min_kw>={min_keyword_count}"
    print(f"  [주제 키워드 반복 ({label})] 분석 중 (전사 전체 기준, {'명사만 사용' if use_noun_only else '스템 사용'})...")
    if not segments:
        return []

    if _topic_keywords_override is not None:
        topic_keywords = _topic_keywords_override
    else:
        topic_keywords = _get_topic_keywords(
            segments,
            min_freq=min_freq,
            max_keywords=max_keywords,
            max_segment_ratio=max_segment_ratio,
            min_length=min_keyword_len,
        )
        if use_llm_filter and topic_keywords:
            topic_keywords = _filter_topic_keywords_by_llm(segments, topic_keywords)
    if not topic_keywords:
        print(f"    -> 주제 키워드 없음 (반복 단어 부족)")
        return []

    emphasis_segments = []
    for seg in segments:
        words = _extract_content_words(seg.get("text") or "", min_length=min_keyword_len)
        here = set(words) & topic_keywords
        if len(here) < min_keyword_count:
            continue
        repeated_words = sorted(here)
        score = min(25 + len(repeated_words) * 5 + sum(len(w) for w in repeated_words[:5]), 55)
        emphasis_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "emphasis_score": float(score),
            "repeated_topic_keywords": repeated_words[:10],
            "detection_method": "topic_keyword_repeat",
        })

    print(f"    -> 주제 키워드 반복 ({label}): {len(emphasis_segments)}개 (전체의 {len(emphasis_segments)/len(segments)*100:.1f}%)")
    return emphasis_segments
