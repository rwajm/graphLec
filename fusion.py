"""
fusion.py — 멀티모달 강의 데이터 통합

입력:
  - by_slide_v2.json    : 오디오 전사 + 강조 감지 (슬라이드 단위)
  - slide_classified.json : 슬라이드 텍스트 + 시각 강조 + role 분류
  - annotation.json       : 강사 필기 annotation 이벤트

출력:
  - fused.json : 슬라이드 단위 통합 텍스트 + 강조 점수 + 지시어 매핑

사용법:
  python fusion.py
  python fusion.py --audio by_slide_v2.json --classified slide_classified.json
                   --annotation annotation.json --output fused.json
"""

import json
import re
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
#  설정
# ============================================================================

@dataclass
class Config:
    stem:            str  = "lecture"
    output_dir:      Path = Path("output")
    slides_dir:      Path = Path("output_slides")

    # 경로 — 직접 지정 시 우선, 없으면 __post_init__에서 config.output_paths()로 채움
    audio_path:      Path = field(default=None)
    classified_path: Path = field(default=None)
    annotation_path: Path = field(default=None)
    output_path:     Path = field(default=None)

    def __post_init__(self):
        try:
            from config import output_paths
            paths = output_paths(self.stem, self.output_dir, self.slides_dir)
            if self.audio_path      is None: self.audio_path      = paths["by_slide"]
            if self.classified_path is None: self.classified_path = paths["classified"]
            if self.annotation_path is None: self.annotation_path = paths["annotation"]
            if self.output_path     is None: self.output_path     = paths["fused"]
        except ImportError:
            d, s = self.output_dir, self.stem
            if self.audio_path      is None: self.audio_path      = d / f"{s}_by_slide.json"
            if self.classified_path is None: self.classified_path = d / f"{s}_slide_classified.json"
            if self.annotation_path is None: self.annotation_path = d / f"{s}_annotation.json"
            if self.output_path     is None: self.output_path     = d / f"{s}_fused.json"

    # 강조 가중치
    W_AUDIO:    float = 0.8
    W_VISUAL:   float = 1.5
    W_ANNOT:    float = 1.0   # annotation 점수는 자체 배율이 크므로 1.0
    BOTH_BONUS: float = 1.3   # 오디오 + annotation 동시 감지 보너스

    # 시간 윈도우
    TIME_WINDOW_SEC:          float = 20.0  # 강조 합산용 annotation ↔ audio 매칭
    DEICTIC_WINDOW_BEFORE_SEC: float = 5.0  # 지시어 발화 기준 이전
    DEICTIC_WINDOW_AFTER_SEC:  float = 2.0  # 지시어 발화 기준 이후

    # 키워드 필터
    MIN_KEYWORD_LEN: int = 2
    MIN_SLIDE_TEXT_LINE_LEN: int = 4  # 슬라이드 본문 라인 최소 길이 (불릿·번호 제외)
    SLIDE_TEXT_SCORE: float = 0.3     # 슬라이드 본문 등장 1회당 점수
    STOPWORDS: frozenset = frozenset({
        # 조사·접속사
        "이", "그", "저", "은", "는", "가", "을", "를", "의", "에", "도",
        "와", "과", "하고", "이고", "이며", "그리고", "그래서", "하지만",
        "또한", "즉", "따라서", "그러나", "또는", "및",
        # 서술어
        "있습니다", "있어요", "합니다", "해요", "됩니다", "돼요",
        "입니다", "이에요", "이다", "한다", "된다",
        # 일반 명사·대명사
        "것", "거", "수", "때", "더", "많이", "같은", "이런", "그런",
        # 추상 메타 단어 (강의 구조어)
        "개념", "정의", "목표", "목적", "기능", "시작", "발전", "차이",
        "종류", "특징", "핵심", "단어", "강의", "내용", "설명", "이해",
        "개요", "소개", "정리", "비교", "분석", "예시", "문제",
        # 일반 동사·행위어
        "실행", "요청", "종료", "생각", "과정", "사용", "제공",
        "처리", "수행", "동작", "발생", "설치", "구현", "관련",
        # 영어 불용어
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "and", "or", "for", "with", "that", "this", "be", "by",
    })


# ============================================================================
#  가중치 테이블
# ============================================================================

# 슬라이드 디자인 강조 타입 → 점수
SLIDE_EMPHASIS_WEIGHTS: dict[str, float] = {
    "color":     3.0,
    "highlight": 3.0,
    "bold":      2.0,
    "underline": 2.0,
    "box":       1.0,
    "callout":   1.0,
    "italic":    0.5,
    "other":     0.5,
}

# annotation 타입 → 점수
ANNOT_TYPE_WEIGHTS: dict[str, float] = {
    "circle":           3.0,
    "underline":        3.0,
    "bracket":          3.0,
    "arrow":            2.0,
    "handwritten_text": 2.0,  # handwritten_content != null 인 경우만
    "cross":            1.0,
    "other":            1.0,
}

# annotation confidence → 배율
ANNOT_CONFIDENCE_MULT: dict[str, float] = {
    "high":   1.0,
    "medium": 0.7,
    "low":    0.4,
}

# 지시어 패턴
DEICTIC_PATTERNS = re.compile(
    r'(?<!\w)(이것|이거|이것들|이게|저것|저거|저게|여기|저기|이쪽|저쪽|이 부분|저 부분|이 내용|이 개념)(?!\w)'
)


# ============================================================================
#  점수 계산 함수
# ============================================================================

def score_slide_emphasis(slide_emphasis_items: list[dict]) -> float:
    """slide_classified의 slide_emphasis 배열 → 시각 강조 점수."""
    total = 0.0
    for item in slide_emphasis_items:
        raw_type = item.get("type") or "other"
        t = (raw_type[0] if isinstance(raw_type, list) else raw_type).lower()
        total += SLIDE_EMPHASIS_WEIGHTS.get(t, SLIDE_EMPHASIS_WEIGHTS["other"])
    return round(total, 3)


def score_annotation_emphasis(annotations: list[dict]) -> float:
    """annotation 배열 → annotation 강조 점수."""
    total = 0.0
    for ann in annotations:
        ann_type = (ann.get("type") or "other").lower()
        # handwritten_text는 내용이 인식된 경우만
        if ann_type == "handwritten_text" and not ann.get("handwritten_content"):
            continue
        type_w = ANNOT_TYPE_WEIGHTS.get(ann_type, ANNOT_TYPE_WEIGHTS["other"])
        conf   = (ann.get("confidence") or "low").lower()
        conf_m = ANNOT_CONFIDENCE_MULT.get(conf, ANNOT_CONFIDENCE_MULT["low"])
        total += type_w * conf_m
    return round(total, 3)


def score_audio_emphasis(contexts: list[dict]) -> float:
    """by_slide_v2 contexts 배열 → 오디오 강조 점수 (0~1 정규화)."""
    max_audio = 60.0  # 오디오 score 최대값 기준
    scores = []
    for ctx in contexts:
        detail = ctx.get("emphasis", {}).get("detail") or {}
        audio  = detail.get("audio")
        if audio and audio.get("score"):
            scores.append(audio["score"])
    if not scores:
        return 0.0
    # 슬라이드 내 context 중 최대값을 대표 점수로 사용, 0~1 정규화
    return round(min(max(scores), max_audio) / max_audio, 3)


# ============================================================================
#  키워드 추출 및 소스 태깅
# ============================================================================

def normalize_keyword(word: str) -> str:
    word = word.lower().strip()
    word = re.sub(r'[^\w가-힣a-z0-9]', '', word)
    word = re.sub(
        r'(은|는|이|가|을|를|의|에서|에게|에|로|으로|와|과|도|만|까지|부터|처럼|이란|란)$',
        '', word
    )
    return word.strip()


def extract_keywords_from_text(text: str, stopwords: frozenset, min_len: int) -> set[str]:
    tokens = re.split(r'[\s,，.·/\-–—()（）\[\]]+', text)
    result = set()
    for t in tokens:
        kw = normalize_keyword(t)
        if len(kw) >= min_len and kw not in stopwords:
            result.add(kw)
    return result


def build_emphasized_keywords(
    audio_keywords: list[str],        # by_slide_v2 emphasis.keywords.all_keywords
    visual_keywords: list[str],        # slide_classified slide_emphasis[].text
    annotation_keywords: list[str],    # annotation target_content 토큰
    slide_text_keywords: list[str],    # slide_text 본문 라인 스캔 (4번째 소스)
    cfg: Config,
) -> list[dict]:
    """
    네 소스의 키워드를 통합, 소스 태깅.
      - audio / visual / annotation: 강조 신호 → sources 배열에 포함
      - slide_text: 본문 빈도 신호 → sources에 미포함, slide_text_score에만 누적
    같은 키워드가 여러 강조 소스에서 나오면 sources 배열에 모두 포함.
    """
    kw_map: dict[str, dict] = {}

    def add(kw: str, source: str, score_key: str, score_val: float):
        kw = normalize_keyword(kw)
        if not kw or len(kw) < cfg.MIN_KEYWORD_LEN or kw in cfg.STOPWORDS:
            return
        if kw not in kw_map:
            kw_map[kw] = {"keyword": kw, "sources": [], "audio_score": 0.0,
                          "visual_score": 0.0, "annotation_score": 0.0,
                          "slide_text_score": 0.0}
        entry = kw_map[kw]
        # slide_text는 sources에 포함하지 않음 (강조 신호가 아닌 본문 신호)
        if source != "slide_text" and source not in entry["sources"]:
            entry["sources"].append(source)
        entry[score_key] = round(entry[score_key] + score_val, 3)

    for kw in audio_keywords:
        add(kw, "audio", "audio_score", 1.0)

    for kw in visual_keywords:
        add(kw, "visual", "visual_score", 1.0)

    for kw in annotation_keywords:
        add(kw, "annotation", "annotation_score", 1.0)

    for kw in slide_text_keywords:
        add(kw, "slide_text", "slide_text_score", cfg.SLIDE_TEXT_SCORE)

    # sources 리스트 정렬 (재현성)
    result = []
    for entry in kw_map.values():
        entry["sources"] = sorted(entry["sources"])
        result.append(entry)

    # 소스 수 내림차순, 알파벳 오름차순 정렬
    result.sort(key=lambda x: (-len(x["sources"]), x["keyword"]))
    return result


# ============================================================================
#  annotation 인덱싱
# ============================================================================

def build_annotation_index(annotation_data: list[dict]) -> dict[int, list[dict]]:
    """
    slide_index → annotation 이벤트 리스트 인덱스.
    annotation.json은 annotation 이벤트(annot_index별) 배열이므로
    같은 slide_index에 여러 이벤트가 있을 수 있음.
    """
    index: dict[int, list[dict]] = {}
    for event in annotation_data:
        sid = event["slide_index"]
        if sid not in index:
            index[sid] = []
        index[sid].append(event)
    return index


def flatten_annotations_for_slide(events: list[dict]) -> list[dict]:
    """
    슬라이드의 annotation 이벤트 리스트 → 개별 annotation 목록 (timestamp 포함).
    """
    result = []
    for event in events:
        ts = event.get("timestamp_sec")
        for ann in event.get("annotations", []):
            result.append({**ann, "timestamp_sec": ts})
    return result


# ============================================================================
#  지시어 매칭
# ============================================================================

def find_deictic_target(
    seg_start: float,
    seg_text: str,
    slide_annotations: list[dict],  # flatten_annotations_for_slide 결과
    cfg: Config,
) -> Optional[dict]:
    """
    segment 텍스트에 지시어가 있으면, 발화 시점 기준 윈도우 내 annotation 중
    가장 가까운 것을 지시 대상으로 반환.
    """
    if not DEICTIC_PATTERNS.search(seg_text):
        return None
    if not slide_annotations:
        return None

    t_low  = seg_start - cfg.DEICTIC_WINDOW_BEFORE_SEC
    t_high = seg_start + cfg.DEICTIC_WINDOW_AFTER_SEC

    candidates = [
        ann for ann in slide_annotations
        if ann.get("timestamp_sec") is not None
        and t_low <= ann["timestamp_sec"] <= t_high
    ]
    if not candidates:
        return None

    # 시간적으로 가장 가까운 annotation 선택
    closest = min(candidates, key=lambda a: abs(a["timestamp_sec"] - seg_start))

    return {
        "target_content":   closest.get("target_content"),
        "annotation_type":  closest.get("type"),
        "bbox":             closest.get("target_bbox") or closest.get("annotation_bbox"),
        "timestamp_sec":    closest.get("timestamp_sec"),
        "confidence":       closest.get("confidence"),
    }


# ============================================================================
#  both_bonus 계산
# ============================================================================

def calc_both_bonus(
    audio_score: float,
    annot_score: float,
    annotation_timestamps: list[float],
    audio_contexts: list[dict],
    cfg: Config,
) -> float:
    """
    오디오 강조 구간과 annotation 타임스탬프가 TIME_WINDOW_SEC 내에 겹치면 BOTH_BONUS 부여.
    """
    if audio_score == 0.0 or annot_score == 0.0:
        return 0.0

    for ctx in audio_contexts:
        detail = ctx.get("emphasis", {}).get("detail") or {}
        if not (detail.get("audio") and detail["audio"].get("score", 0) > 0):
            continue
        ctx_start = ctx["start"]
        for ts in annotation_timestamps:
            if abs(ctx_start - ts) <= cfg.TIME_WINDOW_SEC:
                return cfg.BOTH_BONUS

    return 0.0


# ============================================================================
#  슬라이드 ID 변환
# ============================================================================

def slide_index_to_id(idx: int) -> str:
    """2 → 'slide_002'"""
    return f"slide_{idx:03d}"


def slide_id_to_index(sid: str) -> int:
    """'slide_002' → 2"""
    return int(sid.split("_")[-1])


# ============================================================================
#  메인 퓨전 로직
# ============================================================================

def run_fusion(cfg: Config) -> dict:
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    with open(cfg.audio_path,      encoding="utf-8") as f:
        audio_data = json.load(f)
    with open(cfg.classified_path, encoding="utf-8") as f:
        classified_data = json.load(f)
    with open(cfg.annotation_path, encoding="utf-8") as f:
        annotation_data = json.load(f)

    audio_slides      = audio_data["slides"]
    classified_slides = classified_data["slides"]
    annot_index       = build_annotation_index(annotation_data)

    # audio를 slide_index 기준으로 인덱싱
    audio_index: dict[int, dict] = {s["slide_index"]: s for s in audio_slides}

    # classified를 slide_id 기준으로 인덱싱
    classified_index: dict[str, dict] = {s["slide_id"]: s for s in classified_slides}

    print(f"로드 완료: audio {len(audio_slides)}개, "
          f"classified {len(classified_slides)}개, "
          f"annotation events {len(annotation_data)}개")

    fused_slides = []

    for cl_slide in classified_slides:
        slide_id  = cl_slide["slide_id"]
        slide_num = cl_slide["slide_number"]

        # audio 슬라이드 매핑 (slide_number로 매칭)
        au_slide = audio_index.get(slide_num)

        # annotation 이벤트 (이 슬라이드에 해당하는 것)
        annot_events   = annot_index.get(slide_num, [])
        flat_annots    = flatten_annotations_for_slide(annot_events)
        annot_ts_list  = [a["timestamp_sec"] for a in flat_annots if a.get("timestamp_sec")]

        # ── 강조 점수 계산 ───────────────────────────────────────────────────
        visual_score = score_slide_emphasis(cl_slide.get("slide_emphasis", []))
        annot_score  = score_annotation_emphasis(flat_annots)
        audio_score  = score_audio_emphasis(au_slide["contexts"] if au_slide else [])

        both_bonus = calc_both_bonus(
            audio_score, annot_score, annot_ts_list,
            au_slide["contexts"] if au_slide else [],
            cfg,
        )

        total_score = round(
            cfg.W_VISUAL * visual_score
            + cfg.W_ANNOT  * annot_score
            + cfg.W_AUDIO  * audio_score
            + both_bonus,
            3,
        )

        # ── 강조 키워드 수집 ─────────────────────────────────────────────────
        audio_kws: list[str] = []
        if au_slide:
            for ctx in au_slide["contexts"]:
                detail = ctx.get("emphasis", {}).get("detail") or {}
                kws    = detail.get("keywords", {}).get("all_keywords", [])
                audio_kws.extend(kws)

        visual_kws: list[str] = []
        for item in cl_slide.get("slide_emphasis", []):
            txt = item.get("text", "")
            if txt:
                visual_kws.extend(extract_keywords_from_text(txt, cfg.STOPWORDS, cfg.MIN_KEYWORD_LEN))

        annot_kws: list[str] = []
        for ann in flat_annots:
            tc = ann.get("target_content", "")
            if tc:
                annot_kws.extend(extract_keywords_from_text(tc, cfg.STOPWORDS, cfg.MIN_KEYWORD_LEN))

        # 4번째 소스: 슬라이드 본문 라인 단위 스캔
        # 강조 신호 없이도 슬라이드에 반복 등장하는 핵심 개념 포착
        # 두 가지 방식 병행:
        #   (a) 단어 분리: "독점", "자원" 등 단일 개념어
        #   (b) 라인 전체: "파일 시스템", "메모리 관리" 등 공백 포함 복합어 보존
        slide_text_kws: list[str] = []
        for line in cl_slide.get("t1", "").splitlines():
            line = line.strip()
            if len(line) < cfg.MIN_SLIDE_TEXT_LINE_LEN:
                continue  # 불릿(•, □), 번호(1.), 짧은 기호 제외

            # (a) 단어 분리 토큰
            slide_text_kws.extend(
                extract_keywords_from_text(line, cfg.STOPWORDS, cfg.MIN_KEYWORD_LEN)
            )

            # (b) 복합어 후보: 공백 포함 라인이 짧으면(2~4어절) 라인 자체도 후보로 추가
            #     "파일 시스템 관리(file system management)" 같은 라인을
            #     괄호·영문·불릿 제거 후 2~4어절 복합어로 포착
            #     주의: normalize_keyword는 공백을 제거하므로 복합어에는 사용 금지
            line_clean = re.sub(r'\(.*?\)', '', line).strip()           # 괄호 내용 제거
            line_clean = re.sub(r'[a-zA-Z0-9/]', '', line_clean).strip() # 영문·숫자·슬래시 제거
            line_clean = re.sub(r'^[\s\-·•□▪◦]+', '', line_clean)      # 앞 불릿·대시 제거
            line_clean = re.sub(r'\s+', ' ', line_clean).strip()
            words = [w for w in line_clean.split() if len(w) >= 2]       # 1글자 제거
            compound = ' '.join(words)
            if 2 <= len(words) <= 4 and compound not in cfg.STOPWORDS:
                if len(compound) >= cfg.MIN_KEYWORD_LEN:
                    slide_text_kws.append(compound)

        emphasized_keywords = build_emphasized_keywords(
            audio_kws, visual_kws, annot_kws, slide_text_kws, cfg
        )

        # ── contexts + segments (지시어 매칭 포함) ──────────────────────────
        fused_contexts = []
        if au_slide:
            for ctx in au_slide["contexts"]:
                detail   = ctx.get("emphasis", {}).get("detail") or {}
                stressed = ctx["emphasis"].get("detected", False)

                fused_segs = []
                for seg in ctx.get("segments", []):
                    deictic_target = find_deictic_target(
                        seg["start"], seg["text"], flat_annots, cfg
                    )
                    fused_segs.append({
                        "start":          seg["start"],
                        "end":            seg["end"],
                        "text":           seg["text"],
                        "stressed":       stressed,  # context 단위 플래그를 segment에 상속
                        "deictic_target": deictic_target,
                    })

                fused_contexts.append({
                    "context_index": ctx["context_index"],
                    "start":         ctx["start"],
                    "end":           ctx["end"],
                    "text":          ctx["text"],
                    "stressed":      stressed,
                    "segments":      fused_segs,
                })

        # ── annotation 요약 ──────────────────────────────────────────────────
        annotations_summary = []
        for ann in flat_annots:
            ann_type = (ann.get("type") or "other").lower()
            if ann_type == "handwritten_text" and not ann.get("handwritten_content"):
                continue
            type_w = ANNOT_TYPE_WEIGHTS.get(ann_type, ANNOT_TYPE_WEIGHTS["other"])
            conf   = (ann.get("confidence") or "low").lower()
            conf_m = ANNOT_CONFIDENCE_MULT.get(conf, ANNOT_CONFIDENCE_MULT["low"])
            annotations_summary.append({
                "type":            ann.get("type"),
                "target_content":  ann.get("target_content"),
                "handwritten_content": ann.get("handwritten_content"),
                "score":           round(type_w * conf_m, 3),
                "confidence":      ann.get("confidence"),
                "timestamp_sec":   ann.get("timestamp_sec"),
                "bbox":            ann.get("target_bbox") or ann.get("annotation_bbox"),
            })

        # ── 슬라이드 통합 ────────────────────────────────────────────────────
        fused_slide = {
            "slide_id":     slide_id,
            "slide_number": slide_num,
            "title":        cl_slide.get("title", ""),
            "slide_text":   cl_slide.get("t1", ""),
            "role":         cl_slide.get("role"),
            "start_sec":    au_slide["start_sec"] if au_slide else None,
            "end_sec":      au_slide["end_sec"]   if au_slide else None,

            "emphasis_score": {
                "audio":      round(cfg.W_AUDIO  * audio_score,  3),
                "visual":     round(cfg.W_VISUAL * visual_score, 3),
                "annotation": round(cfg.W_ANNOT  * annot_score,  3),
                "both_bonus": both_bonus,
                "total":      total_score,
            },

            "emphasized_keywords": emphasized_keywords,
            "contexts":            fused_contexts,
            "annotations_summary": annotations_summary,
        }

        fused_slides.append(fused_slide)
        print(f"  {slide_id} | role={cl_slide.get('role'):12s} | "
              f"total={total_score:.2f} "
              f"(vis={visual_score:.1f} ann={annot_score:.1f} aud={audio_score:.2f} bonus={both_bonus:.1f})")

    # ── 출력 ─────────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "total_slides": len(fused_slides),
            "source_files": {
                "audio":      str(cfg.audio_path),
                "classified": str(cfg.classified_path),
                "annotation": str(cfg.annotation_path),
            },
            "fusion_weights": {
                "W_AUDIO":                    cfg.W_AUDIO,
                "W_VISUAL":                   cfg.W_VISUAL,
                "W_ANNOT":                    cfg.W_ANNOT,
                "BOTH_BONUS":                 cfg.BOTH_BONUS,
                "TIME_WINDOW_SEC":            cfg.TIME_WINDOW_SEC,
                "DEICTIC_WINDOW_BEFORE_SEC":  cfg.DEICTIC_WINDOW_BEFORE_SEC,
                "DEICTIC_WINDOW_AFTER_SEC":   cfg.DEICTIC_WINDOW_AFTER_SEC,
            },
        },
        "slides": fused_slides,
    }

    return output


# ============================================================================
#  엔트리포인트
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="멀티모달 강의 데이터 퓨전",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stem", required=True,
                        help="강의 파일 stem (예: os1-1). 경로는 config.output_paths()로 자동 완성됨")
    parser.add_argument("--output_dir", default="output",      help="출력 디렉토리 (기본: output)")
    parser.add_argument("--slides_dir", default="output_slides",help="슬라이드 디렉토리 (기본: output_slides)")
    args = parser.parse_args()

    cfg = Config(
        stem       = args.stem,
        output_dir = Path(args.output_dir),
        slides_dir = Path(args.slides_dir),
    )

    for p, label in [
        (cfg.audio_path,      "audio"),
        (cfg.classified_path, "classified"),
        (cfg.annotation_path, "annotation"),
    ]:
        if not p.exists():
            print(f"❌ {label} 파일 없음: {p}")
            return

    print("\n" + "="*60)
    print("🔀 멀티모달 퓨전 시작")
    print("="*60)

    start = time.time()
    output = run_fusion(cfg)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    total_kw = sum(len(s["emphasized_keywords"]) for s in output["slides"])
    total_deictic = sum(
        1
        for s in output["slides"]
        for ctx in s["contexts"]
        for seg in ctx["segments"]
        if seg.get("deictic_target")
    )

    print("\n" + "="*60)
    print("✅ 완료")
    print("="*60)
    print(f"  슬라이드       : {output['metadata']['total_slides']}개")
    print(f"  강조 키워드    : {total_kw}개")
    print(f"  지시어 매칭    : {total_deictic}개")
    print(f"  처리 시간      : {elapsed:.2f}초")
    print(f"  출력 파일      : {cfg.output_path}")


if __name__ == "__main__":
    main()