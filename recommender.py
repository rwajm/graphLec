"""
recommender.py
──────────────
강의 추천 시스템

두 가지 모드:
  1. 이력 기반 추천  — 시청 이력에서 다음 강의 추천
  2. 자연어 질의 추천 — 질의에서 키워드 추출 → 메타데이터 유사도 계산

CLI 실행:
  # 자연어 질의
  python recommender.py --query "메모리 관리 방법 알고 싶어"

  # 이력 기반 (시청 강의 순서대로 입력)
  python recommender.py --history os1-1 os1-2

  # 둘 다
  python recommender.py --history os1-1 --query "스케줄링 알고 싶어"

  # 메타데이터 디렉토리 지정
  python recommender.py --metadata_dir metadata/ --query "운영체제란 무엇인가"

모듈로 사용:
  from recommender import Recommender
  rec = Recommender("metadata/")
  rec.record_view("os1-1")
  results = rec.recommend_from_history(top_k=5)
  results = rec.recommend_from_query("프로세스 스케줄링 알고 싶어")
"""

import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_2"))
MODEL = "gemini-2.5-flash"


# ============================================================================
#  설정
# ============================================================================

@dataclass
class RecommenderConfig:
    # 이력 기반 추천 가중치 (합계 = 1.0)
    W_DOMAIN:  float = 0.20
    W_KEYWORD: float = 0.35
    W_CONCEPT: float = 0.30
    W_PREREQ:  float = 0.15

    # 질의 기반 추천 가중치 (합계 = 1.0)
    WQ_DOMAIN: float = 0.30   # 질의 추론 도메인 일치
    WQ_TITLE:  float = 0.40   # 강의 제목 키워드 일치
    WQ_KW:     float = 0.30   # top_keywords + top_concepts Jaccard

    # 이력 감쇠 (최근 강의일수록 weight 높음)
    HISTORY_KEYWORD_DECAY: float = 0.9
    MAX_HISTORY:           int   = 20


# ============================================================================
#  데이터 모델
# ============================================================================

@dataclass
class LectureMetadata:
    video_id:          str
    title:             str
    instructor:        str
    domain:            str
    top_keywords:      list[str]
    top_concepts:      list[str]
    prerequisites:     list[str]
    duration_sec:      int
    summary:           str
    knowledge_density: float
    difficulty_level:  str


@dataclass
class ViewRecord:
    video_id: str
    keywords: list[str]
    domain:   str


@dataclass
class LearningHistory:
    records: list[ViewRecord] = field(default_factory=list)

    def add(self, record: ViewRecord, max_history: int = 20):
        self.records.append(record)
        if len(self.records) > max_history:
            self.records = self.records[-max_history:]

    @property
    def watched_ids(self) -> set[str]:
        return {r.video_id for r in self.records}

    def accumulated_keywords(self, decay: float = 0.9) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for i, record in enumerate(reversed(self.records)):
            for kw in record.keywords:
                scores[kw] += decay ** i
        return dict(scores)


# ============================================================================
#  메타데이터 컬렉션
# ============================================================================

class MetadataCollection:
    def __init__(self, metadata_dir: str):
        self.lectures: dict[str, LectureMetadata] = {}
        self._load(Path(metadata_dir))

    def _load(self, directory: Path):
        files = list(directory.glob("*_metadata.json"))
        if not files:
            raise FileNotFoundError(f"메타데이터 파일 없음: {directory}")
        for path in files:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            concept_names = [
                c["name"] if isinstance(c, dict) else c
                for c in raw.get("top_concepts", [])
            ]
            lec = LectureMetadata(
                video_id          = raw["video_id"],
                title             = raw["title"],
                instructor        = raw.get("instructor", ""),
                domain            = raw.get("domain", "unknown"),
                top_keywords      = raw.get("top_keywords", []),
                top_concepts      = concept_names,
                prerequisites     = raw.get("prerequisites", []),
                duration_sec      = raw.get("duration_sec", 0),
                summary           = raw.get("summary", ""),
                knowledge_density = raw.get("knowledge_density", 0.0),
                difficulty_level  = raw.get("difficulty_level", "unclassified"),
            )
            self.lectures[lec.video_id] = lec
        print(f"[로드] {len(self.lectures)}개 강의 메타데이터 로드 완료\n")

    def get(self, video_id: str) -> Optional[LectureMetadata]:
        return self.lectures.get(video_id)

    def all(self) -> list[LectureMetadata]:
        return list(self.lectures.values())

    def vocabulary(self) -> set[str]:
        """전체 강의 메타데이터에 등장하는 키워드/개념 어휘 집합"""
        vocab: set[str] = set()
        for lec in self.lectures.values():
            vocab.update(lec.top_keywords)
            vocab.update(lec.top_concepts)
            vocab.update(lec.title.split())
        return vocab


# ============================================================================
#  유사도 계산
# ============================================================================

def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def prereq_chain_score(
    source: LectureMetadata,
    target: LectureMetadata,
    history_keywords: set[str],
) -> float:
    """target의 선수 지식이 source + 이력으로 충족되는 비율"""
    if not target.prerequisites:
        return 0.0  # 무관 강의 추천 방지 (이전 0.5는 hist1-1 등이 진입하는 문제 유발)
    known = set(source.top_concepts) | set(source.top_keywords) | history_keywords
    satisfied = sum(1 for p in target.prerequisites if p in known)
    return satisfied / len(target.prerequisites)


def compute_similarity(
    source: LectureMetadata,
    target: LectureMetadata,
    history_keywords: set[str],
    cfg: RecommenderConfig,
) -> dict:
    domain_score  = 1.0 if source.domain == target.domain else 0.0
    keyword_score = jaccard(source.top_keywords, target.top_keywords)
    concept_score = jaccard(source.top_concepts, target.top_concepts)
    prereq_score  = prereq_chain_score(source, target, history_keywords)

    total = (
        cfg.W_DOMAIN  * domain_score
        + cfg.W_KEYWORD * keyword_score
        + cfg.W_CONCEPT * concept_score
        + cfg.W_PREREQ  * prereq_score
    )
    return {
        "score":         round(total, 4),
        "domain_score":  round(domain_score, 4),
        "keyword_score": round(keyword_score, 4),
        "concept_score": round(concept_score, 4),
        "prereq_score":  round(prereq_score, 4),
    }


def compute_query_similarity(
    query_keywords: list[str],
    query_domain:   Optional[str],        # Gemini 추론 도메인 (없으면 None)
    target: LectureMetadata,
    cfg: RecommenderConfig,
) -> dict:
    """
    질의 기반 유사도 — 항상 0~1 범위.

    세 신호의 가중합:
      domain_score : 추론 도메인 == 강의 도메인       (0 or 1)
      title_score  : 질의 키워드 ∩ 강의 제목 단어 비율 (0~1)
      kw_score     : Jaccard(질의키워드, top_kw+concepts) (0~1)

    domain이 None이면 domain_score=0, 나머지 두 가중치를 0.5:0.5로 재분배.
    """
    # ── domain ──────────────────────────────────────────────────────────────
    if query_domain and query_domain != "unknown":
        domain_score = 1.0 if target.domain == query_domain else 0.0
        w_domain, w_title, w_kw = cfg.WQ_DOMAIN, cfg.WQ_TITLE, cfg.WQ_KW
    else:
        # 도메인 미확정 → domain 신호 제거 후 나머지 재분배
        domain_score = 0.0
        w_domain, w_title, w_kw = 0.0, 0.50, 0.50

    # ── title ───────────────────────────────────────────────────────────────
    title_words = set(target.title.replace(",", " ").replace("·", " ").split())
    matched = sum(1 for qk in query_keywords if any(qk in tw for tw in title_words))
    title_score = matched / max(len(query_keywords), 1)

    # ── kw (Jaccard) ────────────────────────────────────────────────────────
    target_all = list(set(target.top_keywords + target.top_concepts))
    kw_score = jaccard(query_keywords, target_all)

    # ── 가중합 → 항상 0~1 ───────────────────────────────────────────────────
    total = (
        w_domain * domain_score
        + w_title  * title_score
        + w_kw     * kw_score
    )

    return {
        "score":        round(total, 4),
        "domain_score": round(domain_score, 4),
        "title_score":  round(title_score, 4),
        "kw_score":     round(kw_score, 4),
    }


# ============================================================================
#  Gemini — 자연어 질의 분석 (키워드 + 도메인 동시 추출)
# ============================================================================

DOMAIN_CANDIDATES = [
    "cs/operating_system", "cs/network", "cs/data_structure",
    "cs/algorithm", "cs/database", "cs/software_engineering",
    "math/linear_algebra", "math/statistics", "math/calculus",
    "ml/deep_learning", "ml/machine_learning", "other",
]

def analyze_query(query: str) -> tuple[list[str], Optional[str]]:
    """
    질의에서 키워드와 도메인을 Gemini 1회 호출로 동시 추출.

    반환:
      keywords : 핵심 개념·기술어 2~6개
      domain   : DOMAIN_CANDIDATES 중 하나, 해당 없으면 None
    """
    prompt = f"""다음 강의 검색 질의를 분석해줘.

질의: "{query}"

다음 JSON 형식으로만 출력해 (설명 없이):
{{
  "keywords": ["키워드1", "키워드2", ...],
  "domain": "도메인 문자열 또는 null"
}}

keywords 조건:
- 질의자가 배우고 싶은 핵심 개념·기술어만 (2~6개)
- "공부", "방법", "알고 싶어", "추천" 같은 메타 표현 제외
- 예) "프로세스 스케줄링 알고 싶어" → ["프로세스", "스케줄링"]
- 예) "머신러닝 전에 수학 뭐 알아야 해?" → ["선형대수", "통계", "미적분"]

domain 조건:
- 아래 중 가장 적합한 것 하나만 선택:
  {', '.join(DOMAIN_CANDIDATES[:-1])}
- 강의 도메인과 무관한 질의(요리, 여행 등)이면 null
- 확신하기 어려우면 null"""

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"temperature": 0.0},
    )
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)
    keywords = parsed.get("keywords", [])
    domain   = parsed.get("domain") or None   # "null" 문자열도 None 처리
    if domain not in DOMAIN_CANDIDATES:
        domain = None
    return keywords, domain


# ============================================================================
#  추천 결과
# ============================================================================

@dataclass
class RecommendResult:
    video_id:     str
    title:        str
    domain:       str
    instructor:   str
    score:        float
    score_detail: dict
    reason:       str
    summary:      str


def _build_reason(detail: dict, mode: str) -> str:
    if mode == "history":
        parts = []
        if detail.get("domain_score", 0) == 1.0:
            parts.append("같은 도메인")
        if detail.get("keyword_score", 0) > 0.1:
            parts.append(f"키워드 유사 {detail['keyword_score']:.0%}")
        if detail.get("concept_score", 0) > 0.1:
            parts.append(f"개념 유사 {detail['concept_score']:.0%}")
        if detail.get("prereq_score", 0) >= 0.8:
            parts.append("선수 지식 충족")
        return " · ".join(parts) if parts else "관련 강의"
    else:
        parts = []
        if detail.get("domain_score", 0) == 1.0:
            parts.append("도메인 일치")
        if detail.get("title_score", 0) > 0:
            parts.append(f"제목 일치 {detail['title_score']:.0%}")
        if detail.get("kw_score", 0) > 0:
            parts.append(f"키워드 일치 {detail['kw_score']:.0%}")
        return " · ".join(parts) if parts else "관련 강의"


# ============================================================================
#  추천 엔진
# ============================================================================

class Recommender:
    def __init__(self, metadata_dir: str = "metadata/", config: Optional[RecommenderConfig] = None):
        self.collection = MetadataCollection(metadata_dir)
        self.history    = LearningHistory()
        self.cfg        = config or RecommenderConfig()

    def record_view(self, video_id: str):
        lec = self.collection.get(video_id)
        if not lec:
            raise ValueError(f"알 수 없는 강의 ID: {video_id}")
        self.history.add(
            ViewRecord(video_id=video_id, keywords=lec.top_keywords, domain=lec.domain),
            max_history=self.cfg.MAX_HISTORY,
        )
        print(f"[이력 기록] {video_id} — {lec.title}")

    def recommend_from_history(self, top_k: int = 5) -> list[RecommendResult]:
        if not self.history.records:
            return self._recommend_cold_start(top_k)

        last = self.history.records[-1]
        source = self.collection.get(last.video_id)
        accumulated = set(self.history.accumulated_keywords(self.cfg.HISTORY_KEYWORD_DECAY))

        candidates = []
        for target in self.collection.all():
            if target.video_id in self.history.watched_ids:
                continue
            detail = compute_similarity(source, target, accumulated, self.cfg)
            candidates.append((target, detail))

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        return [
            RecommendResult(
                video_id     = t.video_id,
                title        = t.title,
                domain       = t.domain,
                instructor   = t.instructor,
                score        = d["score"],
                score_detail = d,
                reason       = _build_reason(d, "history"),
                summary      = t.summary,
            )
            for t, d in candidates[:top_k]
        ]

    def recommend_from_query(self, query: str, top_k: int = 5) -> list[RecommendResult]:
        print(f"[질의 분석] {query}")
        query_keywords, query_domain = analyze_query(query)
        print(f"[추출 키워드] {query_keywords}")
        print(f"[추론 도메인] {query_domain or '미확정'}\n")

        candidates = []
        for target in self.collection.all():
            detail = compute_query_similarity(query_keywords, query_domain, target, self.cfg)
            candidates.append((target, detail))

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        return [
            RecommendResult(
                video_id     = t.video_id,
                title        = t.title,
                domain       = t.domain,
                instructor   = t.instructor,
                score        = d["score"],
                score_detail = d,
                reason       = _build_reason(d, "query"),
                summary      = t.summary,
            )
            for t, d in candidates
        ]

    def _recommend_cold_start(self, top_k: int) -> list[RecommendResult]:
        """이력 없을 때: knowledge_density 낮은 순 (입문 강의 우선)"""
        all_lecs = sorted(self.collection.all(), key=lambda x: x.knowledge_density)
        return [
            RecommendResult(
                video_id=t.video_id, title=t.title, domain=t.domain,
                instructor=t.instructor, score=0.0, score_detail={},
                reason="입문 강의 추천", summary=t.summary,
            )
            for t in all_lecs[:top_k]
        ]

    @staticmethod
    def print_results(results: list[RecommendResult], title: str = "추천 결과", top_k: int = 3):
        # 점수 있는 것 / 없는 것 분리
        scored   = [r for r in results if r.score > 0]
        unscored = [r for r in results if r.score == 0]
        top      = scored[:top_k]

        print(f"\n{'='*62}")
        print(f"  {title}")
        print(f"{'='*62}")

        if not top:
            print("  ※ 질의와 일치하는 강의를 찾지 못했습니다.")
        else:
            print(f"  ▶ 추천 강의 (Top {min(top_k, len(top))})")
            print(f"  {'-'*58}")
            for i, r in enumerate(top, 1):
                print(f"  {i}. {r.video_id}: {r.title}  [{r.instructor}]")
                print(f"     도메인: {r.domain}")
                print(f"     점수:   {r.score:.4f}  |  {r.reason}")
                d = r.score_detail
                if "domain_score" in d:
                    print(f"     신호:   domain={d['domain_score']:.2f}  "
                          f"keyword={d['keyword_score']:.2f}  "
                          f"concept={d['concept_score']:.2f}  "
                          f"prereq={d['prereq_score']:.2f}")
                elif "kw_score" in d:
                    print(f"     신호:   domain={d.get('domain_score',0):.2f}  "
                          f"title={d.get('title_score',0):.2f}  "
                          f"kw={d.get('kw_score',0):.2f}")
                print(f"     요약:   {r.summary[:80]}...")
                print()

        # 전체 점수 표
        print(f"  {'─'*58}")
        print(f"  ▶ 전체 강의 점수 (추천 근거)")
        print(f"  {'─'*58}")
        print(f"  {'video_id':<12} {'제목':<22} {'도메인':<26} {'점수':>6}")
        print(f"  {'─'*58}")
        for r in results:
            marker = " ◀" if r in top else ""
            print(f"  {r.video_id:<12} {r.title[:20]:<22} {r.domain:<26} {r.score:>6.4f}{marker}")
        print()


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="강의 추천 시스템",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""예시:
  python recommender.py --query "메모리 관리 방법 알고 싶어"
  python recommender.py --history os1-1 os1-2
  python recommender.py --history os1-1 --query "스케줄링 알고 싶어"
  python recommender.py --metadata_dir metadata/ --query "운영체제란 무엇인가"
"""
    )
    parser.add_argument("--metadata_dir", default="metadata/",
                        help="메타데이터 디렉토리 (기본: metadata/)")
    parser.add_argument("--query", default=None,
                        help="자연어 질의 (예: '메모리 관리 알고 싶어')")
    parser.add_argument("--history", nargs="*", default=None,
                        help="시청 이력 강의 ID (공백으로 구분, 예: os1-1 os1-2)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="추천 결과 수 (기본: 5)")
    args = parser.parse_args()

    rec = Recommender(metadata_dir=args.metadata_dir)

    # 이력 등록
    if args.history:
        for vid in args.history:
            rec.record_view(vid)
        print()

    ran = False

    # 이력 기반 추천
    if args.history:
        results = rec.recommend_from_history(top_k=args.top_k)
        rec.print_results(results, "이력 기반 추천", top_k=3)
        ran = True

    # 자연어 질의 추천
    if args.query:
        results = rec.recommend_from_query(args.query, top_k=args.top_k)
        rec.print_results(results, f"질의 기반 추천: '{args.query}'", top_k=3)
        ran = True

    # 아무 옵션도 없으면 대화형 모드
    if not ran:
        print("옵션 없이 실행 시 대화형 모드로 진입합니다.")
        print("종료: Ctrl+C 또는 'q' 입력\n")
        while True:
            try:
                query = input("질의 입력 > ").strip()
                if query.lower() in ("q", "quit", "exit"):
                    break
                if not query:
                    continue
                results = rec.recommend_from_query(query, top_k=args.top_k)
                rec.print_results(results, f"질의: '{query}'", top_k=3)
            except KeyboardInterrupt:
                print("\n종료")
                break