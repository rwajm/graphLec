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
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_1"))
MODEL = "gemini-2.5-flash"


# ============================================================================
#  설정
# ============================================================================

@dataclass
class RecommenderConfig:
    # 유사도 가중치 (합계 = 1.0)
    W_DOMAIN:  float = 0.20
    W_KEYWORD: float = 0.35
    W_CONCEPT: float = 0.30
    W_PREREQ:  float = 0.15

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
    target: LectureMetadata,
) -> dict:
    """
    질의 키워드 기반 유사도.
      - kw_score   : query_keywords vs top_keywords + top_concepts (Jaccard)
      - title_score: 질의 키워드가 강의 제목에 포함된 비율
                     제목은 강의가 실제로 다루는 핵심 주제를 직접 나타내므로 높은 가중치 부여
    """
    TITLE_WEIGHT = 2.0

    target_all = list(set(target.top_keywords + target.top_concepts))
    kw_score = jaccard(query_keywords, target_all)

    title_words = set(target.title.replace(",", " ").replace("·", " ").split())
    # 방향 고정: qk가 tw에 포함될 때만 매칭
    # qk="자료구조", tw="구조" → "자료구조" in "구조" → False (오매칭 방지)
    # qk="이미지 분류", tw="분류" → "이미지 분류" in "분류" → False
    # qk="트리",  tw="트리와"   → "트리" in "트리와" → True ✓
    matched = sum(1 for qk in query_keywords if qk in title_words)
    title_score = matched / max(len(query_keywords), 1)

    total = kw_score + TITLE_WEIGHT * title_score

    return {
        "score":       round(total, 4),
        "kw_score":    round(kw_score, 4),
        "title_score": round(title_score, 4),
    }


# ============================================================================
#  Gemini — 자연어 질의 키워드 추출
# ============================================================================

def extract_keywords_from_query(query: str) -> list[str]:
    prompt = f"""다음 질의에서 강의 검색에 사용할 핵심 키워드를 추출해줘.

질의: "{query}"

조건:
- 질의자가 실제로 배우고 싶은 **대상 개념·기술어**만 추출
- 수단·방법·맥락 단어 제외 (예: "공부", "방법", "알고 싶어", "추천", "배우기 전에")
- "머신러닝 공부하기 전에 수학 뭐 알아야 해?" → ["선형대수", "통계", "미적분"] (수학 선수지식, 머신러닝X)
- "딥러닝으로 이미지 분류하는 방법" → ["CNN", "이미지 분류", "딥러닝"] (분류 알고리즘X)
- "SQL 쿼리 최적화" → ["SQL", "인덱스", "쿼리 최적화"] (최적화 단독X)
- 2~6개 추출
- JSON 배열만 출력: ["키워드1", "키워드2", ...]
- 설명 없이 JSON만 출력"""

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"temperature": 0.0},
    )
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


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
        raw_keywords = extract_keywords_from_query(query)

        # 메타데이터 어휘 집합으로 필터링
        # — Gemini가 '프로세스 분석', '프로세스 개선' 같이 메타데이터에 없는
        #   키워드를 추출하면 Jaccard 분모만 늘어 점수가 불안정해짐
        # — 어휘 집합에 있는 키워드 + 어휘와 부분 일치하는 키워드만 유지
        vocab = self.collection.vocabulary()

        def _is_valid(kw: str) -> bool:
            # 1) 정확 일치
            if kw in vocab:
                return True
            # 2) 키워드의 첫 번째 토큰(핵심어)이 vocab에 정확히 존재하는지 확인
            #    예: "프로세스 관리" → "프로세스"가 vocab에 있으면 유효
            #        "프로세스 분석" → "프로세스"가 vocab에 있어도 "분석"은 없으므로
            #                         복합어 전체가 vocab에 없으면 제외
            tokens = kw.split()
            # 모든 토큰이 vocab 단어를 포함하거나 포함되는 경우만 허용
            return all(
                any(tok in v or v in tok for v in vocab if len(v) >= 2)
                for tok in tokens
            )

        query_keywords = [kw for kw in raw_keywords if _is_valid(kw)]
        if not query_keywords:
            query_keywords = raw_keywords  # 필터 결과가 빈 경우 원본 사용

        print(f"[추출 키워드] {raw_keywords}")
        if query_keywords != raw_keywords:
            filtered_out = [k for k in raw_keywords if k not in query_keywords]
            print(f"[필터 제거]   {filtered_out} (메타데이터 어휘 없음)")
        print(f"[유효 키워드] {query_keywords}\n")

        candidates = []
        for target in self.collection.all():
            detail = compute_query_similarity(query_keywords, target)
            candidates.append((target, detail))

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        # ── 도메인 연관 보너스 ──────────────────────────────────────────────
        # 1위 강의 도메인을 기준으로 같은 도메인 0점 강의에 +0.15 부여
        # (상위 3개 기준은 무관 도메인까지 포함될 수 있어 1위만 사용)
        DOMAIN_BONUS = 0.15
        top_domain: str = ""
        if candidates and candidates[0][1]["score"] > 0:
            top_domain = candidates[0][0].domain

        if top_domain:
            for t, d in candidates:
                if t.domain == top_domain and d["score"] == 0:
                    d["score"]        = round(d["score"] + DOMAIN_BONUS, 4)
                    d["domain_bonus"] = DOMAIN_BONUS

        candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        # 전체 결과 반환 — print_results에서 top_k/score>0 필터링
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
                    title_str  = f"  title={d['title_score']:.2f}" if d.get("title_score") else ""
                    domain_str = f"  domain_bonus={d['domain_bonus']:.2f}" if d.get("domain_bonus") else ""
                    print(f"     신호:   kw={d['kw_score']:.2f}{title_str}{domain_str}")
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