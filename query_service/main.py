"""
GraphLEC 질의 전용 FastAPI 서비스.

Django는 /internal/query 로 stem + question 만 전달한다 (lecture_id → stem 변환은 Django 담당).

/internal/query: 질문 유형에 따라
  - 구조(structural): LLM이 Cypher 생성 → Neo4j 조회(읽기 전용 검증, $stem 필수)
  - 내용(content): 고정 Cypher 템플릿 + 키워드(파라미터 바인딩, stem 필터)
  이후 Gemini 답변. 그래프 근거가 없으면 답변 거부.
  Lance는 linked_node_id가 조회된 노드 id와 일치할 때만 보강(최대 4개).

/internal/query_graph_evidence: 키워드 기반 엣지 근거(디버그·경량 조회용, 기존 유지).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from lance_ingest import default_lance_root, lance_search  # noqa: E402

app = FastAPI(title="GraphLEC Query Service", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_2") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-2.0-flash")
TOP_K = int(os.getenv("GRAPHLEC_TOP_K", "8"))
GRAPH_TOP_K = int(os.getenv("GRAPHLEC_GRAPH_TOP_K", "12"))
RETRY_DELAYS = [0, 5, 15, 30]
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

GRAPH_SCHEMA = """
노드 타입과 주요 프로퍼티:
- Video          : id, stem, title
- Slides         : id, stem
- Scenes         : id, stem
- Slide          : id, stem, slide_number, title, slide_text, role, start_sec, end_sec, emphasis_total
- Scene          : id, stem, slide_id, context_index, start, end, stressed
- Segment        : id, stem, start, end, text, stressed
- AnnotationEmphasis : id, stem, type, target_content, score, confidence, timestamp_sec
- Concept        : id, stem, name

관계 (방향 중요):
- (Video)-[:HAS_SLIDES]->(Slides)
- (Video)-[:HAS_SCENES]->(Scenes)
- (Slides)-[:CONTAINS]->(Slide)
- (Scenes)-[:CONTAINS]->(Scene)
- (Slide)-[:HAS_SCENE]->(Scene)
- (Scene)-[:HAS_SEGMENT]->(Segment)
- (Slide)-[:HAS_ANNOTATION]->(AnnotationEmphasis)
- (Segment)-[:REFERS_TO]->(AnnotationEmphasis)
- (Segment)-[:MENTIONS]->(Concept)
- (Slide)-[:APPEARS_IN]->(Concept)
- (Concept)-[:is_a|part_of|implements|abstracts|prerequisite_of|uses|calls|compared_to|extends|replaces|solves|optimizes]->(Concept)

금지 패턴:
- (Segment)-[:APPEARS_IN]->(...)
- (...)-[:MENTIONS]->(Slide)
- (Concept)-[:APPEARS_IN]->(Slide)

모든 노드 패턴에는 반드시 {stem: $stem} 를 포함한다. 쿼리 실행 시 stem 파라미터가 전달된다.
"""

CYPHER_SYSTEM_PROMPT = f"""
너는 강의 영상 분석 그래프 DB를 조회하는 Cypher 전문가다.
아래 스키마를 기반으로 사용자 질문에 맞는 Cypher만 생성한다.

{GRAPH_SCHEMA}

규칙:
1. 읽기 전용: MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT / SKIP 만 사용한다. CREATE, MERGE, DELETE, SET, REMOVE, LOAD CSV, FOREACH, CALL 은 쓰지 않는다.
2. 모든 노드는 반드시 {{stem: $stem}} 조건을 포함한다. 예: MATCH (s:Slide {{stem: $stem}})
3. 파라미터는 $stem 만 외부에서 넣는다. 사용자 입력 문자열을 쿼리 문자열에 직접 이어붙이지 않는다. 검색은 $needle 등 추가 파라미터를 쓸 수 있다.
4. Cypher만 출력한다. 코드 블록(```cypher ... ```) 안에 작성한다.
5. RETURN에 필요한 필드만 명시한다. LIMIT는 30 이하로 둔다.
6. 타임스탬프가 필요하면 Segment.start, Segment.end, Scene.start, Scene.end, Slide.start_sec 등을 활용한다.
"""

ANSWER_SYSTEM_PROMPT = """
너는 강의 영상 분석 지식 그래프(및 선택적 보강 텍스트)를 바탕으로 질문에 답하는 어시스턴트다.

[근거 규칙]
- 답변의 사실·관계·용어는 반드시 제공된 "근거" 안에 있을 때만 쓴다.
- 근거에 없으면 추측하지 말고 짧게 "근거에 없어 답할 수 없다"고 한다.
- [Lance 보강 텍스트]가 있으면 그래프 근거와 모순 없을 때만 설명을 보강한다.

[답변 스타일]
1. 직접 답변을 먼저 한다. 길이는 질문에 맞게 조절한다.
2. 출처는 괄호로 짧게: (슬라이드 N), (약 t초). 근거에 없으면 억지로 쓰지 않는다.
3. 질문에 사례·예시·예를 들어 등이 있으면 근거에서 1개 사례를 넣고, 없으면 없다고 말한다.
4. 결과를 표·목록으로 그대로 나열하지 않고 문단으로 통합한다.
5. "정의:", "음성 발췌:" 같은 인위적 소제목 블록을 만들지 않는다.
6. 한국어로 답한다.
"""


class InternalQueryRequest(BaseModel):
    stem: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class InternalGraphQueryRequest(BaseModel):
    stem: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=GRAPH_TOP_K, ge=1, le=100)


class RetrievedChunk(BaseModel):
    chunk_id: str = ""
    stem: str = ""
    chunk_type: str = ""
    text: str = ""
    score: Optional[float] = None
    slide_number: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    linked_node_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    timestamps: list[dict]
    graph: dict
    retrieved_chunks: list[RetrievedChunk]


class GraphEvidence(BaseModel):
    src_id: str
    src_labels: list[str]
    rel_type: str
    tgt_id: str
    tgt_labels: list[str]
    src_text: str = ""
    tgt_text: str = ""


class GraphEvidenceResponse(BaseModel):
    stem: str
    question: str
    keywords: list[str]
    evidence: list[GraphEvidence]
    count: int


STRUCTURAL_KEYWORDS = [
    "몇 번",
    "슬라이드",
    "시간",
    "타임",
    "언제",
    "어디서",
    "강조",
    "몇 초",
    "구간",
    "씬",
    "장면",
    "scene",
    "몇 분",
]

STOPWORDS = {
    "무엇",
    "무슨",
    "어떤",
    "왜",
    "어떻게",
    "언제",
    "어디",
    "누가",
    "몇",
    "이유",
    "방법",
    "설명",
    "의미",
    "정의",
    "개념",
    "차이",
    "관계",
    "있나요",
    "있어요",
    "인가요",
    "인지",
    "하는",
    "하나요",
    "해줘",
    "알려줘",
    "대해",
    "대한",
    "위한",
    "위해",
    "그리고",
    "또는",
    "하지만",
    "그러나",
}

_CYPHER_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|LOAD\s+CSV|FOREACH|DROP|INSERT|CALL)\b",
    re.IGNORECASE,
)


def classify_question(question: str) -> str:
    for kw in STRUCTURAL_KEYWORDS:
        if kw in question:
            return "structural"
    return "content"


def extract_keywords_from_question(question: str) -> list[str]:
    keywords: list[str] = []
    quoted = re.findall(
        r"['\u2018\u2019\u201c\u201d\u300c\u300d]"
        r"([^'\u2018\u2019\u201c\u201d\u300c\u300d]+)"
        r"['\u2018\u2019\u201c\u201d\u300c\u300d]",
        question,
    )
    keywords.extend(quoted)

    clean = re.sub(r"['\u2018\u2019\u201c\u201d\u300c\u300d]", " ", question)
    clean = re.sub(r"[?？!！.,，。·]", " ", clean)
    tokens = clean.split()
    for tok in tokens:
        tok = re.sub(
            r"(은|는|이|가|을|를|의|에|에서|에게|로|으로|와|과|도|만|까지|부터|처럼|이란|란|으로서|로서)$",
            "",
            tok,
        )
        tok = tok.strip()
        if len(tok) >= 2 and tok not in STOPWORDS and tok not in keywords:
            keywords.append(tok)

    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _validate_readonly_cypher(cypher: str) -> None:
    if not cypher or not cypher.strip():
        raise ValueError("빈 Cypher")
    if _CYPHER_FORBIDDEN.search(cypher):
        raise ValueError("허용되지 않는 Cypher 키워드가 포함되어 있습니다.")
    if "$stem" not in cypher:
        raise ValueError("Cypher에 $stem 파라미터가 필요합니다.")


def _gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY 환경 변수가 없습니다.")
    return genai.Client(api_key=GEMINI_API_KEY)


def _call_gemini_raw(contents: str, system_instruction: str) -> str:
    client = _gemini_client()
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = client.models.generate_content(
                model=GEMINI_ANSWER_MODEL,
                contents=contents,
                config={"system_instruction": system_instruction},
            )
            return (r.text or "").strip()
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


def _call_gemini_answer(context: str, question: str) -> str:
    client = _gemini_client()
    contents = f"질문: {question}\n\n근거:\n{context}"
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = client.models.generate_content(
                model=GEMINI_ANSWER_MODEL,
                contents=contents,
                config={"system_instruction": ANSWER_SYSTEM_PROMPT},
            )
            return (r.text or "").strip()
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


def generate_cypher(question: str, stem: str) -> str:
    text = _call_gemini_raw(
        f"stem 파라미터 값은 실행 시 전달된다. 질문: {question}\n\n위 질문에 맞는 Cypher만 생성하라.",
        CYPHER_SYSTEM_PROMPT,
    )
    m = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _run_cypher_dicts(session, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = session.run(cypher, **params)
    return [dict(record) for record in result]


def run_cypher_structural(session, cypher: str, stem: str) -> list[dict[str, Any]]:
    _validate_readonly_cypher(cypher)
    return _run_cypher_dicts(session, cypher, {"stem": stem})


def run_cypher_safe_structural(session, question: str, stem: str) -> tuple[list[dict[str, Any]], str]:
    cypher = generate_cypher(question, stem)
    try:
        return run_cypher_structural(session, cypher, stem), cypher
    except Exception as e:
        fix_prompt = (
            f"다음 Cypher 실행 또는 검증에 실패했다:\n```cypher\n{cypher}\n```\n"
            f"오류: {e}\n\n읽기 전용만 사용하고 모든 노드에 {{stem: $stem}} 를 포함하라. "
            f"수정된 Cypher만 코드 블록(```cypher```)으로 출력하라."
        )
        fixed_text = _call_gemini_raw(fix_prompt, CYPHER_SYSTEM_PROMPT)
        m = re.search(r"```(?:cypher)?\s*(.*?)```", fixed_text, re.DOTALL)
        fixed_cypher = m.group(1).strip() if m else fixed_text.strip()
        return run_cypher_structural(session, fixed_cypher, stem), fixed_cypher


def _relevance_score(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def run_content_queries(
    session, stem: str, keywords: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_all: list[dict[str, Any]] = []
    seen_segs: set[tuple[Any, Any]] = set()
    seen_slides: set[Any] = set()

    q_sub = """
    MATCH (sub:Concept {stem: $stem})-[r]->(c:Concept {stem: $stem})
    WHERE type(r) IN ['part_of','is_a','implements']
      AND toLower(coalesce(c.name,'')) CONTAINS toLower($kw)
    RETURN coalesce(sub.id,'') AS sub_id, sub.name AS sub_concept,
           coalesce(c.id,'') AS concept_id, c.name AS parent_concept
    LIMIT 20
    """
    q_seg = """
    MATCH (slide:Slide {stem: $stem})-[:HAS_SCENE]->(scene:Scene {stem: $stem})
          -[:HAS_SEGMENT]->(seg:Segment {stem: $stem})-[:MENTIONS]->(c:Concept {stem: $stem})
    WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($kw)
    RETURN coalesce(seg.text,'') AS segment_text, seg.start AS start, seg.end AS end,
           slide.slide_number AS slide_number,
           coalesce(slide.id,'') AS slide_id, coalesce(seg.id,'') AS segment_id, coalesce(c.id,'') AS concept_id,
           c.name AS concept
    ORDER BY seg.start LIMIT 15
    """
    q_slide_concept = """
    MATCH (slide:Slide {stem: $stem})-[:APPEARS_IN]->(c:Concept {stem: $stem})
    WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($kw)
    RETURN slide.slide_number AS slide_number, coalesce(slide.id,'') AS slide_id,
           slide.title AS title, slide.slide_text AS slide_text, coalesce(c.id,'') AS concept_id
    ORDER BY slide.slide_number LIMIT 5
    """
    q_slide_text = """
    MATCH (slide:Slide {stem: $stem})
    WHERE toLower(coalesce(slide.slide_text,'')) CONTAINS toLower($kw)
    RETURN slide.slide_number AS slide_number, coalesce(slide.id,'') AS slide_id,
           slide.title AS title, slide.slide_text AS slide_text
    ORDER BY slide.slide_number LIMIT 5
    """
    q_seg_text = """
    MATCH (slide:Slide {stem: $stem})-[:HAS_SCENE]->(scene:Scene {stem: $stem})
          -[:HAS_SEGMENT]->(seg:Segment {stem: $stem})
    WHERE toLower(coalesce(seg.text,'')) CONTAINS toLower($kw)
    RETURN coalesce(seg.text,'') AS segment_text, seg.start AS start, seg.end AS end,
           slide.slide_number AS slide_number,
           coalesce(slide.id,'') AS slide_id, coalesce(seg.id,'') AS segment_id
    ORDER BY seg.start LIMIT 15
    """

    for kw in keywords:
        for key, q in (
            ("sub_concepts", q_sub),
            ("segments", q_seg),
            ("slides", q_slide_concept),
            ("slides", q_slide_text),
            ("segments", q_seg_text),
        ):
            for row in _run_cypher_dicts(session, q, {"stem": stem, "kw": kw}):
                raw_all.append(row)
                if key == "segments":
                    k2 = (row.get("slide_number"), row.get("start"))
                    if k2 not in seen_segs:
                        seen_segs.add(k2)
                        results["segments"].append(row)
                elif key == "slides":
                    sn = row.get("slide_number")
                    if sn not in seen_slides:
                        seen_slides.add(sn)
                        results["slides"].append(row)
                else:
                    results["sub_concepts"].append(row)

    if results.get("segments"):
        results["segments"].sort(
            key=lambda r: _relevance_score(str(r.get("segment_text", "")), keywords),
            reverse=True,
        )
    if results.get("slides"):
        results["slides"].sort(
            key=lambda r: _relevance_score(
                str(r.get("slide_text", "")) + str(r.get("title", "")), keywords
            ),
            reverse=True,
        )

    return dict(results), raw_all


def _build_content_context(question: str, structured: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        f"질문: {question}",
        "",
        "[지식 그래프 조회 결과 — 내용 질문]",
        "아래 데이터만 사실로 사용한다.",
    ]
    if structured.get("sub_concepts"):
        subs = list(
            dict.fromkeys(str(r.get("sub_concept", "")) for r in structured["sub_concepts"] if r.get("sub_concept"))
        )
        if subs:
            lines.append(f"[구성 개념] {', '.join(subs[:15])}")

    if structured.get("segments"):
        lines.append("")
        lines.append("[음성·개념 연결 구간]")
        for r in structured["segments"][:8]:
            t = r.get("start", 0) or 0
            try:
                t = float(t)
            except (TypeError, ValueError):
                t = 0.0
            mm, ss = int(t) // 60, int(t) % 60
            lines.append(f"  {mm}:{ss:02d} | {r.get('segment_text', '')}")
            if r.get("concept"):
                lines.append(f"    (개념: {r.get('concept')})")

    if structured.get("slides"):
        lines.append("")
        lines.append("[슬라이드 본문 요약]")
        for r in structured["slides"][:3]:
            lines.append(
                f"  슬라이드 {r.get('slide_number')} '{r.get('title', '')}'\n"
                f"  {str(r.get('slide_text', ''))[:400]}"
            )

    return "\n".join(lines)


def _build_structural_context(question: str, raw_rows: list[dict[str, Any]], cypher: str) -> str:
    lines = [
        f"질문: {question}",
        "",
        "[지식 그래프 조회 결과 — 구조 질문]",
        f"실행 Cypher:\n{cypher[:2000]}",
        "",
        "결과(JSON, 상위 25행):",
        json.dumps(raw_rows[:25], ensure_ascii=False, default=str),
    ]
    return "\n".join(lines)


def _collect_ids_from_content(structured: dict[str, list[dict[str, Any]]]) -> set[str]:
    ids: set[str] = set()
    for key in ("sub_concepts", "segments", "slides"):
        for r in structured.get(key, []):
            for fld in ("sub_id", "concept_id", "slide_id", "segment_id"):
                v = r.get(fld)
                if v:
                    ids.add(str(v).strip())
    return {x for x in ids if x}


def _collect_ids_from_raw_rows(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    id_like = re.compile(r"^(slide_|segment/|concept/|annotation/)")

    def walk(v: Any) -> None:
        if isinstance(v, str) and id_like.match(v):
            ids.add(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for row in rows:
        walk(row)
    return ids


def _timestamps_from_content(structured: dict[str, list[dict[str, Any]]]) -> list[dict]:
    ts: list[dict] = []
    for r in structured.get("segments", []):
        start = r.get("start")
        if start is None:
            continue
        try:
            s = float(start)
        except (TypeError, ValueError):
            continue
        end = r.get("end")
        try:
            e = float(end) if end is not None else s
        except (TypeError, ValueError):
            e = s
        label = str(r.get("segment_text", ""))[:50]
        ts.append({"label": label, "start": s, "end": e})
    ts.sort(key=lambda x: x["start"])
    return ts[:20]


def _timestamps_from_structural_rows(rows: list[dict[str, Any]]) -> list[dict]:
    ts: list[dict] = []
    for row in rows:
        for key, val in row.items():
            if key in ("start", "start_sec") and val is not None:
                try:
                    s = float(val)
                except (TypeError, ValueError):
                    continue
                end_val = row.get("end") or row.get("end_sec") or s
                try:
                    e = float(end_val)
                except (TypeError, ValueError):
                    e = s
                label = str(row.get("text") or row.get("title") or row.get("name") or "구간")[:50]
                ts.append({"label": label, "start": s, "end": e})
    seen: set[tuple[float, float]] = set()
    out: list[dict] = []
    for t in sorted(ts, key=lambda x: x["start"]):
        k = (t["start"], t["end"])
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:20]


def _graph_from_content_structured(structured: dict[str, list[dict[str, Any]]]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(nid: str, label: str, ntype: str, title: str = "") -> None:
        if nid and nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "label": (label or nid)[:30],
                "color": "#FF6B6B" if ntype == "Concept" else "#4ECDC4",
                "title": title[:300],
                "type": ntype,
            }

    for r in structured.get("sub_concepts", []):
        sid, cid = str(r.get("sub_id", "")), str(r.get("concept_id", ""))
        if sid:
            add_node(sid, str(r.get("sub_concept", sid)), "Concept")
        if cid:
            add_node(cid, str(r.get("parent_concept", cid)), "Concept")
        if sid and cid:
            edges.append({"from": sid, "to": cid, "label": "concept_rel"})

    for r in structured.get("segments", []):
        slid, segid, cid = (
            str(r.get("slide_id", "")),
            str(r.get("segment_id", "")),
            str(r.get("concept_id", "")),
        )
        if slid:
            add_node(slid, f"S{r.get('slide_number')}", "Slide", str(r.get("segment_text", ""))[:200])
        if segid:
            add_node(segid, "seg", "Segment", str(r.get("segment_text", ""))[:200])
        if cid:
            add_node(cid, str(r.get("concept", cid)), "Concept")
        if slid and segid:
            edges.append({"from": slid, "to": segid, "label": "HAS_SEGMENT"})
        if segid and cid:
            edges.append({"from": segid, "to": cid, "label": "MENTIONS"})

    for r in structured.get("slides", []):
        slid = str(r.get("slide_id", ""))
        cid = str(r.get("concept_id", ""))
        if slid:
            add_node(
                slid,
                f"S{r.get('slide_number')}",
                "Slide",
                str(r.get("slide_text", ""))[:200],
            )
        if cid:
            add_node(cid, "", "Concept")
        if slid and cid:
            edges.append({"from": slid, "to": cid, "label": "APPEARS_IN"})

    return {"nodes": list(nodes.values()), "edges": edges}


def _graph_from_structural_rows(rows: list[dict[str, Any]]) -> dict:
    """LLM Cypher 결과가 다양해 완전한 그래프는 어렵고, 노드 id 문자열만 수집."""
    nodes: dict[str, dict] = {}
    id_like = re.compile(r"^(slide_|segment/|concept/|annotation/)")

    def add_from_val(v: Any) -> None:
        if isinstance(v, str) and id_like.match(v) and v not in nodes:
            nodes[v] = {"id": v, "label": v[:30], "color": "#B0C4DE", "title": "", "type": "node"}

    for row in rows:
        for v in row.values():
            add_from_val(v)
            if isinstance(v, dict):
                for x in v.values():
                    add_from_val(x)
    return {"nodes": list(nodes.values()), "edges": []}


def _distance_to_score(dist: float) -> float:
    if dist is None or math.isnan(dist):
        return 0.0
    return 1.0 / (1.0 + float(dist))


def _rows_to_chunks(df) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for _, row in df.iterrows():
        dist = row.get("_distance")
        try:
            d = float(dist) if dist is not None else None
        except (TypeError, ValueError):
            d = None
        text = str(row.get("text", ""))
        out.append(
            RetrievedChunk(
                chunk_id=str(row.get("chunk_id", "")),
                stem=str(row.get("stem", "")),
                chunk_type=str(row.get("chunk_type", "")),
                text=text[:2000],
                score=_distance_to_score(d) if d is not None else None,
                slide_number=row.get("slide_number"),
                start_sec=float(row["start_sec"]) if row.get("start_sec") is not None else None,
                end_sec=float(row["end_sec"]) if row.get("end_sec") is not None else None,
                linked_node_id=str(row["linked_node_id"]) if row.get("linked_node_id") is not None else None,
            )
        )
    return out


def _chunks_to_timestamps(chunks: list[RetrievedChunk]) -> list[dict]:
    ts: list[dict] = []
    for c in chunks:
        if c.start_sec is None:
            continue
        label = (c.text or "")[:50]
        ts.append(
            {
                "label": label,
                "start": c.start_sec,
                "end": c.end_sec if c.end_sec is not None else c.start_sec,
            }
        )
    return ts[:20]


def _extract_keywords(question: str) -> list[str]:
    parts = re.split(r"[^0-9A-Za-z가-힣_]+", (question or "").lower())
    out: list[str] = []
    for p in parts:
        if len(p) < 2:
            continue
        if p not in out:
            out.append(p)
    return out[:8]


def _select_lance_supporting_chunks(
    chunks: list[RetrievedChunk], allowed_node_ids: set[str], max_items: int = 4
) -> list[RetrievedChunk]:
    if not chunks or not allowed_node_ids:
        return []
    selected: list[RetrievedChunk] = []
    for c in chunks:
        linked = (c.linked_node_id or "").strip()
        if linked and linked in allowed_node_ids:
            selected.append(c)
            if len(selected) >= max_items:
                break
    return selected


def _build_hybrid_context_base(graph_context: str, supporting_chunks: list[RetrievedChunk]) -> str:
    lines = [graph_context]
    if supporting_chunks:
        lines.extend(["", "[Lance 보강 텍스트]"])
        for i, c in enumerate(supporting_chunks, start=1):
            meta = f"(슬라이드 {c.slide_number}" if c.slide_number is not None else "(출처"
            if c.start_sec is not None:
                meta += f", {c.start_sec:.1f}초"
            meta += ")"
            lines.append(f"{i}. {meta} {c.text[:500]}")
        lines.append("Lance 텍스트는 그래프 근거를 보강하는 범위에서만 사용한다.")
    return "\n".join(lines)


def _neo4j_driver():
    if not NEO4J_URI or not NEO4J_USER:
        raise HTTPException(
            status_code=503,
            detail="Neo4j 설정이 없습니다. NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD를 확인하세요.",
        )
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except (ServiceUnavailable, Neo4jError) as e:
        raise HTTPException(status_code=503, detail=f"Neo4j 연결 실패: {e}") from e


def graph_search_by_stem_question(stem: str, question: str, top_k: int = GRAPH_TOP_K) -> list[GraphEvidence]:
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    driver = _neo4j_driver()
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (a {stem: $stem})-[r]->(b {stem: $stem})
                WITH a, r, b,
                     toLower(
                        coalesce(a.id, '') + ' ' +
                        coalesce(a.title, '') + ' ' +
                        coalesce(a.text, '') + ' ' +
                        coalesce(a.name, '') + ' ' +
                        coalesce(b.id, '') + ' ' +
                        coalesce(b.title, '') + ' ' +
                        coalesce(b.text, '') + ' ' +
                        coalesce(b.name, '')
                     ) AS haystack
                WITH a, r, b, haystack, [kw IN $keywords WHERE haystack CONTAINS kw] AS hits
                WHERE size(hits) > 0
                RETURN
                    coalesce(a.id, '') AS src_id,
                    labels(a) AS src_labels,
                    type(r) AS rel_type,
                    coalesce(b.id, '') AS tgt_id,
                    labels(b) AS tgt_labels,
                    coalesce(a.text, a.title, a.name, '') AS src_text,
                    coalesce(b.text, b.title, b.name, '') AS tgt_text,
                    size(hits) AS score
                ORDER BY score DESC, rel_type ASC
                LIMIT $top_k
                """,
                stem=stem,
                keywords=keywords,
                top_k=int(top_k),
            )
            out: list[GraphEvidence] = []
            for row in rows:
                out.append(
                    GraphEvidence(
                        src_id=str(row.get("src_id", "")),
                        src_labels=[str(x) for x in (row.get("src_labels") or [])],
                        rel_type=str(row.get("rel_type", "")),
                        tgt_id=str(row.get("tgt_id", "")),
                        tgt_labels=[str(x) for x in (row.get("tgt_labels") or [])],
                        src_text=str(row.get("src_text", "") or "")[:800],
                        tgt_text=str(row.get("tgt_text", "") or "")[:800],
                    )
                )
            return out
    finally:
        driver.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/internal/query_graph_evidence", response_model=GraphEvidenceResponse)
async def internal_query_graph_evidence(req: InternalGraphQueryRequest) -> GraphEvidenceResponse:
    stem = req.stem.strip()
    question = req.question.strip()
    if not stem or not question:
        raise HTTPException(status_code=400, detail="stem/question은 비어 있을 수 없습니다.")

    evidence = graph_search_by_stem_question(stem=stem, question=question, top_k=req.top_k)
    return GraphEvidenceResponse(
        stem=stem,
        question=question,
        keywords=_extract_keywords(question),
        evidence=evidence,
        count=len(evidence),
    )


def _empty_query_response() -> QueryResponse:
    return QueryResponse(
        answer="그래프 근거가 없어 답변할 수 없습니다. 질문을 더 구체화하거나 강의 그래프 생성/적재 상태를 확인해주세요.",
        timestamps=[],
        graph={"nodes": [], "edges": []},
        retrieved_chunks=[],
    )


@app.post("/internal/query", response_model=QueryResponse)
async def internal_query(req: InternalQueryRequest) -> QueryResponse:
    stem = req.stem.strip()
    question = req.question.strip()
    if not stem or not question:
        raise HTTPException(status_code=400, detail="stem/question은 비어 있을 수 없습니다.")

    driver = _neo4j_driver()
    q_type = classify_question(question)
    graph_context = ""
    allowed_ids: set[str] = set()
    timestamps: list[dict] = []
    graph: dict = {"nodes": [], "edges": []}

    try:
        with driver.session() as session:
            if q_type == "content":
                keywords = extract_keywords_from_question(question)
                if not keywords:
                    return _empty_query_response()
                structured, _raw = run_content_queries(session, stem, keywords)
                n_total = sum(len(v) for v in structured.values())
                if n_total == 0:
                    return _empty_query_response()
                graph_context = _build_content_context(question, structured)
                allowed_ids = _collect_ids_from_content(structured)
                timestamps = _timestamps_from_content(structured)
                graph = _graph_from_content_structured(structured)
            else:
                try:
                    raw_rows, cypher_used = run_cypher_safe_structural(session, question, stem)
                except Exception:
                    return _empty_query_response()
                if not raw_rows:
                    return _empty_query_response()
                graph_context = _build_structural_context(question, raw_rows, cypher_used)
                allowed_ids = _collect_ids_from_raw_rows(raw_rows)
                timestamps = _timestamps_from_structural_rows(raw_rows)
                graph = _graph_from_structural_rows(raw_rows)
    finally:
        driver.close()

    supporting_chunks: list[RetrievedChunk] = []
    try:
        lance_root = default_lance_root()
        if lance_root.exists() and allowed_ids:
            df = lance_search(
                stem=stem,
                query=question,
                lance_root=lance_root,
                top_k=TOP_K,
            )
            if df is not None and len(df) > 0:
                supporting_chunks = _select_lance_supporting_chunks(
                    _rows_to_chunks(df),
                    allowed_node_ids=allowed_ids,
                    max_items=4,
                )
    except Exception:
        supporting_chunks = []

    context = _build_hybrid_context_base(graph_context, supporting_chunks)
    try:
        answer = _call_gemini_answer(context, question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 답변 생성 실패: {e}") from e

    return QueryResponse(
        answer=answer,
        timestamps=_chunks_to_timestamps(supporting_chunks) if supporting_chunks else timestamps,
        graph=graph,
        retrieved_chunks=supporting_chunks,
    )
