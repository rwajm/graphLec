"""
GraphLEC 질의 전용 FastAPI 서비스.

Django는 /internal/query 로 stem + question 만 전달한다 (lecture_id → stem 변환은 Django 담당).

LanceDB 벡터 검색 + Gemini 답변 생성.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Optional

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

ANSWER_SYSTEM_PROMPT = """
너는 강의 영상 분석 결과를 바탕으로 질문에 직접 답하는 어시스턴트야.

답변 형식:
1. 질문에 대한 직접적인 답변을 먼저 제시한다. (2~4문장)
2. 필요한 경우 핵심 근거만 간결하게 덧붙인다. (슬라이드 번호 또는 시간 표기)
3. 제공된 근거 텍스트만 사용하고, 없는 내용은 추측하지 않는다.
4. 조회된 데이터를 그대로 나열하지 않는다.
5. 한국어로 답변한다.
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


def _gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY 환경 변수가 없습니다.")
    return genai.Client(api_key=GEMINI_API_KEY)


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


def _chunks_to_graph(chunks: list[RetrievedChunk]) -> dict:
    nodes = []
    for c in chunks:
        nid = c.chunk_id or c.linked_node_id or "unknown"
        ctype = c.chunk_type or (
            "segment" if (c.chunk_id or "").startswith("segment/") else "slide"
        )
        label = f"S{c.slide_number} {ctype}"[:40]
        nodes.append(
            {
                "id": nid,
                "label": label.strip(),
                "color": "#4ECDC4" if ctype == "slide" else "#96CEB4",
                "title": (c.text or "")[:300],
                "type": ctype,
            }
        )
    return {"nodes": nodes, "edges": []}


def _extract_keywords(question: str) -> list[str]:
    parts = re.split(r"[^0-9A-Za-z가-힣_]+", (question or "").lower())
    out: list[str] = []
    for p in parts:
        if len(p) < 2:
            continue
        if p not in out:
            out.append(p)
    return out[:8]


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


@app.post("/internal/query", response_model=QueryResponse)
async def internal_query(req: InternalQueryRequest) -> QueryResponse:
    lance_root = default_lance_root()
    if not lance_root.exists():
        raise HTTPException(
            status_code=503,
            detail=f"LanceDB 데이터 없음: {lance_root}. 파이프라인 Stage 7 (Lance 인덱스)를 실행하세요.",
        )

    try:
        df = lance_search(
            stem=req.stem,
            query=req.question,
            lance_root=lance_root,
            top_k=TOP_K,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {e}") from e

    if df is None or len(df) == 0:
        return QueryResponse(
            answer="선택한 강의(stem)에 대한 검색 인덱스가 비어 있거나, 질문과 맞는 구간을 찾지 못했습니다.",
            timestamps=[],
            graph={"nodes": [], "edges": []},
            retrieved_chunks=[],
        )

    chunks = _rows_to_chunks(df)
    context_parts = []
    for c in chunks:
        meta = f"(슬라이드 {c.slide_number}" if c.slide_number is not None else "(출처"
        if c.start_sec is not None:
            meta += f", {c.start_sec:.1f}초"
        meta += ")"
        context_parts.append(f"{meta}\n{c.text}")

    context = "\n\n---\n\n".join(context_parts)
    try:
        answer = _call_gemini_answer(context, req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 답변 생성 실패: {e}") from e

    return QueryResponse(
        answer=answer,
        timestamps=_chunks_to_timestamps(chunks),
        graph=_chunks_to_graph(chunks),
        retrieved_chunks=chunks,
    )
