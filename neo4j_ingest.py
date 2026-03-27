"""
{stem}_nodes.parquet / {stem}_edges.parquet → Neo4j 적재.

레거시 import_graph.py(graph_triples.csv)와 동일한 규칙:
  - predicate == 'type' 에 해당하는 노드: 동적 라벨, MERGE 키는 id(= subject / Parquet의 node_id)
  - 관계: 동적 관계 타입, MATCH (a {{id: src}})-(b {{id: tgt}}) 패턴
  - flatten_props: dict/list 만 JSON 문자열로 (그 외는 그대로) — 레거시와 동일
  - 적재 후 라벨별 인덱스 시도 (레거시: ON (n.id); 멀티 stem 대응: ON (n.stem, n.id))

추가(이 프로젝트 정책):
  - 노드·관계 매칭에 stem 사용 (MATCH (a {{stem, id}})) — 동일 id가 다른 강의에 있을 수 있음
  - 재적재: 전체 DB가 아니라 해당 stem 서브그래프만 DETACH DELETE (레거시 --reset 과 다름)

환경 변수: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent / ".env")

_NEO4J_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_identifier(name: str, kind: str) -> str:
    if not name or not _NEO4J_ID_RE.match(name):
        raise ValueError(f"유효하지 않은 Neo4j {kind} 식별자: {name!r}")
    return name


def flatten_props(props: dict[str, Any]) -> dict[str, Any]:
    """레거시 import_graph.py 와 동일: Neo4j는 MAP 타입 프로퍼티 불가 → dict/list는 JSON 문자열."""
    return {
        k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        for k, v in props.items()
    }


def _parse_properties_json(raw: Any) -> dict[str, Any]:
    s = raw if isinstance(raw, str) else str(raw) if raw is not None else ""
    s = s.strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"properties_json 파싱 실패: {e}") from e
    if not isinstance(obj, dict):
        return {}
    return obj


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _delete_stem(tx, stem: str) -> None:
    tx.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)


def _sanitize_index_token(label: str) -> str:
    """CREATE INDEX 이름용 (영숫자·언더스코어만)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if s and s[0].isdigit():
        s = "L_" + s
    return s or "L"


def ingest_parquet_to_neo4j(
    stem: str,
    output_dir: Path,
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """
    Parquet → Neo4j. 연결 실패·쿼리 오류 시 예외.

    Returns:
        node_count, edge_count, elapsed_sec, uri
    """
    t0 = time.time()
    nodes_path = output_dir / f"{stem}_nodes.parquet"
    edges_path = output_dir / f"{stem}_edges.parquet"

    if not nodes_path.is_file():
        raise FileNotFoundError(f"Neo4j 적재: 노드 Parquet 없음 — {nodes_path}")

    uri = (uri or os.getenv("NEO4J_URI", "")).strip()
    user = (user or os.getenv("NEO4J_USER", "")).strip()
    if password is None:
        password = os.getenv("NEO4J_PASSWORD", "")
    if not uri or not user:
        raise RuntimeError(
            "Neo4j 적재에 NEO4J_URI, NEO4J_USER(및 필요 시 NEO4J_PASSWORD) 환경 변수가 필요합니다. "
            "적재를 건너뛰려면 --skip-neo4j 를 지정하세요."
        )

    ndf = pd.read_parquet(nodes_path)
    edf = pd.read_parquet(edges_path) if edges_path.is_file() else pd.DataFrame()

    driver = GraphDatabase.driver(uri, auth=(user, password))
    c_nodes = 0
    c_rels = 0
    labels_seen: set[str] = set()

    try:
        try:
            driver.verify_connectivity()
        except (ServiceUnavailable, Neo4jError) as e:
            raise RuntimeError(
                f"Neo4j 연결 실패: {uri}. 서버가 떠 있는지, URI·인증·방화벽을 확인하세요. ({e})"
            ) from e

        def work(tx) -> None:
            nonlocal labels_seen
            _delete_stem(tx, stem)

            by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for _, row in ndf.iterrows():
                node_id = str(row.get("node_id", "")).strip()
                if not node_id:
                    continue
                row_stem = str(row.get("stem", stem)).strip() or stem
                if row_stem != stem:
                    raise ValueError(
                        f"Parquet stem({row_stem!r})이 인자 stem({stem!r})과 일치하지 않습니다."
                    )
                label = str(row.get("label", "")).strip()
                if not label:
                    raise ValueError(f"노드 {node_id!r} 에 label 이 없습니다.")
                _require_identifier(label, "라벨")
                labels_seen.add(label)

                raw = _parse_properties_json(row.get("properties_json"))
                props = _strip_none(flatten_props(raw))
                props["id"] = node_id
                props["stem"] = stem
                by_label[label].append(props)

            for label, props_list in by_label.items():
                lab = _require_identifier(label, "라벨")
                tx.run(
                    f"""
                    UNWIND $props_list AS props
                    MERGE (n:`{lab}` {{stem: props.stem, id: props.id}})
                    SET n += props
                    """,
                    props_list=props_list,
                )

            by_rel: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for _, row in edf.iterrows():
                src = str(row.get("src_id", "")).strip()
                tgt = str(row.get("tgt_id", "")).strip()
                rel_type = str(row.get("rel_type", "")).strip()
                if not src or not tgt or not rel_type:
                    continue
                row_stem = str(row.get("stem", stem)).strip() or stem
                if row_stem != stem:
                    raise ValueError(
                        f"Parquet stem({row_stem!r})이 인자 stem({stem!r})과 일치하지 않습니다."
                    )
                _require_identifier(rel_type, "관계 타입")
                raw = _parse_properties_json(row.get("properties_json"))
                item_props = _strip_none(flatten_props(raw))
                by_rel[rel_type].append({"src": src, "tgt": tgt, "props": item_props})

            for rtype, items in by_rel.items():
                rt = _require_identifier(rtype, "관계 타입")
                tx.run(
                    f"""
                    UNWIND $items AS item
                    MATCH (a {{stem: $stem, id: item.src}}), (b {{stem: $stem, id: item.tgt}})
                    MERGE (a)-[r:`{rt}`]->(b)
                    SET r += item.props
                    """,
                    stem=stem,
                    items=items,
                )

        with driver.session() as session:
            session.execute_write(work)

            cn = session.run(
                "MATCH (n {stem: $stem}) RETURN count(n) AS c",
                stem=stem,
            ).single()
            c_nodes = int(cn["c"]) if cn else 0
            cr = session.run(
                """
                MATCH (a {stem: $stem})-[r]->(b {stem: $stem})
                RETURN count(r) AS c
                """,
                stem=stem,
            ).single()
            c_rels = int(cr["c"]) if cr else 0

            for label in sorted(labels_seen):
                lab = _require_identifier(label, "라벨")
                token = _sanitize_index_token(lab)
                idx_name = f"{token}_stem_id"
                try:
                    session.run(
                        f"CREATE INDEX {idx_name} IF NOT EXISTS FOR (n:`{lab}`) ON (n.stem, n.id)"
                    )
                except Exception as e:
                    logger.debug("인덱스 생성 스킵 %s: %s", idx_name, e)

    finally:
        driver.close()

    elapsed = time.time() - t0
    logger.info(
        "Neo4j 적재 완료 stem=%s nodes=%d edges=%d (%.2fs)",
        stem,
        c_nodes,
        c_rels,
        elapsed,
    )
    return {
        "stem": stem,
        "node_count": c_nodes,
        "edge_count": c_rels,
        "elapsed_sec": elapsed,
        "uri": uri,
    }


def stage_neo4j_ingest(args, output_dir: Path) -> dict[str, Any]:
    stem = Path(args.input).stem
    return ingest_parquet_to_neo4j(stem=stem, output_dir=output_dir)
