"""
graph_triples 리스트 → Parquet 3종 저장:
  - {stem}_graph_triples.parquet  (중간 산출물, CSV와 동일 스키마)
  - {stem}_nodes.parquet
  - {stem}_edges.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)

TripleRow = Tuple[str, str, str, str]  # subject, predicate, object, properties_str


class ParquetPaths(TypedDict):
    triples: str
    nodes: str
    edges: str


def triples_to_dataframes(triples: List[TripleRow], stem: str):
    """트리플 행 리스트 → triples / nodes / edges DataFrame."""
    triple_rows: List[Dict[str, Any]] = []
    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    for subj, pred, obj, props_str in triples:
        p = props_str or ""
        triple_rows.append(
            {
                "stem": stem,
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "properties": p,
            }
        )
        if pred == "type":
            if subj not in seen_node_ids:
                seen_node_ids.add(subj)
                node_rows.append(
                    {
                        "stem": stem,
                        "node_id": subj,
                        "label": obj,
                        "properties_json": p if p.strip() else "{}",
                    }
                )
        else:
            edge_rows.append(
                {
                    "stem": stem,
                    "src_id": subj,
                    "rel_type": pred,
                    "tgt_id": obj,
                    "properties_json": p if p.strip() else "{}",
                }
            )

    return (
        pd.DataFrame(triple_rows),
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
    )


def write_graph_parquet_bundle(
    triples: List[TripleRow],
    stem: str,
    output_dir: Path,
) -> ParquetPaths:
    """Parquet 3개 저장. 반환: 절대 경로 문자열 dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tdf, ndf, edf = triples_to_dataframes(triples, stem)

    paths = {
        "triples": output_dir / f"{stem}_graph_triples.parquet",
        "nodes": output_dir / f"{stem}_nodes.parquet",
        "edges": output_dir / f"{stem}_edges.parquet",
    }

    tdf.to_parquet(paths["triples"], index=False)
    ndf.to_parquet(paths["nodes"], index=False)
    edf.to_parquet(paths["edges"], index=False)

    logger.info(
        "✓ Parquet 저장: triples=%d, nodes=%d, edges=%d → %s",
        len(tdf),
        len(ndf),
        len(edf),
        output_dir,
    )

    return {
        "triples": str(paths["triples"].resolve()),
        "nodes": str(paths["nodes"].resolve()),
        "edges": str(paths["edges"].resolve()),
    }
