"""
강의 stem에 해당하는 Parquet(nodes/edges) → 질의 API와 동일 형태의 graph JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def is_safe_stem(stem: str) -> bool:
    """파일명에 사용할 stem. 경로 구분자·이탈 시퀀스만 금지."""
    if not stem or len(stem) > 255:
        return False
    if stem.strip() != stem:
        return False
    if "/" in stem or "\\" in stem or ".." in stem:
        return False
    return True


def _hex_color(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def _str_cell(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except TypeError:
        pass
    return str(x)


def load_kg_from_parquet(output_dir: Path, stem: str) -> dict[str, Any]:
    """
    {stem}_nodes.parquet, {stem}_edges.parquet 를 읽어
    { nodes: [...], edges: [...], node_count, edge_count } 반환.
    엣지 끝점이 nodes에 없으면 스텁 노드를 추가한다.
    """
    if not is_safe_stem(stem):
        raise ValueError("유효하지 않은 stem 입니다.")

    base = output_dir.resolve()
    nodes_path = (base / f"{stem}_nodes.parquet").resolve()
    edges_path = (base / f"{stem}_edges.parquet").resolve()

    if not str(nodes_path).startswith(str(base)) or not str(edges_path).startswith(str(base)):
        raise ValueError("경로가 허용 범위를 벗어났습니다.")

    if not nodes_path.is_file():
        raise FileNotFoundError(f"노드 파일이 없습니다: {nodes_path.name} (Stage 6 Parquet 생성 여부 확인)")

    ndf = pd.read_parquet(nodes_path)
    edf = (
        pd.read_parquet(edges_path)
        if edges_path.is_file()
        else pd.DataFrame(columns=["src_id", "rel_type", "tgt_id", "properties_json", "stem"])
    )

    nodes_out: list[dict[str, Any]] = []
    for _, row in ndf.iterrows():
        nid = _str_cell(row.get("node_id"))
        if not nid:
            continue
        label = _str_cell(row.get("label")) or nid
        props = _str_cell(row.get("properties_json")) or "{}"
        title = props[:800]
        try:
            obj = json.loads(props) if props.strip() else {}
            ntype = _str_cell(obj.get("type")) if isinstance(obj, dict) else ""
        except json.JSONDecodeError:
            ntype = ""
        nodes_out.append(
            {
                "id": nid,
                "label": label[:120],
                "title": title,
                "type": ntype or "node",
                "color": _hex_color(label),
            }
        )

    seen_ids = {n["id"] for n in nodes_out}

    edges_out: list[dict[str, Any]] = []
    for _, row in edf.iterrows():
        src = _str_cell(row.get("src_id"))
        tgt = _str_cell(row.get("tgt_id"))
        rel = _str_cell(row.get("rel_type")) or "related"
        if not src or not tgt:
            continue
        edges_out.append({"from": src, "to": tgt, "label": rel})
        for x in (src, tgt):
            if x not in seen_ids:
                seen_ids.add(x)
                nodes_out.append(
                    {
                        "id": x,
                        "label": x[:80],
                        "title": "",
                        "type": "orphan",
                        "color": "#9ca3af",
                    }
                )

    return {
        "stem": stem,
        "node_count": len(nodes_out),
        "edge_count": len(edges_out),
        "nodes": nodes_out,
        "edges": edges_out,
    }
