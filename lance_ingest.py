"""
fused.json → 청크 생성 → Gemini 임베딩 → Parquet 백업 + LanceDB 적재.

단일 LanceDB 테이블 `chunks` 에 stem 컬럼으로 강의를 구분한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import numpy as np
import pandas as pd

from embedding_utils import DEFAULT_EMBEDDING_MODEL, embed_documents, get_genai_client

CHUNKS_TABLE = "chunks"
BATCH_SIZE = 16


def default_lance_root() -> Path:
    env = os.getenv("GRAPHLEC_LANCE_ROOT")
    if env:
        return Path(env).resolve()
    # 레포 루트 기준 (CWD와 무관)
    repo_root = Path(__file__).resolve().parent
    return (repo_root / "data" / "lancedb").resolve()


def build_chunks_from_fused(fused: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """슬라이드 본문 + 세그먼트 전사를 검색용 청크로 만든다."""
    slides = fused.get("slides") or []
    chunks: List[Dict[str, Any]] = []
    seg_idx = 0

    for slide in slides:
        sid = slide.get("slide_id", "")
        sn = slide.get("slide_number")
        title = (slide.get("title") or "").strip()
        stext = (slide.get("slide_text") or "").strip()
        body = f"[{sid}] {title}\n{stext}".strip()
        if body:
            chunks.append(
                {
                    "chunk_id": f"{sid}_slide_body",
                    "stem": stem,
                    "chunk_type": "slide",
                    "text": body[:12000],
                    "slide_id": sid,
                    "slide_number": sn,
                    "start_sec": slide.get("start_sec"),
                    "end_sec": slide.get("end_sec"),
                    "linked_node_id": sid,
                }
            )

        for ctx in slide.get("contexts") or []:
            for seg in ctx.get("segments") or []:
                seg_id = f"segment/{seg_idx:04d}"
                seg_idx += 1
                t = (seg.get("text") or "").strip()
                if not t:
                    continue
                chunks.append(
                    {
                        "chunk_id": seg_id,
                        "stem": stem,
                        "chunk_type": "segment",
                        "text": f"[{seg_id}] [{sid}] {t}"[:12000],
                        "slide_id": sid,
                        "slide_number": sn,
                        "start_sec": seg.get("start"),
                        "end_sec": seg.get("end"),
                        "linked_node_id": seg_id,
                    }
                )

    return chunks


def ingest_stem_to_lance(
    *,
    stem: str,
    fused_path: Path,
    output_dir: Path,
    lance_root: Optional[Path] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """
    fused_path의 강의를 임베딩해 LanceDB에 넣는다.
    동일 stem 기존 행은 삭제 후 재삽입한다.
    """
    lance_root = lance_root or default_lance_root()
    lance_root.mkdir(parents=True, exist_ok=True)

    if not fused_path.exists():
        raise FileNotFoundError(f"fused 파일 없음: {fused_path}")

    with open(fused_path, encoding="utf-8") as f:
        fused = json.load(f)

    chunks = build_chunks_from_fused(fused, stem)
    if not chunks:
        return {"ok": False, "reason": "no chunks", "count": 0}

    client = get_genai_client()
    texts = [c["text"] for c in chunks]
    vectors: List[List[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vectors.extend(embed_documents(client, batch, model=embedding_model))

    rows: List[Dict[str, Any]] = []
    for c, vec in zip(chunks, vectors):
        rows.append(
            {
                **c,
                "vector": vec,
                "embedding_model": embedding_model,
            }
        )

    df = pd.DataFrame(rows)
    parquet_path = output_dir / f"{stem}_chunks_lance.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    db = lancedb.connect(str(lance_root))
    if CHUNKS_TABLE in db.table_names():
        table = db.open_table(CHUNKS_TABLE)
        stem_sql = stem.replace("'", "''")
        table.delete(f"stem = '{stem_sql}'")
        table.add(df)
    else:
        db.create_table(CHUNKS_TABLE, data=df)

    return {
        "ok": True,
        "stem": stem,
        "count": len(rows),
        "parquet_path": str(parquet_path),
        "lance_root": str(lance_root),
        "table": CHUNKS_TABLE,
        "embedding_model": embedding_model,
    }


def lance_search(
    *,
    stem: str,
    query: str,
    lance_root: Optional[Path] = None,
    top_k: int = 8,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> pd.DataFrame:
    """질의 벡터로 stem 범위 내 top-k 검색."""
    from embedding_utils import embed_query

    lance_root = lance_root or default_lance_root()
    if not lance_root.exists():
        raise FileNotFoundError(f"LanceDB 경로 없음: {lance_root}")

    db = lancedb.connect(str(lance_root))
    if CHUNKS_TABLE not in db.table_names():
        raise FileNotFoundError(f"LanceDB에 '{CHUNKS_TABLE}' 테이블이 없습니다. 먼저 파이프라인 Stage 7을 실행하세요.")

    client = get_genai_client()
    qv = np.array(embed_query(client, query, model=embedding_model), dtype=np.float32)
    table = db.open_table(CHUNKS_TABLE)
    stem_sql = stem.replace("'", "''")
    return (
        table.search(qv)
        .where(f"stem = '{stem_sql}'", prefilter=True)
        .limit(top_k)
        .to_pandas()
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="fused.json → Parquet + LanceDB 적재 (단일 강의)")
    p.add_argument("--stem", required=True, help="강의 stem (파일명과 동일)")
    p.add_argument("--output-dir", default="output", type=Path, help="분석 출력 디렉터리")
    p.add_argument("--slides-dir", default="output_slides", type=Path, help="슬라이드 디렉터리")
    p.add_argument("--lance-root", default=None, type=Path, help="LanceDB 루트 (기본: 레포 data/lancedb)")
    args = p.parse_args()
    try:
        from config import output_paths
    except ImportError:
        output_paths = None  # type: ignore

    if output_paths:
        paths = output_paths(args.stem, args.output_dir, args.slides_dir)
        fused = paths["fused"]
    else:
        fused = args.output_dir / f"{args.stem}_fused.json"

    lr = args.lance_root
    r = ingest_stem_to_lance(
        stem=args.stem,
        fused_path=fused,
        output_dir=args.output_dir,
        lance_root=lr,
    )
    print(r)
