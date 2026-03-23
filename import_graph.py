"""
import_graph.py — graph_triples.csv를 Neo4j에 적재

전략:
  - predicate == 'type' → 노드 생성 (MERGE)
  - predicate != 'type' → 관계 생성 (MERGE)
  - --reset 옵션: 기존 그래프 삭제 후 전체 재적재

사용법:
  python import_graph.py --stem os1-1
  python import_graph.py --stem os1-1 --reset
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase

# ── Neo4j 연결 설정 ──
URI      = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "rhlh1234"
# ────────────────────


def resolve_csv_path(stem: str, output_dir: Path, slides_dir: Path) -> Path:
    try:
        from config import output_paths
        paths = output_paths(stem, output_dir, slides_dir)
        # graph_triples는 config에 없으므로 output_dir 기반으로 생성
        return output_dir / f"{stem}_graph_triples.csv"
    except ImportError:
        return output_dir / f"{stem}_graph_triples.csv"


def load_csv(path: Path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def flatten_props(props: dict) -> dict:
    """Neo4j는 MAP 타입 프로퍼티 불가 → dict/list는 JSON 문자열로 변환."""
    return {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for k, v in props.items()}


def import_graph(csv_path: Path, reset: bool = False):
    rows = load_csv(csv_path)
    print(f"CSV 로드: {len(rows)}개 트리플")

    type_rows = [r for r in rows if r['predicate'] == 'type']
    rel_rows  = [r for r in rows if r['predicate'] != 'type']

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    with driver.session() as session:

        if reset:
            print("기존 그래프 삭제 중...")
            session.run("MATCH (n) DETACH DELETE n")
            print("삭제 완료")

        # ── 노드 생성 ────────────────────────────────────────────────────────
        print(f"노드 생성 중... ({len(type_rows)}개)")
        by_label = defaultdict(list)
        for r in type_rows:
            props = json.loads(r['properties']) if r['properties'].strip() else {}
            props['id'] = r['subject']
            by_label[r['object']].append(flatten_props(props))

        node_count = 0
        for label, props_list in by_label.items():
            result = session.run(
                f"""
                UNWIND $props_list AS props
                MERGE (n:{label} {{id: props.id}})
                SET n += props
                RETURN count(n) AS cnt
                """,
                props_list=props_list,
            )
            cnt = result.single()['cnt']
            node_count += cnt
            print(f"  {label}: {cnt}개")
        print(f"노드 총 {node_count}개 완료")

        # ── 관계 생성 ────────────────────────────────────────────────────────
        print(f"\n관계 생성 중... ({len(rel_rows)}개)")
        by_rel = defaultdict(list)
        for r in rel_rows:
            props = json.loads(r['properties']) if r['properties'].strip() else {}
            by_rel[r['predicate']].append({
                'src': r['subject'],
                'tgt': r['object'],
                'props': flatten_props(props),
            })

        rel_count = 0
        for rtype, items in by_rel.items():
            result = session.run(
                f"""
                UNWIND $items AS item
                MATCH (a {{id: item.src}}), (b {{id: item.tgt}})
                MERGE (a)-[r:{rtype}]->(b)
                SET r += item.props
                RETURN count(r) AS cnt
                """,
                items=items,
            )
            cnt = result.single()['cnt']
            rel_count += cnt
            print(f"  {rtype}: {cnt}개")
        print(f"관계 총 {rel_count}개 완료")

        # ── 인덱스 생성 ──────────────────────────────────────────────────────
        print("\n인덱스 생성 중...")
        for label in by_label:
            try:
                session.run(
                    f"CREATE INDEX {label}_id IF NOT EXISTS FOR (n:{label}) ON (n.id)"
                )
            except Exception:
                pass
        print("인덱스 완료")

    driver.close()
    print(f"\n적재 완료: 노드 {node_count}개, 관계 {rel_count}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="graph_triples.csv → Neo4j 적재")
    parser.add_argument("--stem",       required=True,           help="강의 파일 stem (예: os1-1)")
    parser.add_argument("--output_dir", default="output",        help="출력 디렉토리 (기본: output)")
    parser.add_argument("--slides_dir", default="output_slides", help="슬라이드 디렉토리")
    parser.add_argument("--reset",      action="store_true",     help="기존 그래프 삭제 후 재적재")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.stem, Path(args.output_dir), Path(args.slides_dir))

    if not csv_path.exists():
        print(f"❌ CSV 파일 없음: {csv_path}")
        exit(1)

    import_graph(csv_path, reset=args.reset)