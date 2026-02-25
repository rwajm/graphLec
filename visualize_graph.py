"""
지식그래프 시각화 (독립 실행)

Input:
  - knowledge_graph.json (또는 knowledge_graph_light.json)

Output:
  - knowledge_graph.html

Usage:
  python visualize_graph.py -g ./output/knowledge_graph.json
  python visualize_graph.py -g ./output/knowledge_graph.json -o ./output/my_graph.html
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


EDGE_COLORS = {
    "is_a":           "#81c784",
    "part_of":        "#64b5f6",
    "uses":           "#ffb74d",
    "calls":          "#ba68c8",
    "implements":     "#4db6ac",
    "prerequisite_of":"#f06292",
    "contains":       "#90a4ae",
    "compared_to":    "#aed581",
    "extends":        "#7986cb",
    "replaces":       "#e57373",
    "solves":         "#4dd0e1",
    "optimizes":      "#dce775",
}

# 노드 크기 범위
NODE_SIZE_MIN = 10
NODE_SIZE_MAX = 40
NODE_SIZE_SLIDE = 20


def visualize(graph: Dict, output_path: Path):
    if not PYVIS_AVAILABLE:
        logger.error("pyvis가 설치되지 않았습니다. pip install pyvis")
        return

    nodes = graph["nodes"]
    edges = graph["edges"]

    # min-max 정규화를 위한 frequency 범위 계산
    frequencies = [n.get("frequency", 1) for n in nodes if n["type"] == "concept"]
    freq_min = min(frequencies) if frequencies else 1
    freq_max = max(frequencies) if frequencies else 1
    freq_range = freq_max - freq_min if freq_max != freq_min else 1

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200
    )

    # 노드 추가
    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]

        if node_type == "concept":
            freq = node.get("frequency", 1)
            normalized = (freq - freq_min) / freq_range  # 0~1
            size = NODE_SIZE_MIN + normalized * (NODE_SIZE_MAX - NODE_SIZE_MIN)
            net.add_node(
                node_id,
                label=node_id,
                color="#4fc3f7",
                size=size,
                title=f"개념: {node_id}\n등장: {freq}회"
            )
        else:  # slide
            net.add_node(
                node_id,
                label=f"Slide {node.get('slide_number', '?')}",
                color="#ff8a65",
                size=NODE_SIZE_SLIDE,
                shape="box",
                title=f"{node.get('title', '')}\n{node_id}"
            )

    # 엣지 추가
    for edge in edges:
        color = EDGE_COLORS.get(edge["type"], "#ffffff")
        net.add_edge(
            edge["from"],
            edge["to"],
            color=color,
            title=edge["type"],
            arrows="to"
        )

    net.save_graph(str(output_path))
    logger.info(f"✓ 시각화 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="지식그래프 시각화")
    parser.add_argument("-g", "--graph", default="./output/knowledge_graph.json", help="knowledge_graph.json 경로")
    parser.add_argument("-o", "--output", default=None, help="출력 HTML 경로 (기본: 그래프와 같은 폴더)")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"❌ 파일 없음: {graph_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    graph = data.get("graph", {})
    if not graph:
        print("❌ 'graph' 키가 없습니다. knowledge_graph.json을 확인하세요.")
        return

    output_path = Path(args.output) if args.output else graph_path.parent / "knowledge_graph.html"
    visualize(graph, output_path)


if __name__ == "__main__":
    main()