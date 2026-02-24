"""
지식그래프 생성 파이프라인 (Stage 3)

Input:
  - integrated_text.json: t3 + text_vector
  - slide_extracted.json: image_vector

Output:
  - knowledge_graph.json: 개념, 관계, 벡터 통합
  - knowledge_graph.html: 시각화

파이프라인:
  1. t3 → 개념/관계 추출 (Gemini)
  2. text_vector + image_vector 병합
  3. 그래프 구축 및 시각화
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import google.generativeai as genai

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("⚠️ pyvis not installed. Visualization will be skipped.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

@dataclass
class Config:
    google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
    integrated_text_json: Path = Path("./output/integrated_text.json")
    slide_extracted_json: Path = Path("./output/slide_extracted.json")
    output_dir: Path = Path("./output")
    gemini_model: str = "models/gemini-2.5-flash"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# 12가지 관계 타입
RELATION_TYPES = {
    "is_a", "part_of", "implements", "abstracts",
    "prerequisite_of", "uses", "calls",
    "compared_to", "extends", "replaces",
    "solves", "optimizes"
}

#개념/관계 추출 프롬프트(개념 유형 명시 + 슬라이드 컨텍스트 포함)

EXTRACTION_PROMPT = """
아래 강의 슬라이드에서 핵심 개념과 관계를 추출하여 JSON으로 반환하라.
설명 없이 JSON만 출력.

[슬라이드 정보]
슬라이드 번호: {slide_num}
제목: {title}

[강의 내용]
{content}

출력 형식:
{{
  "concepts": ["개념1", "개념2", ...],
  "relations": [
    {{
      "from": "출발 개념",
      "to": "도착 개념",
      "type": "관계 타입",
      "evidence": "근거 문장"
    }}
  ]
}}

관계 타입 (12가지만 사용):
- is_a: A는 B의 한 종류
- part_of: A는 B의 구성요소
- implements: A는 B를 구현
- abstracts: A는 B들을 추상화
- prerequisite_of: A를 알아야 B 이해 가능
- uses: A는 B를 사용
- calls: A가 B를 호출
- compared_to: A와 B 비교
- extends: A가 B를 확장
- replaces: A가 B를 대체
- solves: A가 B(문제)를 해결
- optimizes: A가 B를 최적화

개념 추출 규칙:
- 슬라이드 제목과 번호를 반드시 참고하여 해당 슬라이드의 주제에 맞는 개념을 추출하라
- 다음 4가지 유형을 모두 포함하라:
  1. 정의/용어: 운영체제, 커널, 로더, 배치 운영체제 등 강의에서 정의하는 개념
  2. 역사적 사건/시스템: ENIAC, IBM 701, GM OS, GM-NAA I/O, EDVAC, CTSS 등 고유명사
  3. 기술적 메커니즘: 다중프로그래밍, 컨텍스트 스위칭, 인터럽트, 타임 슬라이스 등
  4. 문제/현상: 유휴 상태, 교착 상태, 메모리 보호, CPU 활용률 등
- 슬라이드당 최소 5개, 최대 15개 추출
- 개념은 명사 또는 명사구로 추출
- 동일 개념은 하나로 통일 (예: "시스템 호출", "system call" → "시스템 호출")
- 자기 자신과의 관계는 제외
- 너무 일반적이거나 강의 주제와 무관한 단어(예: "방법", "과정", "특징")는 제외
- 섹션 제목이나 목차 표현(예: '운영체제의 태동', '운영체제 종류')은 제외
- 강사가 예시로 언급한 구체적 소프트웨어(크롬, 탐색기 등)와 
  프로그래밍 키워드(malloc 등)도 개념으로 포함
"""


# ============================================================================ #
#  데이터 로더                                                                   #
# ============================================================================ #

class DataLoader:
    @staticmethod
    def load_integrated_text(path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        slides = data.get('slides', [])
        logger.info(f"✓ Loaded {len(slides)} slides (t3 + text_vector)")
        return slides

    @staticmethod
    def load_image_vectors(path: Path) -> Dict[str, List[float]]:
        """slide_id → image_vector 매핑"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vectors = {}
        for slide in data.get('slides', []):
            slide_id = slide.get('slide_id')
            if slide_id and slide.get('image_vector'):
                vectors[slide_id] = slide['image_vector']

        logger.info(f"✓ Loaded {len(vectors)} image vectors")
        return vectors


# ============================================================================ #
#  개념/관계 추출기                                                              #
# ============================================================================ #

class ConceptRelationExtractor:
    """t3에서 개념과 관계 추출"""

    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google_api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        logger.info(f"✓ Gemini initialized for extraction")

    def extract(self, slide: Dict) -> Dict:
        """단일 슬라이드에서 개념/관계 추출"""
        t3 = slide.get("t3", "")

        if not t3.strip():
            slide["concepts"] = []
            slide["relations"] = []
            return slide

        # 슬라이드 번호·제목 프롬프트에 포함
        slide_num = slide.get("slide_number", "?")
        title = slide.get("title", "제목 없음")

        try:
            prompt = EXTRACTION_PROMPT.format(
                slide_num=slide_num,
                title=title,
                content=t3
            )
            response = self.model.generate_content(prompt)
            text = response.text

            # JSON 파싱
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())

            # 관계 타입 검증 및 자기참조 제거
            relations = [
                r for r in result.get("relations", [])
                if r.get("type") in RELATION_TYPES
                and r.get("from") and r.get("to")
                and r.get("from") != r.get("to")
            ]

            slide["concepts"] = result.get("concepts", [])
            slide["relations"] = relations

        except Exception as e:
            logger.error(f"  ✗ Extraction failed for slide {slide_num}: {e}")
            slide["concepts"] = []
            slide["relations"] = []

        return slide

    def extract_batch(self, slides: List[Dict]) -> List[Dict]:
        logger.info(f"Extracting concepts/relations from {len(slides)} slides...")

        for i, slide in enumerate(slides):
            self.extract(slide)
            logger.info(
                f"  [{i+1}/{len(slides)}] Slide {slide.get('slide_number')} "
                f"「{slide.get('title', '')}」: "
                f"{len(slide['concepts'])} concepts, {len(slide['relations'])} relations"
            )

        logger.info(f"✓ Extraction complete")
        return slides


# ============================================================================ #
#  개념 정규화                                                                   #
# ============================================================================ #

class ConceptNormalizer:
    """개념 이름 정규화"""

    SYNONYMS = {
        "시스템 호출": ["system call", "시스템콜", "syscall"],
        "운영체제": ["operating system", "OS", "os"],
        "프로세스": ["process", "프로세서"],
        "커널": ["kernel", "커널 모드"],
        "메모리": ["memory", "RAM", "ram"],
    }

    def __init__(self):
        self.reverse_map = {}
        for canonical, variants in self.SYNONYMS.items():
            for v in variants:
                self.reverse_map[v.lower()] = canonical

    def normalize(self, concept: str) -> str:
        concept = concept.strip()

        # 괄호 내용 제거
        if '(' in concept:
            concept = concept.split('(')[0].strip()

        # 동의어 통일
        lower = concept.lower()
        if lower in self.reverse_map:
            return self.reverse_map[lower]

        return concept


# ============================================================================ #
#  그래프 빌더                                                                   #
# ============================================================================ #

class KnowledgeGraphBuilder:
    """통합 지식그래프 구축"""

    def __init__(self):
        self.normalizer = ConceptNormalizer()

    def build(
        self,
        slides: List[Dict],
        image_vectors: Dict[str, List[float]]
    ) -> Dict:
        """
        그래프 구축

        Returns:
            {
                "nodes": [
                    {
                        "id": "concept_name",
                        "type": "concept",
                        "slides": [slide_ids],
                        "text_vector": [...],   # 등장 슬라이드 text_vector 평균
                        "image_vector": [...]   # 등장 슬라이드 image_vector 평균
                    },
                    {
                        "id": "slide_001",
                        "type": "slide",
                        "title": "...",
                        "text_vector": [...],
                        "image_vector": [...]
                    }
                ],
                "edges": [
                    {"from": "A", "to": "B", "type": "uses", "weight": 1}
                ]
            }
        """
        concept_data = {}  # concept → {slides, text_vectors, image_vectors}
        edges = []
        edge_set = set()

        # 슬라이드별 처리
        for slide in slides:
            slide_id = slide["slide_id"]
            text_vector = slide.get("text_vector")
            image_vector = image_vectors.get(slide_id)

            # 개념 수집
            for concept in slide.get("concepts", []):
                normalized = self.normalizer.normalize(concept)
                if not normalized:
                    continue

                if normalized not in concept_data:
                    concept_data[normalized] = {
                        "slides": [],
                        "text_vectors": [],
                        "image_vectors": []
                    }

                concept_data[normalized]["slides"].append(slide_id)
                if text_vector:
                    concept_data[normalized]["text_vectors"].append(text_vector)
                if image_vector:
                    concept_data[normalized]["image_vectors"].append(image_vector)

            # 관계 수집
            for rel in slide.get("relations", []):
                from_concept = self.normalizer.normalize(rel.get("from", ""))
                to_concept = self.normalizer.normalize(rel.get("to", ""))
                rel_type = rel.get("type", "")

                if not from_concept or not to_concept or from_concept == to_concept:
                    continue

                edge_key = (from_concept, to_concept, rel_type)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append({
                        "from": from_concept,
                        "to": to_concept,
                        "type": rel_type,
                        "slide_id": slide_id,
                        "evidence": rel.get("evidence", "")
                    })

        # 노드 생성
        nodes = []

        # 개념 노드
        for concept, data in concept_data.items():
            node = {
                "id": concept,
                "type": "concept",
                "slides": list(set(data["slides"])),
                "frequency": len(data["slides"])
            }

            # 텍스트 벡터 평균
            if data["text_vectors"]:
                import numpy as np
                node["text_vector"] = np.mean(data["text_vectors"], axis=0).tolist()

            # 이미지 벡터 평균
            if data["image_vectors"]:
                import numpy as np
                node["image_vector"] = np.mean(data["image_vectors"], axis=0).tolist()

            nodes.append(node)

        # 슬라이드 노드
        for slide in slides:
            slide_id = slide["slide_id"]
            node = {
                "id": slide_id,
                "type": "slide",
                "slide_number": slide["slide_number"],
                "title": slide.get("title", ""),
                "timestamp": slide.get("timestamp", 0)
            }

            if slide.get("text_vector"):
                node["text_vector"] = slide["text_vector"]

            if image_vectors.get(slide_id):
                node["image_vector"] = image_vectors[slide_id]

            nodes.append(node)

        # 슬라이드 ↔ 개념 연결
        for concept, data in concept_data.items():
            for slide_id in set(data["slides"]):
                edges.append({
                    "from": slide_id,
                    "to": concept,
                    "type": "contains"
                })

        logger.info(f"✓ Built graph: {len(nodes)} nodes, {len(edges)} edges")

        return {
            "nodes": nodes,
            "edges": edges
        }


# ============================================================================ #
#  그래프 시각화                                                                 #
# ============================================================================ #

class GraphVisualizer:
    """PyVis 기반 그래프 시각화"""

    def visualize(self, graph: Dict, output_path: Path):
        if not PYVIS_AVAILABLE:
            logger.warning("pyvis not available, skipping visualization")
            return

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
        for node in graph["nodes"]:
            node_id = node["id"]
            node_type = node["type"]

            if node_type == "concept":
                size = 15 + node.get("frequency", 1) * 5
                net.add_node(
                    node_id,
                    label=node_id,
                    color="#4fc3f7",
                    size=size,
                    title=f"개념: {node_id}\n등장: {node.get('frequency', 1)}회"
                )
            else:  # slide
                net.add_node(
                    node_id,
                    label=f"Slide {node.get('slide_number', '?')}",
                    color="#ff8a65",
                    size=20,
                    shape="box",
                    title=f"{node.get('title', '')}\n{node_id}"
                )

        # 엣지 색상
        edge_colors = {
            "is_a": "#81c784",
            "part_of": "#64b5f6",
            "uses": "#ffb74d",
            "calls": "#ba68c8",
            "implements": "#4db6ac",
            "prerequisite_of": "#f06292",
            "contains": "#90a4ae",
            "compared_to": "#aed581",
            "extends": "#7986cb",
            "replaces": "#e57373",
            "solves": "#4dd0e1",
            "optimizes": "#dce775"
        }

        # 엣지 추가
        for edge in graph["edges"]:
            color = edge_colors.get(edge["type"], "#ffffff")
            net.add_edge(
                edge["from"],
                edge["to"],
                color=color,
                title=edge["type"],
                arrows="to"
            )

        net.save_graph(str(output_path))
        logger.info(f"✓ Saved visualization: {output_path}")


# ============================================================================ #
#  파이프라인                                                                    #
# ============================================================================ #

class GraphPipeline:
    """지식그래프 생성 파이프라인"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self) -> Dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🕸️ 지식그래프 생성 파이프라인 (Stage 3)")
        print("="*70)
        print(f"📄 Text: {self.config.integrated_text_json}")
        print(f"🖼️ Image: {self.config.slide_extracted_json}")
        print(f"📂 Output: {self.config.output_dir}")

        # Stage 1: 데이터 로드
        print("\n" + "-"*70)
        print("Stage 1: 데이터 로드")
        print("-"*70)

        slides = DataLoader.load_integrated_text(self.config.integrated_text_json)
        image_vectors = DataLoader.load_image_vectors(self.config.slide_extracted_json)

        # Stage 2: 개념/관계 추출
        print("\n" + "-"*70)
        print("Stage 2: 개념/관계 추출 (Gemini)")
        print("-"*70)

        extractor = ConceptRelationExtractor(self.config)
        slides = extractor.extract_batch(slides)

        # Stage 3: 그래프 구축
        print("\n" + "-"*70)
        print("Stage 3: 그래프 구축")
        print("-"*70)

        builder = KnowledgeGraphBuilder()
        graph = builder.build(slides, image_vectors)

        # Stage 4: 결과 저장
        print("\n" + "-"*70)
        print("Stage 4: 결과 저장")
        print("-"*70)

        # 통계
        concept_nodes = [n for n in graph["nodes"] if n["type"] == "concept"]
        slide_nodes = [n for n in graph["nodes"] if n["type"] == "slide"]
        concept_edges = [e for e in graph["edges"] if e["type"] != "contains"]

        result = {
            "metadata": {
                "processing_time": time.time() - start_time,
                "total_concepts": len(concept_nodes),
                "total_slides": len(slide_nodes),
                "total_relations": len(concept_edges)
            },
            "graph": graph,
            "slides": slides
        }

        # JSON 저장
        output_path = self.config.output_dir / "knowledge_graph.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path}")

        # 경량 버전 (벡터 제외)
        result_light = {
            "metadata": result["metadata"],
            "graph": {
                "nodes": [
                    {k: v for k, v in n.items() if k not in ["text_vector", "image_vector"]}
                    for n in graph["nodes"]
                ],
                "edges": graph["edges"]
            }
        }
        light_path = self.config.output_dir / "knowledge_graph_light.json"
        with open(light_path, 'w', encoding='utf-8') as f:
            json.dump(result_light, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved (light): {light_path}")

        # 시각화
        if PYVIS_AVAILABLE:
            visualizer = GraphVisualizer()
            viz_path = self.config.output_dir / "knowledge_graph.html"
            visualizer.visualize(graph, viz_path)

        # 완료 리포트
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ 그래프 생성 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 개념 노드: {len(concept_nodes)}개")
        print(f"  • 슬라이드 노드: {len(slide_nodes)}개")
        print(f"  • 관계 (개념↔개념): {len(concept_edges)}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"  • {light_path}")
        if PYVIS_AVAILABLE:
            print(f"  • {self.config.output_dir / 'knowledge_graph.html'}")
        print(f"\n⏱️ 처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="지식그래프 생성")
    parser.add_argument("-t", "--text", default="./output/integrated_text.json")
    parser.add_argument("-i", "--image", default="./output/slide_extracted.json")
    parser.add_argument("-o", "--output", default="./output")

    args = parser.parse_args()

    config = Config(
        integrated_text_json=Path(args.text),
        slide_extracted_json=Path(args.image),
        output_dir=Path(args.output)
    )

    if not config.integrated_text_json.exists():
        print(f"❌ Text JSON not found: {config.integrated_text_json}")
        return

    if not config.slide_extracted_json.exists():
        print(f"❌ Image JSON not found: {config.slide_extracted_json}")
        return

    pipeline = GraphPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()