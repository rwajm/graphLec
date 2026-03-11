"""
지식그래프 생성 파이프라인 (Stage 3)

Input:
  - integrated_text.json: t3 + text_vector (Stage 2 출력)
  - slide_extracted.json: t1_structure (Stage 1 출력)

Output:
  - knowledge_graph.json: 개념, 관계, 벡터 통합
  - knowledge_graph_light.json: 벡터 제외 경량 버전
  - knowledge_graph.html: 시각화

파이프라인:
  1. t3 + t1_structure → 개념/관계 추출 (Gemini)
  2. 관계 검증 (hallucination 필터)
  3. 그래프 구축 (관계 weight 누적)
  4. 시각화
"""

import os
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

# Stage 1, 2와 동일한 SDK 사용
from google import genai
from google.genai import types

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

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
    max_retries: int = 3
    retry_delay: float = 5.0
    # 동의어 사전 외부 파일 — 강의별로 교체 가능
    synonyms_path: Optional[Path] = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.synonyms_path:
            self.synonyms_path = Path(self.synonyms_path)


# 12가지 관계 타입
RELATION_TYPES = {
    "is_a", "part_of", "implements", "abstracts",
    "prerequisite_of", "uses", "calls",
    "compared_to", "extends", "replaces",
    "solves", "optimizes"
}

# t1_structure 섹션을 포함한 추출 프롬프트
# 슬라이드 구조 정보(다이어그램/표/화살표)를 관계 추출의 우선 근거로 활용
EXTRACTION_PROMPT = """
아래 강의 슬라이드에서 핵심 개념과 관계를 추출하여 JSON으로 반환하라.
설명 없이 JSON만 출력.

[슬라이드 정보]
슬라이드 번호: {slide_num}
제목: {title}

[강의 내용]
{content}

[슬라이드 구조]
{structure}

※ [슬라이드 구조]가 비어있지 않으면, 화살표/계층/비교표 등의 관계를 relations 추출의 우선 근거로 사용하라.
  예) "A → B → C (계층 구조)" → A part_of B, B part_of C 관계 추출
  예) "X vs Y 비교표" → X compared_to Y 관계 추출

출력 형식:
{{
  "concepts": ["개념1", "개념2", ...],
  "relations": [
    {{
      "from": "출발 개념",
      "to": "도착 개념",
      "type": "관계 타입",
      "evidence": "근거 키워드 (20자 이내, 특수문자/개행 금지)"
    }}
  ]
}}

관계 타입 (12가지만 사용):
방향 규칙: "from" → "to" 방향은 반드시 아래 정의를 따를 것.

[계층 관계 — 방향 혼동 주의]
- is_a:    from=하위 개념, to=상위 개념  (모드 레지스터 is_a CPU 구성요소 ❌ / 커널 코드 is_a 커널 ✓)
           "A는 B의 한 종류"이므로 A(from)가 더 구체적, B(to)가 더 일반적
- part_of: from=부분,      to=전체       (CPU part_of 모드 레지스터 ❌ / 모드 레지스터 part_of CPU ✓)
           "A는 B에 포함"이므로 A(from)가 구성요소, B(to)가 전체

[기능/의존 관계]
- implements:    from=구현체,   to=인터페이스/명세  (시스템 호출 implements API ✓)
- abstracts:     from=추상화,   to=구체 대상들      (프로세스 abstracts 프로그램 ✓)
- prerequisite_of: from=선수지식, to=목표 개념      (가상 메모리 prerequisite_of 페이징 ✓)
- uses:          from=사용 주체, to=사용 대상       (커널 uses 시스템 호출 ✓)
- calls:         from=호출자,    to=피호출자         (응용프로그램 calls 시스템 호출 ✓)
- compared_to:   from=비교 대상 A, to=비교 대상 B   (인터럽트 compared_to 폴링 ✓)
- extends:       from=확장체,   to=기반             (Linux extends Unix ✓)
- replaces:      from=새 것,    to=구 것            (syscall replaces int 0x80 ✓)
- solves:        from=해결책,   to=문제             (가상 메모리 solves 물리 메모리 부족 ✓)
- optimizes:     from=최적화 수단, to=최적화 대상   (캐시 optimizes 메모리 접근 ✓)

개념 추출 규칙:
- 슬라이드 제목과 번호를 반드시 참고하여 해당 슬라이드의 주제에 맞는 개념을 추출하라
- 다음 4가지 유형을 모두 포함하라:
  1. 정의/용어: 운영체제, 커널, 로더, 배치 운영체제 등 강의에서 정의하는 개념
  2. 역사적 사건/시스템: ENIAC, IBM 701, GM OS, GM-NAA I/O, EDVAC, CTSS 등 고유명사
  3. 기술적 메커니즘: 다중프로그래밍, 컨텍스트 스위칭, 인터럽트, 타임 슬라이스 등
  4. 문제/현상: 유휴 상태, 교착 상태, 메모리 보호, CPU 활용률 등
- 슬라이드당 최소 5개, 최대 15개 추출
- 개념은 명사 또는 명사구로 추출
- 동일 개념은 슬라이드 전체에서 하나의 표기로 통일할 것
  한글/영문 혼용 금지: fflush → fflush (영문 고유명사는 영문 유지)
  오타 금지: '응용프로gram' → '응용프로그램'
  문장형 개념명 금지: '높은 커널 모드 시간 비율' → '커널 모드 시간 비율'
  (형용사/부사로 시작하는 개념명은 핵심 명사구로 축약)
- 자기 자신과의 관계는 제외
- 너무 일반적이거나 강의 주제와 무관한 단어(예: "방법", "과정", "특징")는 제외
- 섹션 제목이나 목차 표현(예: '운영체제의 태동', '운영체제 종류')은 제외
- 강사가 예시로 언급한 구체적 소프트웨어(크롬, 탐색기 등)와
  프로그래밍 키워드(malloc 등)도 개념으로 포함
- 추출한 개념은 반드시 다른 개념과의 관계(relations)가 1개 이상 있어야 함
  관계를 정의할 수 없는 개념은 추출하지 말 것
- 성능 지표처럼 여러 개념이 묶이는 경우, 공통 상위 개념을 명시적으로 추출하고
  각각을 part_of로 연결할 것
  예) 시스템 처리율, 시스템 호출 횟수 → 둘 다 "시스템 성능 지표" part_of 관계

주의:
- evidence 값은 20자 이내의 짧은 한국어 키워드만 사용
- evidence에 콜론(:), 개행(\\n), 탭(\\t), 따옴표 등 특수문자 사용 금지
- 코드 스니펫이나 긴 문장을 evidence에 넣지 말 것
- relations의 from/to는 반드시 concepts 리스트 안에 있는 개념만 사용할 것
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
    def load_slide_structures(path: Path) -> Dict[str, str]:
        """slide_id → t1_structure 매핑 (Stage 1 출력에서 로드)"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        structures = {}
        for slide in data.get('slides', []):
            slide_id = slide.get('slide_id')
            structure = slide.get('t1_structure', '')
            if slide_id:
                structures[slide_id] = structure

        has_structure = sum(1 for v in structures.values() if v)
        logger.info(
            f"✓ Loaded {len(structures)} slide structures "
            f"({has_structure} with diagram/table info)"
        )
        return structures


# ============================================================================ #
#  개념/관계 추출기                                                              #
# ============================================================================ #

class ConceptRelationExtractor:
    """t3 + t1_structure에서 개념과 관계 추출"""

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.google_api_key)
        logger.info("✓ Gemini initialized for extraction")

    @staticmethod
    def _sanitize_response(text: str) -> str:
        """Gemini 응답에서 JSON 블록만 추출하고 기본 정제 수행"""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        def clean_string_value(m):
            inner = m.group(1)
            inner = re.sub(r'[\n\r\t]', ' ', inner)
            inner = re.sub(r' {2,}', ' ', inner).strip()
            return f'"{inner}"'

        text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, text)
        return text

    @staticmethod
    def _parse_json_robust(text: str) -> dict:
        """JSON 파싱 3단계: 표준 → json_repair → 정규식 fallback"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if JSON_REPAIR_AVAILABLE:
            try:
                repaired = repair_json(text)
                result = json.loads(repaired)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        concepts = []
        relations = []

        concepts_match = re.search(r'"concepts"\s*:\s*(\[.*?\])', text, re.DOTALL)
        if concepts_match:
            try:
                concepts = json.loads(concepts_match.group(1))
            except Exception:
                pass

        relations_match = re.search(r'"relations"\s*:\s*(\[.*?\])', text, re.DOTALL)
        if relations_match:
            try:
                relations = json.loads(relations_match.group(1))
            except Exception:
                pass

        if concepts:
            logger.warning("  ⚠ JSON repair fallback: extracted concepts only")
            return {"concepts": concepts, "relations": relations}

        raise ValueError("JSON parsing failed after all fallback attempts")

    def _call_gemini(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """재시도 로직 포함 Gemini 호출"""
        last_exc = None
        for attempt in range(self.config.max_retries):
            try:
                contents = [prompt, image] if image is not None else [prompt]
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents
                )
                return response.text
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"  ⚠ Gemini call failed "
                    f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        raise last_exc

    def extract(self, slide: Dict, structure: str = "") -> Dict:
        """단일 슬라이드에서 개념/관계 추출"""
        t3 = slide.get("t3", "")

        if not t3.strip():
            slide["concepts"] = []
            slide["relations"] = []
            return slide

        slide_num = slide.get("slide_number", "?")
        title = slide.get("title", "제목 없음")

        slide.setdefault("concepts", [])
        slide.setdefault("relations", [])

        try:
            prompt = EXTRACTION_PROMPT.format(
                slide_num=slide_num,
                title=title,
                content=t3,
                # t1_structure를 프롬프트에 주입 — 비어있으면 "(없음)"으로 표시
                structure=structure.strip() if structure.strip() else "(없음)"
            )

            image_path = slide.get("image_path")
            image = None
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")

            raw_text = self._call_gemini(prompt, image)
            cleaned = self._sanitize_response(raw_text)
            result = self._parse_json_robust(cleaned)

            concepts = result.get("concepts", [])
            concepts_set = set(c.strip() for c in concepts)  # hallucination 필터용

            # 1차 검증: 관계 타입, 자기참조, 빈 evidence 필터
            # 2차 검증: from/to가 이 슬라이드의 concepts 안에 없으면 제거
            # → LLM이 concepts에 없는 개념을 relation에서 만들어내는 hallucination 방지
            relations = [
                r for r in result.get("relations", [])
                if r.get("type") in RELATION_TYPES
                and r.get("from", "").strip()
                and r.get("to", "").strip()
                and r.get("from") != r.get("to")
                and r.get("evidence", "").strip()       # evidence 빈 문자열 관계 제거
                and r.get("from") in concepts_set       # concepts에 없는 from 제거
                and r.get("to") in concepts_set         # concepts에 없는 to 제거
            ]

            slide["concepts"] = concepts
            slide["relations"] = relations

        except Exception as e:
            logger.error(f"  ✗ Extraction failed for slide {slide_num}: {e}")

        return slide

    def extract_batch(
        self, slides: List[Dict], structures: Dict[str, str]
    ) -> List[Dict]:
        logger.info(f"Extracting concepts/relations from {len(slides)} slides...")

        for i, slide in enumerate(slides):
            slide_id = slide.get("slide_id", "")
            structure = structures.get(slide_id, "")
            self.extract(slide, structure)
            logger.info(
                f"  [{i+1}/{len(slides)}] Slide {slide.get('slide_number')} "
                f"「{slide.get('title', '')}」: "
                f"{len(slide['concepts'])} concepts, {len(slide['relations'])} relations"
            )

        logger.info("✓ Extraction complete")
        return slides


# ============================================================================ #
#  개념 정규화                                                                   #
# ============================================================================ #

class ConceptNormalizer:
    """개념 이름 정규화 — 동의어 사전은 외부 JSON 파일에서 로드"""

    # synonyms_path가 없을 때 사용하는 기본 사전
    DEFAULT_SYNONYMS: Dict[str, List[str]] = {
        "시스템 호출": ["system call", "시스템콜", "syscall"],
        "운영체제": ["operating system", "OS", "os"],
        "프로세스": ["process", "프로세서"],
        "커널": ["kernel", "커널 모드"],
        "메모리": ["memory", "RAM", "ram"],
    }

    def __init__(self, synonyms_path: Optional[Path] = None):
        synonyms = self.DEFAULT_SYNONYMS

        # 외부 파일이 있으면 덮어씀 — 강의별 사전으로 교체 가능
        if synonyms_path and Path(synonyms_path).exists():
            try:
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    synonyms = json.load(f)
                logger.info(f"✓ Loaded synonyms from {synonyms_path} ({len(synonyms)} entries)")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to load synonyms file: {e}. Using defaults.")
        else:
            logger.info(f"✓ Using default synonyms ({len(synonyms)} entries)")

        self.reverse_map: Dict[str, str] = {}
        for canonical, variants in synonyms.items():
            for v in variants:
                self.reverse_map[v.lower()] = canonical

    def normalize(self, concept: str) -> str:
        concept = concept.strip()

        if '(' in concept:
            concept = concept.split('(')[0].strip()

        lower = concept.lower()
        if lower in self.reverse_map:
            return self.reverse_map[lower]

        return concept


# ============================================================================ #
#  그래프 빌더                                                                   #
# ============================================================================ #

class KnowledgeGraphBuilder:
    """통합 지식그래프 구축"""

    def __init__(self, synonyms_path: Optional[Path] = None):
        self.normalizer = ConceptNormalizer(synonyms_path)

    def _ensure_concept(
        self,
        concept_data: dict,
        concept: str,
        slide_id: str,
        text_vector,
        image_vector
    ):
        if concept not in concept_data:
            concept_data[concept] = {
                "slides": [],
                "text_vectors": [],
                "image_vectors": []
            }
        concept_data[concept]["slides"].append(slide_id)
        if text_vector:
            concept_data[concept]["text_vectors"].append(text_vector)
        if image_vector:
            concept_data[concept]["image_vectors"].append(image_vector)

    def build(self, slides: List[Dict]) -> Dict:
        """
        그래프 구축

        Returns:
            {
                "nodes": [
                    {
                        "id": "concept_name",
                        "type": "concept",
                        "slide_ids": [slide_ids],  # 등장한 슬라이드 목록
                        "frequency": N,
                        "text_vector": [...]
                    },
                    {
                        "id": "slide_001",
                        "type": "slide",
                        "title": "...",
                        "timestamp": ...,
                        "has_audio": bool,         # Stage 2 신규 필드
                        "t2_coverage": N,          # Stage 2 신규 필드
                        "text_vector": [...]
                    }
                ],
                "edges": [
                    {
                        "from": "A", "to": "B", "type": "uses",
                        "weight": N,               # 등장 횟수 누적
                        "slide_ids": [...],        # 등장한 슬라이드 목록
                        "evidence": "..."          # 첫 등장 슬라이드의 evidence
                    }
                ]
            }
        """
        concept_data = {}
        # 관계 중복 제거 + weight 누적을 위한 맵
        # key: (from, to, type) → {weight, slide_ids, evidence}
        edge_map: Dict[Tuple, Dict] = {}

        for slide in slides:
            slide_id = slide["slide_id"]
            text_vector = slide.get("text_vector")

            # concepts 리스트 기반 노드 수집
            for concept in slide.get("concepts", []):
                normalized = self.normalizer.normalize(concept)
                if not normalized:
                    continue
                self._ensure_concept(concept_data, normalized, slide_id, text_vector, None)

            # 관계 수집 — weight 누적
            for rel in slide.get("relations", []):
                from_concept = self.normalizer.normalize(rel.get("from", ""))
                to_concept = self.normalizer.normalize(rel.get("to", ""))
                rel_type = rel.get("type", "")

                if not from_concept or not to_concept or from_concept == to_concept:
                    continue

                # 관계에 등장하는 개념도 노드로 보장
                self._ensure_concept(concept_data, from_concept, slide_id, text_vector, None)
                self._ensure_concept(concept_data, to_concept, slide_id, text_vector, None)

                edge_key = (from_concept, to_concept, rel_type)
                if edge_key not in edge_map:
                    # 첫 등장 시 evidence 기록
                    edge_map[edge_key] = {
                        "weight": 1,
                        "slide_ids": [slide_id],
                        "evidence": rel.get("evidence", "")
                    }
                else:
                    # 재등장 시 weight 누적, slide_ids 추가
                    edge_map[edge_key]["weight"] += 1
                    edge_map[edge_key]["slide_ids"].append(slide_id)

        # 노드 생성
        nodes = []

        # 개념 노드
        for concept, data in concept_data.items():
            node = {
                "id": concept,
                "type": "concept",
                "slide_ids": list(set(data["slides"])),
                "frequency": len(set(data["slides"]))  # 중복 슬라이드 제외한 실제 등장 슬라이드 수
            }
            if data["text_vectors"]:
                node["text_vector"] = np.mean(data["text_vectors"], axis=0).tolist()

            nodes.append(node)

        # 슬라이드 노드 — Stage 2 신규 필드 포함
        for slide in slides:
            slide_id = slide["slide_id"]
            node = {
                "id": slide_id,
                "type": "slide",
                "slide_number": slide["slide_number"],
                "title": slide.get("title", ""),
                "timestamp": slide.get("timestamp", 0),
                "timestamp_end": slide.get("timestamp_end", 0),
                # Stage 2에서 추가된 오디오 매칭 품질 필드
                "has_audio": slide.get("has_audio", False),
                "t2_coverage": slide.get("t2_coverage", 0),
            }
            if slide.get("text_vector"):
                node["text_vector"] = slide["text_vector"]

            nodes.append(node)

        # 엣지 생성
        edges = []
        for (from_c, to_c, rel_type), data in edge_map.items():
            edges.append({
                "from": from_c,
                "to": to_c,
                "type": rel_type,
                "weight": data["weight"],
                "slide_ids": data["slide_ids"],
                "evidence": data["evidence"]
            })

        # 슬라이드 ↔ 개념 contains 엣지
        for concept, data in concept_data.items():
            for slide_id in set(data["slides"]):
                edges.append({
                    "from": slide_id,
                    "to": concept,
                    "type": "contains",
                    "weight": 1,
                    "slide_ids": [slide_id],
                    "evidence": ""
                })

        logger.info(f"✓ Built graph: {len(nodes)} nodes, {len(edges)} edges")

        return {"nodes": nodes, "edges": edges}


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

        frequencies = [n.get("frequency", 1) for n in graph["nodes"] if n["type"] == "concept"]
        freq_min = min(frequencies) if frequencies else 1
        freq_max = max(frequencies) if frequencies else 1
        freq_range = freq_max - freq_min if freq_max != freq_min else 1

        for node in graph["nodes"]:
            node_id = node["id"]
            node_type = node["type"]

            if node_type == "concept":
                freq = node.get("frequency", 1)
                normalized = (freq - freq_min) / freq_range
                size = 10 + normalized * 30
                net.add_node(
                    node_id,
                    label=node_id,
                    color="#4fc3f7",
                    size=size,
                    title=f"개념: {node_id}\n등장: {node.get('frequency', 1)}회"
                )
            else:  # slide
                has_audio = node.get("has_audio", False)
                # 오디오 없는 슬라이드는 색상으로 구분
                color = "#ff8a65" if has_audio else "#b0bec5"
                net.add_node(
                    node_id,
                    label=f"Slide {node.get('slide_number', '?')}",
                    color=color,
                    size=20,
                    shape="box",
                    title=(
                        f"{node.get('title', '')}\n"
                        f"오디오: {'있음' if has_audio else '없음'} "
                        f"({node.get('t2_coverage', 0)}개 세그먼트)"
                    )
                )

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

        for edge in graph["edges"]:
            color = edge_colors.get(edge["type"], "#ffffff")
            # weight를 엣지 두께로 반영
            width = 1 + min(edge.get("weight", 1) - 1, 4)
            net.add_edge(
                edge["from"],
                edge["to"],
                color=color,
                title=f"{edge['type']} (×{edge.get('weight', 1)})",
                arrows="to",
                width=width
            )

        net.save_graph(str(output_path))
        logger.info(f"✓ Saved visualization: {output_path}")


# ============================================================================ #
#  파이프라인                                                                   #
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
        print(f"🖼️  Slide: {self.config.slide_extracted_json}")
        print(f"📂 Output: {self.config.output_dir}")
        if self.config.synonyms_path:
            print(f"📖 Synonyms: {self.config.synonyms_path}")

        # Stage 1: 데이터 로드
        print("\n" + "-"*70)
        print("Stage 1: 데이터 로드")
        print("-"*70)

        slides = DataLoader.load_integrated_text(self.config.integrated_text_json)
        structures = DataLoader.load_slide_structures(self.config.slide_extracted_json)

        # Stage 2: 개념/관계 추출
        print("\n" + "-"*70)
        print("Stage 2: 개념/관계 추출 (Gemini)")
        print("-"*70)

        extractor = ConceptRelationExtractor(self.config)
        slides = extractor.extract_batch(slides, structures)

        # Stage 3: 그래프 구축
        print("\n" + "-"*70)
        print("Stage 3: 그래프 구축")
        print("-"*70)

        builder = KnowledgeGraphBuilder(self.config.synonyms_path)
        graph = builder.build(slides)

        # Stage 4: 결과 저장
        print("\n" + "-"*70)
        print("Stage 4: 결과 저장")
        print("-"*70)

        concept_nodes = [n for n in graph["nodes"] if n["type"] == "concept"]
        slide_nodes = [n for n in graph["nodes"] if n["type"] == "slide"]
        concept_edges = [e for e in graph["edges"] if e["type"] != "contains"]
        audio_less = [n for n in slide_nodes if not n.get("has_audio", False)]

        result = {
            "metadata": {
                "processing_time": time.time() - start_time,
                "total_concepts": len(concept_nodes),
                "total_slides": len(slide_nodes),
                "total_relations": len(concept_edges),
                "slides_without_audio": len(audio_less),
            },
            "graph": graph,
            "slides": slides
        }

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

        if PYVIS_AVAILABLE:
            visualizer = GraphVisualizer()
            viz_path = self.config.output_dir / "knowledge_graph.html"
            visualizer.visualize(graph, viz_path)

        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ 그래프 생성 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 개념 노드: {len(concept_nodes)}개")
        print(f"  • 슬라이드 노드: {len(slide_nodes)}개")
        print(f"  • 관계 (개념↔개념): {len(concept_edges)}개")
        print(f"  • 오디오 없는 슬라이드: {len(audio_less)}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"  • {light_path}")
        if PYVIS_AVAILABLE:
            print(f"  • {self.config.output_dir / 'knowledge_graph.html'}")
        print(f"\n⏱️  처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="지식그래프 생성")
    parser.add_argument("-t", "--text", default="./output/integrated_text.json")
    parser.add_argument("-s", "--slide", default="./output/slide_extracted.json")
    parser.add_argument("-o", "--output", default="./output")
    parser.add_argument(
        "--synonyms", default=None,
        help="동의어 사전 JSON 경로 (없으면 기본 사전 사용)"
    )

    args = parser.parse_args()

    config = Config(
        integrated_text_json=Path(args.text),
        slide_extracted_json=Path(args.slide),
        output_dir=Path(args.output),
        synonyms_path=Path(args.synonyms) if args.synonyms else None,
    )

    if not config.integrated_text_json.exists():
        print(f"❌ Text JSON not found: {config.integrated_text_json}")
        return

    if not config.slide_extracted_json.exists():
        print(f"❌ Slide JSON not found: {config.slide_extracted_json}")
        return

    pipeline = GraphPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()