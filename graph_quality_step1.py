"""
지식그래프 구조적 건전성 검증 스크립트 (Step 1)

Input:
  - knowledge_graph.json

측정 항목:
  1. 고립 노드 비율
  2. 슬라이드 커버리지 + 흐름(Flow) 시각화
  3. 관계 타입 분포 (편향도)
  4. 슬라이드-개념 연결 밀도
  5. 엔티티 중복 후보 탐지 (편집거리 + 임베딩 벡터)
  6. 그래프 위상 구조 (연결 성분, Diameter)
  7. 논리적 정합성 (자기참조, 중복 엣지, 계층 순환)

Usage:
  python graph_quality_step1.py -g ./output/knowledge_graph.json
  python graph_quality_step1.py -g ./output/knowledge_graph.json -o ./output/step1_quality_report.json
  python graph_quality_step1.py -g ./output/knowledge_graph.json --sim-threshold 0.8 --vec-threshold 0.92
"""

import json
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ============================================================================ #
#  유틸리티                                                                      #
# ============================================================================ #

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), curr[j] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def lex_similarity(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / max_len


def cosine_similarity(a: List[float], b: List[float]) -> Optional[float]:
    try:
        import numpy as np
        a, b = np.array(a), np.array(b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except ImportError:
        return None  # numpy 없으면 스킵


# ============================================================================ #
#  1. 고립 노드 비율                                                              #
# ============================================================================ #

def check_isolated_nodes(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """개념↔개념 엣지 기준 연결되지 않은 개념 노드"""
    concept_nodes = {n["id"] for n in nodes if n["type"] == "concept"}
    connected = set()
    for edge in edges:
        if edge["type"] != "contains":
            connected.add(edge["from"])
            connected.add(edge["to"])
    connected &= concept_nodes
    isolated = concept_nodes - connected
    total = len(concept_nodes)

    return {
        "total_concept_nodes": total,
        "isolated_count": len(isolated),
        "isolated_ratio": round(len(isolated) / total, 4) if total else 0.0,
        "isolated_nodes": sorted(isolated)
    }


# ============================================================================ #
#  2. 슬라이드 커버리지 + 흐름 시각화                                              #
# ============================================================================ #

def check_slide_coverage(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """슬라이드 커버리지 + 연속 미커버 구간 탐지"""
    slide_nodes = {n["id"]: n for n in nodes if n["type"] == "slide"}

    covered = set()
    for edge in edges:
        if edge["type"] == "contains" and edge["from"] in slide_nodes:
            covered.add(edge["from"])

    uncovered = set(slide_nodes.keys()) - covered
    total = len(slide_nodes)

    uncovered_info = []
    for sid in sorted(uncovered):
        n = slide_nodes[sid]
        uncovered_info.append({
            "slide_id": sid,
            "slide_number": n.get("slide_number"),
            "title": n.get("title", "")
        })

    # 슬라이드 번호 순서로 흐름 바 생성 (■커버, □미커버)
    all_slides_sorted = sorted(
        slide_nodes.values(),
        key=lambda x: x.get("slide_number", 0)
    )
    flow_bar = "".join("■" if s["id"] in covered else "□" for s in all_slides_sorted)

    # 연속 미커버 구간 탐지
    gap_runs = []
    run_start = None
    for i, s in enumerate(all_slides_sorted):
        if s["id"] not in covered:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                gap_runs.append({
                    "start_slide": all_slides_sorted[run_start].get("slide_number"),
                    "end_slide": all_slides_sorted[i - 1].get("slide_number"),
                    "length": i - run_start
                })
                run_start = None
    if run_start is not None:
        gap_runs.append({
            "start_slide": all_slides_sorted[run_start].get("slide_number"),
            "end_slide": all_slides_sorted[-1].get("slide_number"),
            "length": len(all_slides_sorted) - run_start
        })

    return {
        "total_slides": total,
        "covered_count": len(covered),
        "uncovered_count": len(uncovered),
        "coverage_ratio": round(len(covered) / total, 4) if total else 0.0,
        "flow_bar": flow_bar,
        "consecutive_gap_runs": gap_runs,
        "uncovered_slides": uncovered_info
    }


# ============================================================================ #
#  3. 관계 타입 분포                                                              #
# ============================================================================ #

RELATION_TYPES = {
    "is_a", "part_of", "implements", "abstracts",
    "prerequisite_of", "uses", "calls",
    "compared_to", "extends", "replaces",
    "solves", "optimizes"
}

def check_relation_type_distribution(edges: List[Dict]) -> Dict:
    concept_edges = [e for e in edges if e["type"] != "contains"]
    total = len(concept_edges)
    counts = Counter(e["type"] for e in concept_edges)

    distribution = {}
    for rtype in RELATION_TYPES:
        cnt = counts.get(rtype, 0)
        distribution[rtype] = {
            "count": cnt,
            "ratio": round(cnt / total, 4) if total else 0.0
        }
    for rtype, cnt in counts.items():
        if rtype not in RELATION_TYPES:
            distribution[f"[unknown] {rtype}"] = {
                "count": cnt,
                "ratio": round(cnt / total, 4) if total else 0.0
            }

    sorted_dist = sorted(distribution.items(), key=lambda x: x[1]["count"], reverse=True)
    top_type, top_data = sorted_dist[0] if sorted_dist else ("none", {"count": 0, "ratio": 0.0})
    used_types = sum(1 for v in distribution.values() if v["count"] > 0)

    return {
        "total_concept_edges": total,
        "used_type_count": used_types,
        "total_type_count": len(RELATION_TYPES),
        "is_biased": top_data["ratio"] >= 0.5,
        "top_type": top_type,
        "top_type_ratio": top_data["ratio"],
        "distribution": dict(sorted_dist)
    }


# ============================================================================ #
#  4. 슬라이드-개념 연결 밀도                                                      #
# ============================================================================ #

def check_slide_concept_density(nodes: List[Dict], edges: List[Dict]) -> Dict:
    slide_ids = {n["id"] for n in nodes if n["type"] == "slide"}
    concept_per_slide = {sid: 0 for sid in slide_ids}

    for edge in edges:
        if edge["type"] == "contains" and edge["from"] in slide_ids:
            concept_per_slide[edge["from"]] += 1

    counts = list(concept_per_slide.values())
    total_slides = len(counts)
    if total_slides == 0:
        return {"error": "슬라이드 노드 없음"}

    avg = sum(counts) / total_slides
    buckets = {"0": 0, "1-4": 0, "5-9": 0, "10-14": 0, "15+": 0}
    for c in counts:
        if c == 0:       buckets["0"] += 1
        elif c <= 4:     buckets["1-4"] += 1
        elif c <= 9:     buckets["5-9"] += 1
        elif c <= 14:    buckets["10-14"] += 1
        else:            buckets["15+"] += 1

    details = sorted(
        [{"slide_id": sid, "concept_count": cnt} for sid, cnt in concept_per_slide.items()],
        key=lambda x: x["concept_count"]
    )

    return {
        "total_slides": total_slides,
        "avg_concepts_per_slide": round(avg, 2),
        "min_concepts": min(counts),
        "max_concepts": max(counts),
        "zero_concept_slides": buckets["0"],
        "distribution_buckets": buckets,
        "per_slide_detail": details
    }


# ============================================================================ #
#  5. 엔티티 중복 후보 (편집거리 + 임베딩 벡터)                                    #
# ============================================================================ #

# ============================================================================ #
#  API 사용량 추적                                                                #
# ============================================================================ #

class UsageTracker:
    """임베딩/LLM 호출의 시간 및 토큰 사용량 누적 추적"""
    def __init__(self):
        self.embed_calls    = 0
        self.embed_tokens   = 0      # 입력 토큰 (문자 수 기반 추정 포함)
        self.llm_calls      = 0
        self.llm_input_tokens  = 0
        self.llm_output_tokens = 0
        self.elapsed_sec    = 0.0

    def add_embed(self, text: str, resp=None):
        self.embed_calls += 1
        if resp and hasattr(resp, "usage_metadata") and resp.usage_metadata:
            self.embed_tokens += getattr(resp.usage_metadata, "total_token_count", 0)
        else:
            # API가 토큰 수를 반환하지 않을 때 문자 수로 추정 (1토큰 ≈ 2~3자 한국어)
            self.embed_tokens += max(1, len(text) // 2)

    def add_llm(self, resp=None):
        self.llm_calls += 1
        if resp and hasattr(resp, "usage_metadata") and resp.usage_metadata:
            m = resp.usage_metadata
            self.llm_input_tokens  += getattr(m, "prompt_token_count", 0)
            self.llm_output_tokens += getattr(m, "candidates_token_count", 0)

    def summary(self) -> Dict:
        return {
            "elapsed_sec":        round(self.elapsed_sec, 2),
            "embed_calls":        self.embed_calls,
            "embed_tokens_est":   self.embed_tokens,
            "llm_calls":          self.llm_calls,
            "llm_input_tokens":   self.llm_input_tokens,
            "llm_output_tokens":  self.llm_output_tokens,
            "llm_total_tokens":   self.llm_input_tokens + self.llm_output_tokens,
        }


_usage = UsageTracker()   # 전역 추적기


def _embed_concept_names(
    concept_ids: List[str],
    api_key: str,
    embedding_model: str = "models/gemini-embedding-001",
    embedding_dim: int = 768,
    batch_size: int = 50
) -> Dict[str, List[float]]:
    """
    개념 이름 문자열 자체를 임베딩.
    슬라이드 맥락 벡터(text_vector)와 별개로, 이름의 의미를 직접 비교하기 위함.
    예) 'AI'와 '인공지능'은 슬라이드가 달라도 이름 임베딩 유사도는 높게 나옴.
    """
    if not GENAI_AVAILABLE:
        return {}

    client = genai.Client(api_key=api_key)
    result = {}

    for i in range(0, len(concept_ids), batch_size):
        batch = concept_ids[i:i + batch_size]
        for cid in batch:
            try:
                resp = client.models.embed_content(
                    model=embedding_model,
                    contents=cid,
                    config=types.EmbedContentConfig(output_dimensionality=embedding_dim)
                )
                result[cid] = resp.embeddings[0].values
                _usage.add_embed(cid, resp)
            except Exception:
                pass  # 실패한 개념은 스킵
        time.sleep(0.2)  # rate limit 방지

    return result


def run_llm_judge(
    candidates: List[Dict],
    concept_node_map: Dict[str, Dict],  # concept_id → node (slides 정보 포함)
    api_key: str,
    domain_hint: str = "",
    model: str = "gemini-2.5-flash",
    max_judge: int = 30
) -> List[Dict]:
    """
    임베딩 후보 상위 N쌍을 Gemini에 넘겨 same/different 최종 판정.
    - 슬라이드 등장 컨텍스트 제공으로 문맥적 동일성 판단 강화
    - domain_hint로 도메인 특화 약어(레지스터명 등) 오탐 방지
    - reasoning 요청으로 사용자 최종 결정 근거 제공
    """
    if not GENAI_AVAILABLE or not candidates:
        return []

    client = genai.Client(api_key=api_key)
    targets = candidates[:max_judge]

    # 슬라이드 컨텍스트 포함한 쌍 목록 구성
    pairs_text_parts = []
    for i, c in enumerate(targets):
        a_slides = concept_node_map.get(c["concept_a"], {}).get("slides", [])
        b_slides = concept_node_map.get(c["concept_b"], {}).get("slides", [])
        a_ctx = f"슬라이드 {', '.join(str(s) for s in sorted(a_slides)[:3])}" if a_slides else "슬라이드 정보 없음"
        b_ctx = f"슬라이드 {', '.join(str(s) for s in sorted(b_slides)[:3])}" if b_slides else "슬라이드 정보 없음"
        pairs_text_parts.append(
            f"{i+1}. '{c['concept_a']}' (등장: {a_ctx})  vs  '{c['concept_b']}' (등장: {b_ctx})"
        )
    pairs_text = "\n".join(pairs_text_parts)

    domain_section = f"\n도메인 힌트: {domain_hint}" if domain_hint else ""

    prompt = f"""다음은 컴퓨터과학 강의 지식그래프에서 추출된 개념 쌍입니다.
각 개념이 등장한 슬라이드 번호를 참고하여, 두 개념이 동일한 의미인지(same) 다른 의미인지(different) 판단하세요.{domain_section}

{pairs_text}

판단 기준:
- same: 약어=원어 (AI=인공지능), 번역어 (kernel=커널), 표기 변형 (시스템호출=시스템 호출)
- different: 상위-하위 관계 (운영체제 ≠ 커널), 연관되지만 별개 개념
- uncertain: 문맥 없이 판단 불가
- CPU 레지스터(rax, rbx, rsp 등), 어셈블리 명령어처럼 한 글자가 완전히 다른 자원을 가리키는 경우는 반드시 different

반드시 아래 JSON 형식으로만 출력하세요:
[
  {{"index": 1, "verdict": "same", "reason": "영문 약어와 한글 원어 관계"}},
  {{"index": 2, "verdict": "different", "reason": "상위-하위 개념, 등장 슬라이드도 다름"}},
  ...
]"""

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        _usage.add_llm(resp)

        text = resp.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        verdicts = json.loads(text.strip())
        verdict_map = {v["index"]: v for v in verdicts}

        judged = []
        for i, c in enumerate(targets):
            v = verdict_map.get(i + 1, {})
            judged.append({
                **c,
                "verdict": v.get("verdict", "unknown"),
                "reason":  v.get("reason", "")
            })
        return judged

    except Exception as e:
        return [{**c, "verdict": "error", "reason": str(e)} for c in targets]


def check_entity_duplicates(
    nodes: List[Dict],
    lex_threshold: float = 0.80,
    vec_threshold: float = 0.80,
    max_candidates: int = 50,
    api_key: str = "",
    use_llm_judge: bool = False,
    domain_hint: str = ""
) -> Dict:
    """
    3단계 중복 탐지:
      1. Semantic — 개념명 직접 임베딩 코사인 유사도 ≥ vec_threshold (1차 필터)
      2. Lexical  — 길이 >5자 쌍, 편집거리 유사도 ≥ lex_threshold (2차 필터, 추가)
      3. LLM Judge — 위 후보를 Gemini로 same/different 최종 판정 (선택)
    """
    t_start = time.time()
    concept_nodes = [n for n in nodes if n["type"] == "concept"]
    n = len(concept_nodes)
    concept_ids = [c["id"] for c in concept_nodes]
    # concept_id → node 매핑 (LLM Judge 컨텍스트용)
    concept_node_map = {c["id"]: c for c in concept_nodes}

    # --- Semantic: 개념명 직접 임베딩 ---
    name_vectors: Dict[str, List[float]] = {}
    if api_key and GENAI_AVAILABLE:
        print(f"  개념명 임베딩 중... ({n}개)")
        name_vectors = _embed_concept_names(concept_ids, api_key)
    else:
        # fallback: 기존 text_vector 사용 (슬라이드 맥락 벡터)
        name_vectors = {c["id"]: c["text_vector"] for c in concept_nodes if c.get("text_vector")}

    lex_candidates = []
    vec_candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            a_id, b_id = concept_ids[i], concept_ids[j]

            # 1차: Semantic
            a_vec = name_vectors.get(a_id)
            b_vec = name_vectors.get(b_id)
            if a_vec and b_vec:
                sim = cosine_similarity(a_vec, b_vec)
                if sim is not None and sim >= vec_threshold:
                    vec_candidates.append({
                        "concept_a": a_id,
                        "concept_b": b_id,
                        "similarity": round(sim, 4),
                        "method": "semantic"
                    })

            # 2차: Lexical (두 개념 모두 길이 >5자일 때만)
            if len(a_id) > 5 and len(b_id) > 5:
                len_ratio = abs(len(a_id) - len(b_id)) / max(len(a_id), len(b_id), 1)
                if len_ratio <= 0.5:
                    lex_sim = lex_similarity(a_id, b_id)
                    if lex_sim >= lex_threshold:
                        lex_candidates.append({
                            "concept_a": a_id,
                            "concept_b": b_id,
                            "similarity": round(lex_sim, 4),
                            "method": "lexical"
                        })

    vec_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    lex_candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # --- LLM Judge ---
    llm_results = []
    if use_llm_judge and api_key:
        # semantic 후보 우선, 없으면 lexical로 폴백
        judge_pool = vec_candidates if vec_candidates else lex_candidates
        if judge_pool:
            print(f"  LLM Judge 실행 중... ({min(30, len(judge_pool))}쌍)")
            llm_results = run_llm_judge(
                judge_pool, concept_node_map, api_key,
                domain_hint=domain_hint
            )

    using_name_embed = bool(api_key and GENAI_AVAILABLE and name_vectors)
    elapsed = time.time() - t_start
    _usage.elapsed_sec += elapsed

    return {
        "total_concept_nodes": n,
        "lex_threshold": lex_threshold,
        "vec_threshold": vec_threshold,
        "using_name_embedding": using_name_embed,
        "elapsed_sec": round(elapsed, 2),
        "lexical_candidate_count": len(lex_candidates),
        "semantic_candidate_count": len(vec_candidates),
        "lexical_candidates": lex_candidates[:max_candidates],
        "semantic_candidates": vec_candidates[:max_candidates],
        "llm_judge_count": len(llm_results),
        "llm_judge_results": llm_results,
        "merge_suggestions": [r for r in llm_results if r.get("verdict") == "same"]
    }


# ============================================================================ #
#  6. 그래프 위상 구조 (연결 성분, Diameter)                                       #
# ============================================================================ #

def _build_adjacency_undirected(concept_ids: Set[str], edges: List[Dict]) -> Dict[str, Set[str]]:
    adj = defaultdict(set)
    for edge in edges:
        if edge["type"] == "contains":
            continue
        f, t = edge["from"], edge["to"]
        if f in concept_ids and t in concept_ids:
            adj[f].add(t)
            adj[t].add(f)
    return adj


def _bfs_farthest(start: str, adj: Dict[str, Set[str]], component: Set[str]) -> Tuple[int, str]:
    """BFS로 start에서 가장 먼 노드와 거리 반환"""
    visited = {start: 0}
    queue = [start]
    far_node, far_dist = start, 0
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited and nb in component:
                visited[nb] = visited[cur] + 1
                queue.append(nb)
                if visited[nb] > far_dist:
                    far_dist = visited[nb]
                    far_node = nb
    return far_dist, far_node


def check_topology(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """
    연결 성분 개수, 최대 컴포넌트 Diameter (BFS 2회 근사),
    소규모 고립 섬 목록
    """
    concept_ids = {n["id"] for n in nodes if n["type"] == "concept"}
    adj = _build_adjacency_undirected(concept_ids, edges)

    visited = {}
    comp_id = 0
    components: Dict[int, Set[str]] = {}

    def bfs_component(start):
        comp = set()
        queue = [start]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited[cur] = comp_id
            comp.add(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    queue.append(nb)
        return comp

    for cid in concept_ids:
        if cid not in visited:
            comp = bfs_component(cid)
            components[comp_id] = comp
            comp_id += 1

    sizes = sorted([len(c) for c in components.values()], reverse=True)
    largest_comp = max(components.values(), key=len)

    # Diameter: 최대 컴포넌트만, BFS 2회
    diameter = None
    diameter_endpoints = None
    if len(largest_comp) > 1:
        start_node = next(iter(largest_comp))
        _, far1 = _bfs_farthest(start_node, adj, largest_comp)
        d, far2 = _bfs_farthest(far1, adj, largest_comp)
        diameter = d
        diameter_endpoints = [far1, far2]

    small_islands = [
        sorted(comp) for comp in components.values() if len(comp) <= 3
    ]

    return {
        "total_concept_nodes": len(concept_ids),
        "component_count": len(components),
        "component_size_distribution": sizes,
        "largest_component_size": sizes[0] if sizes else 0,
        "largest_component_diameter": diameter,
        "diameter_endpoints": diameter_endpoints,
        "small_islands_count": len(small_islands),
        "small_islands": small_islands[:20]
    }


# ============================================================================ #
#  7. 논리적 정합성                                                               #
# ============================================================================ #

def check_logical_consistency(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """
    - 자기 참조(self-loop) 탐지
    - 중복 엣지 탐지 (동일 from-to-type)
    - 계층 순환 탐지 (is_a, part_of 대상 DFS)
    """
    # 자기 참조
    self_loops = [
        {"from": e["from"], "to": e["to"], "type": e["type"]}
        for e in edges if e["from"] == e["to"]
    ]

    # 중복 엣지
    edge_counter = Counter(
        (e["from"], e["to"], e["type"]) for e in edges if e["type"] != "contains"
    )
    duplicate_edges = [
        {"from": k[0], "to": k[1], "type": k[2], "count": v}
        for k, v in edge_counter.items() if v > 1
    ]

    # 계층 순환 탐지 (is_a, part_of)
    HIERARCHY_TYPES = {"is_a", "part_of"}
    concept_ids = {n["id"] for n in nodes if n["type"] == "concept"}
    hier_adj = defaultdict(set)
    for edge in edges:
        if edge["type"] in HIERARCHY_TYPES and edge["from"] in concept_ids:
            hier_adj[edge["from"]].add(edge["to"])

    cycles_found = []

    def dfs_cycle(node, visited, stack, path):
        visited.add(node)
        stack.add(node)
        path.append(node)
        for nb in hier_adj.get(node, []):
            if nb not in visited:
                dfs_cycle(nb, visited, stack, path)
            elif nb in stack:
                cycle_start = path.index(nb)
                cycles_found.append(path[cycle_start:] + [nb])
        path.pop()
        stack.discard(node)

    visited_global = set()
    for cid in concept_ids:
        if cid not in visited_global:
            dfs_cycle(cid, visited_global, set(), [])

    return {
        "self_loop_count": len(self_loops),
        "self_loops": self_loops,
        "duplicate_edge_count": len(duplicate_edges),
        "duplicate_edges": duplicate_edges[:20],
        "hierarchy_cycle_count": len(cycles_found),
        "hierarchy_cycles": cycles_found[:10]
    }

# ============================================================================ #
#  Auto Fix                                                                     #
# ============================================================================ #

def auto_fix(data: Dict, report: Dict, api_key: str = "") -> Tuple[Dict, Dict]:
    """
    분석 결과를 바탕으로 자동 수정 수행.
    - 계층 순환: 역방향 엣지 자동 제거
    - 고립 섬: 유사 노드 후보 제안 (병합은 수동)
    원본 data를 deepcopy해서 수정하므로 원본 파일은 보존됨.
    """
    import copy
    data = copy.deepcopy(data)
    graph = data.get("graph", {})
    edges = graph.get("edges", [])
    nodes = graph.get("nodes", [])

    fix_log = {"removed_edges": [], "island_suggestions": []}

    # ── 1. 계층 순환 엣지 제거 (LLM Judge로 방향 판정) ──────────────────────
    HIERARCHY_TYPES = {"is_a", "part_of"}
    cycles = report.get("logical_consistency", {}).get("hierarchy_cycles", [])

    edges_to_remove = set()
    if cycles and api_key and GENAI_AVAILABLE:
        # 각 사이클에서 두 엣지 쌍을 추출해 LLM에게 어느 쪽이 틀렸는지 판정 요청
        # cycle = [A, B, A] → 엣지 (A→B), (B→A) 중 하나가 역방향
        pairs = []
        for cycle in cycles:
            if len(cycle) >= 3:
                pairs.append((cycle[0], cycle[1], cycle[1], cycle[0]))

        pairs_text = "\n".join(
            f"{i+1}. [{typ}] '{a}'→'{b}'  vs  '{c}'→'{d}'"
            for i, (a, b, c, d) in enumerate(pairs)
            for typ in ["is_a/part_of"]
        )
        # 실제로는 엣지 타입도 포함해서 더 정확하게
        pairs_text = "\n".join(
            f"{i+1}. '{a}' → '{b}'  vs  '{b}' → '{a}'"
            for i, (a, b, _, _) in enumerate(pairs)
        )

        prompt = f"""다음은 지식그래프의 계층 관계(is_a / part_of) 사이클에서 발견된 엣지 쌍입니다.
각 쌍에서 어느 방향이 올바른 계층 관계인지 판정하세요.

규칙:
- is_a: "하위 개념 → 상위 개념" (커널 코드 is_a 커널 ✓, 커널 is_a 커널 코드 ❌)
- part_of: "부분 → 전체" (모드 레지스터 part_of CPU ✓, CPU part_of 모드 레지스터 ❌)
- 컴퓨터과학/운영체제 도메인 기준으로 판단

{pairs_text}

반드시 아래 JSON 배열로만 출력하세요:
[
  {{"index": 1, "correct": "A→B", "remove": "B→A", "reason": "판단 근거"}},
  ...
]"""

        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            text = resp.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            verdicts = json.loads(text.strip())
            for v in verdicts:
                remove_str = v.get("remove", "")   # "B→A" 형태
                if "→" in remove_str:
                    f, t = [x.strip() for x in remove_str.split("→")]
                    edges_to_remove.add((f, t))
                    fix_log.setdefault("llm_judge_verdicts", []).append({
                        "correct": v.get("correct"),
                        "removed": remove_str,
                        "reason":  v.get("reason", "")
                    })
        except Exception as e:
            print(f"  ❌ LLM Judge 실패 — 계층 순환 자동 수정 스킵: {e}")
            print(f"     순환 목록은 리포트 [7] 계층 순환 항목을 확인하세요.")
            fix_log["cycle_fix_error"] = str(e)
    else:
        if cycles:
            print(f"  ⚠️  --api-key 없음 — 계층 순환 {len(cycles)}건 자동 수정 불가")
            print(f"     --api-key를 제공하면 LLM Judge로 방향 판정 후 수정합니다.")
            fix_log["cycle_fix_skipped"] = f"api_key 없음, {len(cycles)}건 미처리"

    new_edges = []
    for edge in edges:
        key = (edge["from"], edge["to"])
        if edge["type"] in HIERARCHY_TYPES and key in edges_to_remove:
            fix_log["removed_edges"].append({
                "from": edge["from"], "to": edge["to"],
                "type": edge["type"], "reason": "계층 순환 제거 (LLM Judge 판정)"
            })
        else:
            new_edges.append(edge)
    graph["edges"] = new_edges

    # ── 2. 고립 섬 처리 제안 (자동 병합 없음) ────────────────────────────────
    concept_ids = [n["id"] for n in nodes if n["type"] == "concept"]
    islands = report.get("topology", {}).get("small_islands", [])

    for island in islands:
        for island_node in island:
            candidates = sorted(
                [c for c in concept_ids if c != island_node],
                key=lambda x: levenshtein(island_node, x)
            )[:3]
            fix_log["island_suggestions"].append({
                "island_node": island_node,
                "merge_candidates": candidates,
                "action": "수동 확인 필요 — 병합 또는 삭제"
            })

    data["graph"] = graph
    return data, fix_log

# ============================================================================ #
#  리포트 출력                                                                   #
# ============================================================================ #

def print_report(report: Dict):
    sep = "=" * 62
    print(f"\n{sep}")
    print("📊 지식그래프 구조적 건전성 리포트 (Step 1)")
    print(sep)
    s = report["summary"]
    print(f"  노드 {s['total_nodes']}개 (개념 {s['concept_nodes']}, 슬라이드 {s['slide_nodes']})  |  엣지 {s['total_edges']}개")

    # 그래프 구축 메타데이터
    gm = report.get("graph_build_metadata", {})
    if gm.get("processing_time_sec"):
        print(f"\n[그래프 구축 정보]")
        print(f"  Stage 3 처리 시간 : {gm['processing_time_sec']:.1f}초")
        if gm.get("total_concepts"):
            print(f"  추출 개념/관계    : {gm['total_concepts']}개 개념, {gm['total_relations']}개 관계")

    # 1. 고립 노드
    r = report["isolated_nodes"]
    status = "⚠️  주의" if r["isolated_ratio"] > 0.1 else "✅ 양호"
    print(f"\n[1] 고립 노드  {status}")
    print(f"  고립 노드 : {r['isolated_count']} / {r['total_concept_nodes']}개  ({r['isolated_ratio']*100:.1f}%)")
    if r["isolated_nodes"]:
        preview = r["isolated_nodes"][:5]
        suffix = f" 외 {len(r['isolated_nodes'])-5}개" if len(r['isolated_nodes']) > 5 else ""
        print(f"  예시      : {', '.join(preview)}{suffix}")

    # 2. 슬라이드 커버리지
    r = report["slide_coverage"]
    status = "⚠️  주의" if r["coverage_ratio"] < 0.9 else "✅ 양호"
    print(f"\n[2] 슬라이드 커버리지  {status}")
    print(f"  커버됨    : {r['covered_count']} / {r['total_slides']}개  ({r['coverage_ratio']*100:.1f}%)")
    print(f"  흐름 (■커버 □미커버): {r['flow_bar']}")
    if r["consecutive_gap_runs"]:
        print(f"  연속 미커버 구간:")
        for gap in r["consecutive_gap_runs"]:
            print(f"    슬라이드 {gap['start_slide']}~{gap['end_slide']}  ({gap['length']}장 연속)")
    if r["uncovered_slides"]:
        print(f"  미커버 슬라이드:")
        for s in r["uncovered_slides"][:5]:
            print(f"    • [{s['slide_id']}] {s['title']}")
        if len(r["uncovered_slides"]) > 5:
            print(f"    ... 외 {len(r['uncovered_slides'])-5}개")

    # 3. 관계 타입 분포
    r = report["relation_distribution"]
    status = "⚠️  편향" if r["is_biased"] else "✅ 양호"
    print(f"\n[3] 관계 타입 분포  {status}")
    print(f"  전체 개념 엣지 : {r['total_concept_edges']}개  |  사용 타입 : {r['used_type_count']}/{r['total_type_count']}가지")
    print(f"  최다 타입      : {r['top_type']}  ({r['top_type_ratio']*100:.1f}%)")
    print(f"  {'관계 타입':<22} {'건수':>5}  {'비율':>6}")
    print(f"  {'-'*40}")
    for rtype, data in r["distribution"].items():
        bar = "█" * int(data["ratio"] * 20)
        print(f"  {rtype:<22} {data['count']:>5}  {data['ratio']*100:>5.1f}%  {bar}")

    # 4. 밀도
    r = report["slide_concept_density"]
    status = "⚠️  주의" if r["avg_concepts_per_slide"] < 5 else "✅ 양호"
    print(f"\n[4] 슬라이드-개념 연결 밀도  {status}")
    print(f"  평균 : {r['avg_concepts_per_slide']}개/슬라이드  |  최소 {r['min_concepts']} / 최대 {r['max_concepts']}")
    print(f"  개념 0개 슬라이드 : {r['zero_concept_slides']}개")
    print(f"  분포:")
    for bucket, cnt in r["distribution_buckets"].items():
        bar = "█" * cnt
        print(f"    {bucket:>5}개 : {cnt:>3}슬라이드  {bar}")

    # 5. 엔티티 중복
    r = report.get("entity_duplicates", {})
    if r.get("skipped"):
        print(f"\n[5] 엔티티 중복 후보  — (재분석 시 생략)")
    else:
        total_cands = r["lexical_candidate_count"] + r["semantic_candidate_count"]
        status = "⚠️  확인 필요" if total_cands > 0 else "✅ 없음"
        print(f"\n[5] 엔티티 중복 후보  {status}")
        embed_note = "개념명 직접 임베딩" if r["using_name_embedding"] else "text_vector fallback"
        print(f"  Semantic 방식              : {embed_note}")
        print(f"  Lexical  (임계값≥{r['lex_threshold']}, 길이>5) : {r['lexical_candidate_count']}쌍")
        print(f"  Semantic (임계값≥{r['vec_threshold']})        : {r['semantic_candidate_count']}쌍")
        if r["lexical_candidates"]:
            print(f"  Lexical 상위:")
            for c in r["lexical_candidates"][:5]:
                print(f"    {c['similarity']:.2f}  '{c['concept_a']}'  ↔  '{c['concept_b']}'")
        if r["semantic_candidates"]:
            print(f"  Semantic 상위:")
            for c in r["semantic_candidates"][:5]:
                print(f"    {c['similarity']:.2f}  '{c['concept_a']}'  ↔  '{c['concept_b']}'")
        if r["llm_judge_results"]:
            same = [x for x in r["llm_judge_results"] if x.get("verdict") == "same"]
            diff = [x for x in r["llm_judge_results"] if x.get("verdict") == "different"]
            print(f"  LLM Judge 결과             : same {len(same)}쌍 / different {len(diff)}쌍")
            if same:
                print(f"  병합 제안:")
                for s in same[:5]:
                    print(f"    ✂️  '{s['concept_a']}'  →  '{s['concept_b']}'  ({s['reason']})")

    # 6. 위상 구조
    r = report["topology"]
    status = "⚠️  분절" if r["component_count"] > 3 else "✅ 양호"
    print(f"\n[6] 그래프 위상 구조  {status}")
    print(f"  연결 성분 수        : {r['component_count']}개")
    print(f"  최대 컴포넌트 크기  : {r['largest_component_size']}개 노드")
    if r["largest_component_diameter"] is not None:
        print(f"  Diameter (최대경로) : {r['largest_component_diameter']}홉")
        print(f"  경로 끝점           : {r['diameter_endpoints'][0]}  ↔  {r['diameter_endpoints'][1]}")
    print(f"  소규모 고립 섬 (≤3) : {r['small_islands_count']}개")
    if r["small_islands"]:
        for island in r["small_islands"][:5]:
            print(f"    {island}")

    # 7. 논리적 정합성
    r = report["logical_consistency"]
    issues = r["self_loop_count"] + r["duplicate_edge_count"] + r["hierarchy_cycle_count"]
    status = "⚠️  오류 있음" if issues > 0 else "✅ 이상 없음"
    print(f"\n[7] 논리적 정합성  {status}")
    print(f"  자기 참조(self-loop)    : {r['self_loop_count']}건")
    print(f"  중복 엣지               : {r['duplicate_edge_count']}건")
    print(f"  계층 순환(is_a/part_of) : {r['hierarchy_cycle_count']}건")
    if r["duplicate_edges"]:
        print(f"  중복 엣지 예시:")
        for e in r["duplicate_edges"][:3]:
            print(f"    [{e['type']}] {e['from']} → {e['to']}  ({e['count']}회)")
    if r["hierarchy_cycles"]:
        print(f"  순환 예시:")
        for cycle in r["hierarchy_cycles"][:2]:
            print(f"    {' → '.join(cycle)}")

    # 분석 성능 (API 사용량 + 소요 시간)
    perf = report.get("analysis_performance", {})
    if perf:
        print(f"\n{'─'*62}")
        print(f"[성능 요약]")
        print(f"  총 소요 시간    : {perf['total_elapsed_sec']}초")
        if perf.get("embed_calls", 0) > 0:
            print(f"  임베딩 호출     : {perf['embed_calls']}회  (추정 토큰: {perf['embed_tokens_est']})")
        if perf.get("llm_calls", 0) > 0:
            print(f"  LLM Judge 호출  : {perf['llm_calls']}회")
            print(f"  LLM 토큰        : 입력 {perf['llm_input_tokens']} / 출력 {perf['llm_output_tokens']} / 합계 {perf['llm_total_tokens']}")

    # Auto Fix 결과
    fix = report.get("auto_fix", {})
    if fix:
        print(f"\n[🔧 Auto Fix]")
        removed = fix.get("removed_edges", [])
        print(f"  제거된 순환 엣지 : {len(removed)}개")
        for e in removed:
            print(f"    ✂️  [{e['type']}] {e['from']} → {e['to']}")

        suggestions = fix.get("island_suggestions", [])
        if suggestions:
            print(f"  고립 섬 병합 제안 : {len(suggestions)}개 노드")
            for s in suggestions:
                print(f"    • '{s['island_node']}'  →  후보: {s['merge_candidates']}")

    print(f"\n{sep}")

# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="지식그래프 구조적 건전성 검증 (Step 1)")
    parser.add_argument("-g", "--graph", default="./output/knowledge_graph.json")
    parser.add_argument("-o", "--output", default=None, help="JSON 리포트 저장 경로")
    parser.add_argument("--sim-threshold", type=float, default=0.80, help="Lexical 중복 임계값 (기본 0.80)")
    parser.add_argument("--vec-threshold", type=float, default=0.80, help="Semantic 중복 임계값 (기본 0.80)")
    parser.add_argument("--api-key", default=os.getenv("GOOGLE_API_KEY", ""),
                        help="Google API Key (개념명 임베딩 + LLM Judge용, 없으면 text_vector fallback)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Semantic 후보 상위 30쌍을 Gemini로 same/different 판정 (API 비용 발생)")
    parser.add_argument("--domain-hint", default="",
                        help="LLM Judge 프롬프트에 주입할 도메인 힌트 (예: '운영체제 강의. rax/rbx는 CPU 레지스터로 서로 다른 개념')")
    parser.add_argument("--fix", action="store_true",
                    help="계층 순환 자동 제거 + 고립 섬 제안 출력 후 수정본 저장")
    parser.add_argument("--fix-output", default=None,
                        help="수정된 그래프 저장 경로 (기본: 원본파일명_fixed.json)")
    args = parser.parse_args()

    total_start = time.time()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"❌ 파일 없음: {graph_path}")
        return

    print(f"📂 로드 중: {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # knowledge_graph.json의 처리 시간 메타데이터 가져오기 (있으면)
    graph_build_meta = data.get("metadata", {})

    print(f"  노드 {len(nodes)}개, 엣지 {len(edges)}개 로드 완료")
    print("⏳ 분석 중...")

    report = {
        "source": str(graph_path),
        "graph_build_metadata": {
            "processing_time_sec": graph_build_meta.get("processing_time"),
            "total_concepts":      graph_build_meta.get("total_concepts"),
            "total_slides":        graph_build_meta.get("total_slides"),
            "total_relations":     graph_build_meta.get("total_relations"),
        },
        "summary": {
            "total_nodes":    len(nodes),
            "total_edges":    len(edges),
            "concept_nodes":  sum(1 for n in nodes if n["type"] == "concept"),
            "slide_nodes":    sum(1 for n in nodes if n["type"] == "slide"),
        },
        "isolated_nodes":        check_isolated_nodes(nodes, edges),
        "slide_coverage":        check_slide_coverage(nodes, edges),
        "relation_distribution": check_relation_type_distribution(edges),
        "slide_concept_density": check_slide_concept_density(nodes, edges),
        "entity_duplicates":     check_entity_duplicates(
                                     nodes,
                                     lex_threshold=args.sim_threshold,
                                     vec_threshold=args.vec_threshold,
                                     api_key=args.api_key,
                                     use_llm_judge=args.llm_judge,
                                     domain_hint=args.domain_hint
                                 ),
        "topology":              check_topology(nodes, edges),
        "logical_consistency":   check_logical_consistency(nodes, edges),
    }

    # ── 1단계: 원본 분석 리포트 ──────────────────────────────────────────────
    print_report(report)

    base_dir = Path(args.output).parent if args.output else graph_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    report_path = base_dir / "quality_report_step1.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 리포트 저장: {report_path}")

    # ── 2단계: 계층 순환이 있으면 수정 → 재분석 ──────────────────────────────
    cycles = report.get("logical_consistency", {}).get("hierarchy_cycles", [])
    if not cycles:
        print("ℹ️  계층 순환 없음 — fixed 파일 생성 생략")
        return

    print(f"\n🔧 계층 순환 {len(cycles)}건 발견 — Auto Fix 실행 중...")
    fixed_data, fix_log = auto_fix(data, report, api_key=args.api_key)
    removed = fix_log.get("removed_edges", [])

    if not removed:
        print("ℹ️  제거된 엣지 없음 (LLM 판정 실패 등) — fixed 파일 생성 생략")
        return

    # knowledge_graph_fixed.json 저장
    fixed_graph_path = graph_path.parent / "knowledge_graph_fixed.json"
    with open(fixed_graph_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    print(f"💾 수정 그래프 저장: {fixed_graph_path}  ({len(removed)}개 엣지 제거)")

    # ── 3단계: 수정된 그래프 재분석 ───────────────────────────────────────────
    print("\n⏳ 수정본 재분석 중...")
    fixed_graph = fixed_data.get("graph", {})
    fixed_nodes = fixed_graph.get("nodes", [])
    fixed_edges = fixed_graph.get("edges", [])

    fixed_report = {
        "source": str(fixed_graph_path),
        "based_on": str(report_path),
        "auto_fix": fix_log,
        "summary": {
            "total_nodes":   len(fixed_nodes),
            "total_edges":   len(fixed_edges),
            "concept_nodes": sum(1 for n in fixed_nodes if n["type"] == "concept"),
            "slide_nodes":   sum(1 for n in fixed_nodes if n["type"] == "slide"),
        },
        "isolated_nodes":        check_isolated_nodes(fixed_nodes, fixed_edges),
        "slide_coverage":        check_slide_coverage(fixed_nodes, fixed_edges),
        "relation_distribution": check_relation_type_distribution(fixed_edges),
        "slide_concept_density": check_slide_concept_density(fixed_nodes, fixed_edges),
        "topology":              check_topology(fixed_nodes, fixed_edges),
        "logical_consistency":   check_logical_consistency(fixed_nodes, fixed_edges),
        "entity_duplicates": {"skipped": True},
    }

    print_report(fixed_report)

    fixed_report_path = base_dir / "quality_report_step1_fixed.json"
    with open(fixed_report_path, "w", encoding="utf-8") as f:
        json.dump(fixed_report, f, indent=2, ensure_ascii=False)
    print(f"💾 수정본 리포트 저장: {fixed_report_path}")

if __name__ == "__main__":
    main()