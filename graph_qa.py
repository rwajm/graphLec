"""
지식그래프 기반 Q&A 시스템

Input:
  - knowledge_graph.json: 개념, 관계, 벡터 포함

Features:
  - 텍스트 벡터로 관련 개념 검색
  - 그래프 탐색으로 관련 개념 확장 (1-hop)
  - 확장된 컨텍스트로 답변 생성
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
from google import genai
from google.genai import types

# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_DIM = 768


# ============================================================================ #
#  지식그래프 Q&A 시스템                                                         #
# ============================================================================ #

class KnowledgeGraphQA:
    def __init__(self, graph_path: str):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # 그래프 로드
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.graph = data.get('graph', {})
        self.slides = data.get('slides', [])
        
        self.nodes = {n['id']: n for n in self.graph.get('nodes', [])}
        self.edges = self.graph.get('edges', [])
        
        # 인덱스 구축
        self._build_indices()
        
        print(f"✓ 로드 완료")
        print(f"  • 개념: {len(self.concept_nodes)}개")
        print(f"  • 슬라이드: {len(self.slide_nodes)}개")
        print(f"  • 관계: {len(self.edges)}개")
    
    def _build_indices(self):
        """검색용 인덱스 구축"""
        self.concept_nodes = {}
        self.slide_nodes = {}
        self.adjacency = {}  # concept → [(neighbor, relation_type)]
        
        # 노드 분류
        for node_id, node in self.nodes.items():
            if node['type'] == 'concept':
                self.concept_nodes[node_id] = node
            else:
                self.slide_nodes[node_id] = node
        
        # 인접 리스트 구축 (개념 간 관계만)
        for edge in self.edges:
            if edge['type'] == 'contains':
                continue
            
            from_id = edge['from']
            to_id = edge['to']
            rel_type = edge['type']
            
            if from_id not in self.adjacency:
                self.adjacency[from_id] = []
            self.adjacency[from_id].append((to_id, rel_type))
            
            # 양방향 탐색용 (역방향)
            if to_id not in self.adjacency:
                self.adjacency[to_id] = []
            self.adjacency[to_id].append((from_id, f"inv_{rel_type}"))
    
    def _embed(self, text: str) -> List[float]:
        """텍스트 → 벡터"""
        if len(text) > 10000:
            text = text[:10000]
        
        resp = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
        )
        return resp.embeddings[0].values
    
    def _cosine_sim(self, a, b) -> float:
        """코사인 유사도"""
        if a is None or b is None:
            return 0.0
        a, b = np.array(a), np.array(b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return np.dot(a, b) / (na * nb)
    
    def _search_concepts(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict]]:
        """질문과 유사한 개념 검색"""
        query_vec = self._embed(query)
        
        scores = []
        for concept_id, node in self.concept_nodes.items():
            text_vec = node.get('text_vector')
            if text_vec:
                sim = self._cosine_sim(query_vec, text_vec)
                scores.append((sim, concept_id, node))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
    
    def _expand_concepts(self, concept_ids: List[str], hops: int = 1) -> Set[str]:
        """그래프 탐색으로 관련 개념 확장"""
        expanded = set(concept_ids)
        current = set(concept_ids)
        
        for _ in range(hops):
            next_level = set()
            for concept in current:
                neighbors = self.adjacency.get(concept, [])
                for neighbor, _ in neighbors:
                    if neighbor not in expanded and neighbor in self.concept_nodes:
                        next_level.add(neighbor)
            expanded.update(next_level)
            current = next_level
        
        return expanded
    
    def _get_concept_context(self, concept_id: str) -> str:
        """개념에 대한 컨텍스트 생성"""
        node = self.concept_nodes.get(concept_id, {})
        
        # 관련 슬라이드 찾기
        related_slides = node.get('slides', [])
        
        # 관계 찾기
        relations = []
        for edge in self.edges:
            if edge['type'] == 'contains':
                continue
            if edge['from'] == concept_id:
                relations.append(f"  → {edge['to']} ({edge['type']})")
            elif edge['to'] == concept_id:
                relations.append(f"  ← {edge['from']} ({edge['type']})")
        
        # 슬라이드 내용 수집
        slide_contents = []
        for slide_id in related_slides[:2]:  # 최대 2개 슬라이드
            for slide in self.slides:
                if slide.get('slide_id') == slide_id:
                    t3 = slide.get('t3', '')
                    if t3:
                        slide_contents.append(
                            f"<source id=\"{slide_id}\">\n{t3[:500]}\n</source>"
                        )
                    break
        
        context = f"【{concept_id}】\n"
        if relations:
            context += "관계:\n" + "\n".join(relations[:5]) + "\n"
        if slide_contents:
            context += "내용:\n" + "\n".join(slide_contents)
        context += "\n</concept>"

        return context
    
    def _ask_with_full_context(self, question: str) -> str:
        """Fallback: t3 전체 전달 (Long-Context 방식)"""
        print("⚠️ 유사도 낮음 → 전체 컨텍스트 사용")
        
        all_t3 = []
        for slide in self.slides:
            t3 = slide.get('t3', '')
            if t3:
                # 슬라이드 ID와 내용을 명확히 구분하여 추가
                all_t3.append(f"[슬라이드: {slide.get('slide_id', 'Unknown')}]\n{t3}")
        
        full_context = "\n\n---\n\n".join(all_t3)
        
        # 컨텍스트 길이 제한 설정
        if len(full_context) > 100000:
            full_context = full_context[:100000]
        
        prompt = f"""당신은 강의 자료만을 근거로 답변하는 학습 도우미입니다.
        아래 <lecture_content> 태그 안의 정보만 사용하여 질문에 답변하세요.

        <lecture_content>
        {full_context}
        </lecture_content>

        [질문]
        {question}

        [답변 규칙 - 반드시 준수]
        1. 위 lecture_content에 있는 정보만 사용하세요. 외부 지식을 추가하지 마세요.
        2. 모든 주장에는 출처를 반드시 표기하세요.
        - 슬라이드 출처: [슬라이드: slide_id]
        3. lecture_content에서 찾을 수 없는 내용은 "해당 정보는 강의 자료에 없습니다."라고 답하세요.
        4. 답변 마지막에 [참고 자료] 섹션을 추가하고 사용한 슬라이드를 나열하세요.
        """

        # 실제 모델 호출 및 응답 반환 로직 추가
        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )
        
        return response.text

    def ask(self, question: str) -> str:
        """질문에 답변"""
        print(f"\n❓ 질문: {question}")
        
        # 1. 관련 개념 검색
        top_concepts = self._search_concepts(question, top_k=3)
        
        # Fallback 조건
        if not top_concepts or top_concepts[0][0] < 0.5:
            return self._ask_with_full_context(question)

        print(f"\n🔍 검색된 개념:")
        for sim, concept_id, _ in top_concepts:
            print(f"  • {concept_id} (유사도: {sim:.2f})")
        
        # 2. 그래프 탐색으로 확장
        seed_concepts = [c[1] for c in top_concepts]
        expanded = self._expand_concepts(seed_concepts, hops=1)
        
        print(f"\n🕸️ 확장된 개념: {len(expanded)}개")
        
        # 3. 컨텍스트 구성
        contexts = []
        for concept_id in list(expanded)[:7]:
            ctx = self._get_concept_context(concept_id)
            contexts.append(ctx)
        
        full_context = "\n\n---\n\n".join(contexts)
        
        # 4. 답변 생성
        prompt = f"""당신은 강의 자료만을 근거로 답변하는 학습 도우미입니다.
    아래 <knowledge_graph> 태그 안의 정보만 사용하여 질문에 답변하세요.

    <knowledge_graph>
    {full_context}
    </knowledge_graph>

    [질문]
    {question}

    [답변 규칙 - 반드시 준수]
    1. 위 knowledge_graph에 명시된 정보만 사용하세요. 외부 지식을 추가하지 마세요.
    2. 모든 주장에는 출처를 반드시 표기하세요.
    - 슬라이드 출처: [슬라이드: slide_id]
    - 개념 관계 출처: [개념: concept_id]
    3. knowledge_graph에서 찾을 수 없는 내용은 "해당 정보는 강의 자료에 없습니다."라고 답하세요.
    4. 답변 마지막에 [참고 자료] 섹션을 추가하고 사용한 모든 출처를 나열하세요.
    """
        
        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )
        
        return response.text
    
    def explain_concept(self, concept: str) -> str:
        """특정 개념 설명"""
        # 정확히 일치하는 개념 찾기
        if concept in self.concept_nodes:
            target = concept
        else:
            # 유사 개념 검색
            results = self._search_concepts(concept, top_k=1)
            if not results or results[0][0] < 0.3:
                return f"'{concept}' 관련 개념을 찾지 못했습니다."
            target = results[0][1]
        
        print(f"\n📖 개념: {target}")
        
        # 관련 개념 확장
        expanded = self._expand_concepts([target], hops=1)
        
        # 컨텍스트 구성
        contexts = []
        for concept_id in list(expanded)[:5]:
            ctx = self._get_concept_context(concept_id)
            contexts.append(ctx)
        
        full_context = "\n\n---\n\n".join(contexts)
        
        prompt = f"""아래 지식그래프 정보를 바탕으로 '{target}' 개념을 설명하세요.

[지식그래프 정보]
{full_context}

지침:
- 개념의 정의와 특징을 설명하세요.
- 관련된 다른 개념과의 관계를 포함하세요.
- 쉽게 이해할 수 있도록 설명하세요.
"""
        
        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )
        
        return response.text
    
    def show_relations(self, concept: str) -> str:
        """개념의 관계 시각화"""
        if concept not in self.concept_nodes:
            results = self._search_concepts(concept, top_k=1)
            if not results or results[0][0] < 0.3:
                return f"'{concept}' 관련 개념을 찾지 못했습니다."
            concept = results[0][1]
        
        output = [f"\n🔗 '{concept}'의 관계:\n"]
        
        for edge in self.edges:
            if edge['type'] == 'contains':
                continue
            
            if edge['from'] == concept:
                output.append(f"  {concept} ──[{edge['type']}]──▶ {edge['to']}")
            elif edge['to'] == concept:
                output.append(f"  {edge['from']} ──[{edge['type']}]──▶ {concept}")
        
        if len(output) == 1:
            output.append("  (관계 없음)")
        
        return "\n".join(output)


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="지식그래프 Q&A")
    parser.add_argument("-g", "--graph", default="./output/knowledge_graph.json")
    args = parser.parse_args()
    
    if not Path(args.graph).exists():
        print(f"❌ Graph not found: {args.graph}")
        return
    
    qa = KnowledgeGraphQA(args.graph)
    
    print("\n" + "="*50)
    print("💬 지식그래프 Q&A 시스템")
    print("="*50)
    print("명령어:")
    print("  • 질문 입력: 자연어로 질문")
    print("  • /explain <개념>: 개념 설명")
    print("  • /rel <개념>: 관계 보기")
    print("  • /q: 종료")
    print("="*50)
    
    while True:
        user_input = input("\n입력: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == '/q':
            print("종료합니다.")
            break
        
        if user_input.startswith('/explain '):
            concept = user_input[9:].strip()
            answer = qa.explain_concept(concept)
            print(f"\n🤖 설명:\n{answer}")
        
        elif user_input.startswith('/rel '):
            concept = user_input[5:].strip()
            result = qa.show_relations(concept)
            print(result)
        
        else:
            answer = qa.ask(user_input)
            print(f"\n🤖 답변:\n{answer}")


if __name__ == "__main__":
    main()