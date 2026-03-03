"""
지식그래프 QA Faithfulness 검증 스크립트 (Step 3)

"그래프 기반 답변이 슬라이드 원문에 근거하고 있는가?"를 LLM Judge로 검증합니다.

질문 그룹:
  A. 기본 질문          — 전반적인 그래프 faithfulness
  B. 오디오 기반 질문   — 음성 정보가 그래프에 잘 반영됐는가
  C. 이미지 기반 질문   — t1(이미지 텍스트화) 기여도 측정

판정 기준:
  faithful   : 답변 전체가 슬라이드 원문에 근거
  partial    : 일부 근거 있으나 외부 지식 혼입
  unfaithful : 슬라이드에 없는 내용으로 답변

Usage:
  python graph_quality_step3.py -g ./output/knowledge_graph_fixed.json
  python graph_quality_step3.py -g ./output/knowledge_graph_fixed.json -q ./test_questions.json
  python graph_quality_step3.py -g ./output/knowledge_graph_fixed.json -o ./output/step3_report.json
"""

import json
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("❌ google-genai 패키지가 필요합니다: pip install google-genai")

# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL      = "gemini-2.5-flash"
JUDGE_MODEL     = "gemini-2.5-flash"
EMBEDDING_DIM   = 768

GROUP_LABELS = {
    "A_basic":  "A. 기본 질문",
    "B_audio":  "B. 오디오 기반 질문",
    "C_image":  "C. 이미지 기반 질문",
}


# ============================================================================ #
#  KnowledgeGraphQA (graph_qa.py에서 필요한 부분만 인라인)                       #
# ============================================================================ #

class KnowledgeGraphQA:
    def __init__(self, graph_path: str, client):
        self.client = client

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph  = data.get("graph", {})
        self.slides = data.get("slides", [])
        self.nodes  = {n["id"]: n for n in self.graph.get("nodes", [])}
        self.edges  = self.graph.get("edges", [])

        self._build_indices()

    def _build_indices(self):
        self.concept_nodes = {}
        self.slide_nodes   = {}
        self.adjacency     = {}

        for node_id, node in self.nodes.items():
            if node["type"] == "concept":
                self.concept_nodes[node_id] = node
            else:
                self.slide_nodes[node_id] = node

        for edge in self.edges:
            if edge["type"] == "contains":
                continue
            f, t, r = edge["from"], edge["to"], edge["type"]
            self.adjacency.setdefault(f, []).append((t, r))
            self.adjacency.setdefault(t, []).append((f, f"inv_{r}"))

    def _embed(self, text: str) -> List[float]:
        if len(text) > 10000:
            text = text[:10000]
        resp = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
        )
        return resp.embeddings[0].values

    def _cosine_sim(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        a, b = np.array(a), np.array(b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    def _search_concepts(self, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
        query_vec = self._embed(query)
        scores = []
        for cid, node in self.concept_nodes.items():
            vec = node.get("text_vector")
            if vec:
                scores.append((self._cosine_sim(query_vec, vec), cid))
        scores.sort(reverse=True)
        return scores[:top_k]

    def _expand_concepts(self, concept_ids: List[str], hops: int = 1):
        expanded = set(concept_ids)
        current  = set(concept_ids)
        for _ in range(hops):
            next_level = set()
            for c in current:
                for neighbor, _ in self.adjacency.get(c, []):
                    if neighbor not in expanded and neighbor in self.concept_nodes:
                        next_level.add(neighbor)
            expanded |= next_level
            current   = next_level
        return expanded

    def _get_concept_context(self, concept_id: str) -> Tuple[str, List[str]]:
        """컨텍스트 문자열과 사용된 slide_id 목록 반환"""
        node = self.concept_nodes.get(concept_id, {})
        related_slides = node.get("slides", [])

        relations = []
        for edge in self.edges:
            if edge["type"] == "contains":
                continue
            if edge["from"] == concept_id:
                relations.append(f"  → {edge['to']} ({edge['type']})")
            elif edge["to"] == concept_id:
                relations.append(f"  ← {edge['from']} ({edge['type']})")

        slide_contents = []
        used_slides    = []
        for slide_id in related_slides[:2]:
            for slide in self.slides:
                if slide.get("slide_id") == slide_id:
                    t3 = slide.get("t3", "")
                    if t3:
                        slide_contents.append(f"<source id=\"{slide_id}\">\n{t3[:600]}\n</source>")
                        used_slides.append(slide_id)
                    break

        ctx  = f"【{concept_id}】\n"
        if relations:
            ctx += "관계:\n" + "\n".join(relations[:5]) + "\n"
        if slide_contents:
            ctx += "내용:\n" + "\n".join(slide_contents)

        return ctx, used_slides

    def ask_with_sources(self, question: str) -> Tuple[str, List[str]]:
        """답변 + 실제 사용된 slide_id 목록 반환"""
        top_concepts = self._search_concepts(question, top_k=3)

        # Fallback: 전체 t3 사용
        if not top_concepts or top_concepts[0][0] < 0.5:
            all_t3, all_slide_ids = [], []
            for slide in self.slides:
                t3 = slide.get("t3", "")
                if t3:
                    sid = slide.get("slide_id", "")
                    all_t3.append(f"[슬라이드: {sid}]\n{t3}")
                    all_slide_ids.append(sid)

            full_context = "\n\n---\n\n".join(all_t3)
            if len(full_context) > 100000:
                full_context = full_context[:100000]

            prompt = f"""당신은 강의 자료만을 근거로 답변하는 학습 도우미입니다.
아래 <lecture_content> 태그 안의 정보만 사용하여 질문에 답변하세요.

<lecture_content>
{full_context}
</lecture_content>

[질문]
{question}

[답변 규칙]
1. lecture_content에 있는 정보만 사용하세요. 외부 지식을 추가하지 마세요.
2. 강의 자료에서 찾을 수 없는 내용은 "해당 정보는 강의 자료에 없습니다."라고 답하세요.
3. 출처를 반드시 표기하세요 (예: [슬라이드: slide_id]).
"""
            resp = self.client.models.generate_content(model=CHAT_MODEL, contents=prompt)
            return resp.text, all_slide_ids

        # 그래프 탐색 경로
        seed_concepts = [c[1] for c in top_concepts]
        expanded      = self._expand_concepts(seed_concepts, hops=1)

        contexts    = []
        used_slides = []
        for cid in list(expanded)[:7]:
            ctx, sids = self._get_concept_context(cid)
            contexts.append(ctx)
            used_slides.extend(sids)

        used_slides = list(dict.fromkeys(used_slides))  # 순서 유지 중복 제거
        full_context = "\n\n---\n\n".join(contexts)

        prompt = f"""당신은 강의 자료만을 근거로 답변하는 학습 도우미입니다.
아래 <knowledge_graph> 태그 안의 정보만 사용하여 질문에 답변하세요.

<knowledge_graph>
{full_context}
</knowledge_graph>

[질문]
{question}

[답변 규칙]
1. knowledge_graph에 명시된 정보만 사용하세요. 외부 지식을 추가하지 마세요.
2. 강의 자료에서 찾을 수 없는 내용은 "해당 정보는 강의 자료에 없습니다."라고 답하세요.
3. 출처를 반드시 표기하세요 (예: [슬라이드: slide_id]).
"""
        resp = self.client.models.generate_content(model=CHAT_MODEL, contents=prompt)
        return resp.text, used_slides


# ============================================================================ #
#  LLM Faithfulness Judge                                                       #
# ============================================================================ #

FAITHFULNESS_PROMPT = """당신은 강의 Q&A 품질 검증 전문가입니다.
아래 [슬라이드 원문]과 [QA 답변]을 비교하여 답변의 faithfulness를 판정하세요.

[슬라이드 원문]
{slide_content}

[질문]
{question}

[QA 답변]
{answer}

판정 기준:
- "faithful"   : 답변의 모든 주요 주장이 슬라이드 원문에 명시되거나 직접 추론 가능
- "partial"    : 일부 주장은 원문에 근거하나, 일부는 원문에 없는 외부 지식 혼입
- "unfaithful" : 답변의 핵심 내용이 슬라이드 원문에 없거나 원문과 모순됨
- "no_answer"  : 답변이 "강의 자료에 없습니다"로 정보 제공을 거부한 경우

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "verdict": "faithful" | "partial" | "unfaithful" | "no_answer",
  "reason": "판단 근거 한 줄",
  "hallucinated_claims": ["원문에 없는 주장 목록 (faithful이면 빈 배열)"]
}}"""


def judge_faithfulness(
    question: str,
    answer: str,
    used_slide_ids: List[str],
    slides: List[Dict],
    client,
) -> Dict:
    """단일 QA 쌍에 대한 faithfulness 판정"""

    # 사용된 슬라이드 원문 수집 (최대 5개, 각 600자)
    slide_map = {s["slide_id"]: s for s in slides}
    slide_texts = []
    for sid in used_slide_ids[:5]:
        s = slide_map.get(sid, {})
        t3 = s.get("t3", "")
        if t3:
            slide_texts.append(f"[{sid}]\n{t3[:600]}")

    if not slide_texts:
        slide_content = "(슬라이드 원문 없음 — fallback 전체 컨텍스트 사용)"
    else:
        slide_content = "\n\n".join(slide_texts)

    prompt = FAITHFULNESS_PROMPT.format(
        slide_content=slide_content,
        question=question,
        answer=answer[:2000],
    )

    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        text = resp.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())
        token_info = {}
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            m = resp.usage_metadata
            token_info = {
                "input_tokens":  getattr(m, "prompt_token_count", 0),
                "output_tokens": getattr(m, "candidates_token_count", 0),
            }
        return {**result, **token_info, "error": None}

    except Exception as e:
        return {
            "verdict": "error", "reason": str(e),
            "hallucinated_claims": [], "error": str(e)
        }


# ============================================================================ #
#  평가 실행                                                                     #
# ============================================================================ #

def run_evaluation(
    questions: List[Dict],
    qa: KnowledgeGraphQA,
    client,
) -> Tuple[List[Dict], Dict]:
    """전체 질문 평가 + 사용량 집계"""

    results      = []
    total_input  = total_output = 0
    t_start      = time.time()

    for i, q in enumerate(questions):
        qid      = q["id"]
        group    = q["group"]
        question = q["question"]

        print(f"\n[{i+1:>2}/{len(questions)}] {qid} ({group})")
        print(f"  ❓ {question[:60]}")

        # QA 답변 생성
        try:
            answer, used_slides = qa.ask_with_sources(question)
        except Exception as e:
            results.append({
                **q, "answer": "", "used_slides": [],
                "verdict": "error", "reason": str(e),
                "hallucinated_claims": [], "error": str(e)
            })
            print(f"  💥 QA 오류: {e}")
            continue

        print(f"  📄 참조 슬라이드: {used_slides[:3]}")

        # Faithfulness 판정
        judge = judge_faithfulness(question, answer, used_slides, qa.slides, client)

        total_input  += judge.get("input_tokens", 0)
        total_output += judge.get("output_tokens", 0)

        verdict = judge.get("verdict", "error")
        icon    = {"faithful": "✅", "partial": "🔶", "unfaithful": "❌",
                   "no_answer": "⬜", "error": "💥"}.get(verdict, "❓")
        print(f"  {icon} {verdict}  |  {judge.get('reason', '')[:60]}")

        results.append({
            **q,
            "answer":               answer,
            "used_slides":          used_slides,
            "verdict":              verdict,
            "reason":               judge.get("reason", ""),
            "hallucinated_claims":  judge.get("hallucinated_claims", []),
            "error":                judge.get("error"),
        })

    elapsed = time.time() - t_start
    usage = {
        "total_questions": len(results),
        "elapsed_sec":     round(elapsed, 2),
        "input_tokens":    total_input,
        "output_tokens":   total_output,
        "total_tokens":    total_input + total_output,
    }
    return results, usage


# ============================================================================ #
#  지표 계산                                                                     #
# ============================================================================ #

def compute_metrics(results: List[Dict]) -> Dict:
    """전체 및 그룹별 faithfulness 비율 계산"""

    def stats_of(items):
        valid   = [r for r in items if r["verdict"] != "error"]
        if not valid:
            return None
        verdicts = Counter(r["verdict"] for r in valid)
        faithful = verdicts.get("faithful", 0) + verdicts.get("partial", 0) * 0.5
        return {
            "sample_count":      len(valid),
            "faithfulness":      round(faithful / len(valid), 4),
            "strict_faithfulness": round(verdicts.get("faithful", 0) / len(valid), 4),
            "verdicts":          dict(verdicts),
        }

    group_results: Dict[str, List] = defaultdict(list)
    for r in results:
        group_results[r["group"]].append(r)

    group_stats = {g: stats_of(items) for g, items in group_results.items()}

    # 환각 클레임 수집
    hallucinations = [
        {"id": r["id"], "group": r["group"],
         "question": r["question"], "claims": r["hallucinated_claims"]}
        for r in results
        if r.get("hallucinated_claims") and r["verdict"] in ("partial", "unfaithful")
    ]

    return {
        "overall":        stats_of(results),
        "group_stats":    group_stats,
        "hallucinations": hallucinations,
    }


# ============================================================================ #
#  리포트 출력                                                                   #
# ============================================================================ #

def print_report(metrics: Dict, usage: Dict):
    sep = "=" * 64
    print(f"\n{sep}")
    print("🔍 QA Faithfulness 검증 리포트 (Step 3)")
    print(sep)

    overall = metrics["overall"]
    if overall:
        fs  = overall["faithfulness"]
        sfs = overall["strict_faithfulness"]
        status = "✅ 양호" if fs >= 0.8 else "⚠️  주의"
        print(f"\n[전체 Faithfulness]  {status}")
        print(f"  샘플 수         : {overall['sample_count']}개")
        print(f"  Faithfulness    : {fs*100:.1f}%  (partial 0.5 가중)")
        print(f"  Strict          : {sfs*100:.1f}%  (faithful만)")
        v = overall["verdicts"]
        print(f"  판정 분포       : faithful={v.get('faithful',0)}  "
              f"partial={v.get('partial',0)}  "
              f"unfaithful={v.get('unfaithful',0)}  "
              f"no_answer={v.get('no_answer',0)}  "
              f"error={v.get('error',0)}")

    print(f"\n[그룹별 Faithfulness]")
    for gkey, glabel in GROUP_LABELS.items():
        gs = metrics["group_stats"].get(gkey)
        if not gs:
            print(f"\n  — {glabel}  (샘플 없음)")
            continue
        fs_g   = gs["faithfulness"]
        status = "✅" if fs_g >= 0.8 else "⚠️"
        print(f"\n  {status} {glabel}")
        print(f"     샘플 수  : {gs['sample_count']}개")
        print(f"     Faithfulness : {fs_g*100:.1f}%  "
              f"(strict {gs['strict_faithfulness']*100:.1f}%)")
        vd = gs["verdicts"]
        print(f"     판정     : faithful={vd.get('faithful',0)}  "
              f"partial={vd.get('partial',0)}  "
              f"unfaithful={vd.get('unfaithful',0)}  "
              f"no_answer={vd.get('no_answer',0)}")

    # 환각 클레임 요약
    hals = metrics["hallucinations"]
    print(f"\n[환각 클레임 분석]")
    if not hals:
        print("  ✅ 환각 없음")
    else:
        print(f"  총 {len(hals)}개 질문에서 환각 발견")
        for h in hals[:5]:
            print(f"\n  [{h['group']}] {h['id']}: {h['question'][:50]}")
            for claim in h["claims"][:2]:
                print(f"    ✗ {claim[:80]}")
        if len(hals) > 5:
            print(f"  ... 외 {len(hals)-5}개 (JSON 참조)")

    # C그룹 인사이트
    gs_c = metrics["group_stats"].get("C_image")
    gs_b = metrics["group_stats"].get("B_audio")
    gs_a = metrics["group_stats"].get("A_basic")
    print(f"\n[인사이트]")
    if gs_c and gs_a:
        diff = gs_c["strict_faithfulness"] - gs_a["strict_faithfulness"]
        if diff < -0.15:
            print(f"  ⚠️  C그룹(이미지) faithfulness가 A그룹 대비 "
                  f"{abs(diff)*100:.0f}%p 낮음")
            print(f"     → t1(이미지 텍스트화) 품질 개선 또는 이미지 벡터 도입 검토")
        else:
            print(f"  ✅ C그룹(이미지) faithfulness 양호 "
                  f"→ t1 텍스트화가 이미지 정보를 충분히 포착 중")
    if gs_b and gs_a:
        diff = gs_b["strict_faithfulness"] - gs_a["strict_faithfulness"]
        if diff < -0.15:
            print(f"  ⚠️  B그룹(오디오) faithfulness가 A그룹 대비 "
                  f"{abs(diff)*100:.0f}%p 낮음")
            print(f"     → t2(오디오 전사) 정제 품질 개선 필요")

    print(f"\n[API 사용량]")
    print(f"  소요 시간 : {usage['elapsed_sec']}초")
    print(f"  토큰      : 입력 {usage['input_tokens']} / "
          f"출력 {usage['output_tokens']} / 합계 {usage['total_tokens']}")
    print(f"\n{sep}")


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="QA Faithfulness 검증 (Step 3)")
    parser.add_argument("-g", "--graph",
                        default="./output/knowledge_graph_fixed.json",
                        help="knowledge_graph_fixed.json 경로")
    parser.add_argument("-q", "--questions",
                        default="./test_questions.json",
                        help="테스트 질문 JSON 경로")
    parser.add_argument("-o", "--output",
                        default=None,
                        help="리포트 저장 경로 (기본: graph 파일과 같은 디렉토리)")
    parser.add_argument("--api-key",
                        default=os.getenv("GOOGLE_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("❌ --api-key 또는 GOOGLE_API_KEY 환경변수가 필요합니다.")
        return
    if not GENAI_AVAILABLE:
        print("❌ google-genai 패키지를 설치하세요.")
        return

    graph_path = Path(args.graph)
    q_path     = Path(args.questions)

    # knowledge_graph_fixed.json 우선, 없으면 원본 사용
    if not graph_path.exists():
        fallback = graph_path.parent / "knowledge_graph.json"
        if fallback.exists():
            print(f"⚠️  {graph_path} 없음 → {fallback} 사용")
            graph_path = fallback
        else:
            print(f"❌ 파일 없음: {graph_path}")
            return

    if not q_path.exists():
        print(f"❌ 질문 파일 없음: {q_path}")
        return

    print(f"📂 그래프 로드 중: {graph_path}")
    print(f"📋 질문 파일 로드 중: {q_path}")

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    client = genai.Client(api_key=args.api_key)
    qa     = KnowledgeGraphQA(str(graph_path), client)

    print(f"\n📊 질문 수: {len(questions)}개")
    group_counts = Counter(q["group"] for q in questions)
    for g, cnt in sorted(group_counts.items()):
        print(f"  {GROUP_LABELS.get(g, g)}: {cnt}개")

    print(f"\n🤖 평가 시작...")
    results, usage = run_evaluation(questions, qa, client)

    metrics = compute_metrics(results)
    print_report(metrics, usage)

    # 저장
    base_dir = Path(args.output).parent if args.output else graph_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "quality_report_step3.json"

    report = {
        "source":    str(graph_path),
        "questions": str(q_path),
        "metrics":   metrics,
        "api_usage": usage,
        "results":   results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 리포트 저장: {out_path}")


if __name__ == "__main__":
    main()