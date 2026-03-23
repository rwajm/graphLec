"""
query_system.py — 자연어 → Cypher → Neo4j → 답변 생성

사용법:
  python query_system.py --stem os1-1
  python query_system.py --stem os1-1 --port 8000
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
#  설정
# ============================================================================

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "rhlh1234"

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_2") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL   = "gemini-2.0-flash"
RETRY_DELAYS   = [5, 15, 30]  # 429 발생 시 재시도 대기(초)
RETRY_DELAYS   = [5, 15, 30]  # 429 발생 시 재시도 대기(초)

# ============================================================================
#  그래프 스키마 (Cypher 생성용 컨텍스트)
# ============================================================================

GRAPH_SCHEMA = """
노드 타입과 주요 프로퍼티:
- Video          : id, title
- Slides         : id
- Scenes         : id
- Slide          : id, slide_number, title, slide_text(슬라이드 원문), role(core/transitional/elaborated), start_sec, end_sec, emphasis_total
- Scene          : id, slide_id, context_index, start(초), end(초), stressed(bool)
- Segment        : id, start(초), end(초), text(발화 텍스트), stressed(bool)
- AnnotationEmphasis : id, type(underline/circle/arrow/bracket/cross/handwritten_text), target_content, score, confidence, timestamp_sec
- Concept        : id, name

관계 (방향 중요):
- (Video)-[:HAS_SLIDES]->(Slides)
- (Video)-[:HAS_SCENES]->(Scenes)
- (Slides)-[:CONTAINS]->(Slide)
- (Scenes)-[:CONTAINS]->(Scene)
- (Slide)-[:HAS_SCENE]->(Scene)
- (Scene)-[:HAS_SEGMENT]->(Segment)    ← Segment는 반드시 Scene을 통해 접근
- (Slide)-[:HAS_ANNOTATION]->(AnnotationEmphasis)
- (Segment)-[:REFERS_TO]->(AnnotationEmphasis)
- (Segment)-[:MENTIONS]->(Concept)       ← Segment가 Concept을 가리킴
- (Slide)-[:APPEARS_IN]->(Concept)       ← Slide가 Concept을 가리킴
- (Concept)-[:is_a|part_of|implements|abstracts|prerequisite_of|uses|calls|compared_to|extends|replaces|solves|optimizes]->(Concept)

개념 관계 방향 예시 (part_of는 하위→상위):
  파일 시스템 관리 -[:part_of]-> 운영체제 기능 -[:part_of]-> 운영체제
  즉 상위 개념의 구성요소를 찾으려면: MATCH (sub)-[:part_of]->(parent) WHERE parent.name CONTAINS '키워드'

자주 틀리는 패턴 (절대 사용 금지):
- (Segment)-[:APPEARS_IN]->(...) ← APPEARS_IN은 Slide만 사용
- (...)-[:MENTIONS]->(Slide)     ← MENTIONS는 Segment→Concept만
- (Concept)-[:APPEARS_IN]->(Slide) ← 방향 반대

올바른 조회 패턴:
- 개념을 언급한 세그먼트: MATCH (seg:Segment)-[:MENTIONS]->(c:Concept) WHERE c.name CONTAINS '키워드'
- 개념이 등장한 슬라이드: MATCH (s:Slide)-[:APPEARS_IN]->(c:Concept) WHERE c.name CONTAINS '키워드'
- 세그먼트가 속한 슬라이드: MATCH (s:Slide)-[:HAS_SCENE]->(sc:Scene)-[:HAS_SEGMENT]->(seg:Segment)

참고:
- start/end는 영상 내 시간(초 단위)
- role이 'core'인 슬라이드가 핵심 내용 슬라이드
- stressed=true인 Segment/Scene은 강조 발화 구간
- emphasis_total이 높을수록 강조가 많이 된 슬라이드
"""

CYPHER_SYSTEM_PROMPT = f"""
너는 강의 영상 분석 그래프 DB를 조회하는 Cypher 전문가야.
아래 스키마를 기반으로 사용자 질문에 맞는 Cypher 쿼리를 생성해.

{GRAPH_SCHEMA}

규칙:
1. Cypher 쿼리만 출력. 설명 없이.
2. 코드 블록(```cypher ... ```) 안에 작성.
3. 결과는 항상 RETURN에 명시적으로 포함.
4. 노드/관계 조회 시 필요한 프로퍼티만 반환 (전체 노드 반환 지양).
5. 타임스탬프가 필요한 경우 Segment.start, Segment.end, Scene.start, Scene.end 활용.
6. LIMIT은 적절히 사용 (기본 20).
"""

ANSWER_SYSTEM_PROMPT = """
너는 강의 영상 분석 결과를 바탕으로 질문에 직접 답하는 어시스턴트야.

답변 형식:
1. 질문에 대한 직접적인 답변을 먼저 제시한다. (2~4문장)
2. 필요한 경우 핵심 근거만 간결하게 덧붙인다. (슬라이드 번호 또는 시간 표기)
3. 조회된 데이터를 그대로 나열하지 않는다.
4. 정의, 음성 발췌, 슬라이드 목록 같은 섹션을 만들지 않는다.
5. 질문과 무관한 내용은 포함하지 않는다.
6. 한국어로 답변한다.

나쁜 예: "운영체제의 정의: ..., 음성 발췌: ..., 슬라이드 목록: ..."
좋은 예: "운영체제가 독점적 권한을 가져야 하는 이유는 ... 때문입니다. (슬라이드 8)"
"""

# ============================================================================
#  요청/응답 모델
# ============================================================================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer:     str
    cypher:     str
    timestamps: list[dict]   # [{label, start, end}]
    graph:      dict         # {nodes, edges}
    raw_results: list[dict]

# ============================================================================
#  Neo4j / Gemini 클라이언트
# ============================================================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
gemini = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================================
#  핵심 로직
# ============================================================================

def _call_gemini(contents: str, system_instruction: str) -> str:
    """Gemini 호출 + 429 재시도."""
    import time
    last_err = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config={"system_instruction": system_instruction},
            )
            return response.text
        except Exception as e:
            last_err = e
            if '429' not in str(e) and 'RESOURCE_EXHAUSTED' not in str(e):
                raise
    raise last_err


# Cypher 생성과 답변 생성을 단일 호출로 통합
COMBINED_SYSTEM_PROMPT = f"""
너는 강의 영상 분석 지식 그래프를 조회하고 결과를 설명하는 어시스턴트야.
이 그래프는 두 레이어로 구성된다:

■ 구조 레이어 (Structure Layer) — 영상/슬라이드 물리적 구조
  용도: "몇 번 슬라이드에서 나왔어?", "어느 시점에?", "얼마나 강조됐어?" 같은 질문
  핵심 노드: Video, Slides, Scenes, Slide, Scene, Segment, AnnotationEmphasis

■ 개념 레이어 (Concept Layer) — 강의 내용과 지식 관계
  용도: "A가 뭐야?", "A와 B의 관계는?", "핵심 개념은?" 같은 질문
  핵심 노드: Concept
  핵심 관계: is_a, part_of, implements, uses, prerequisite_of 등 12종

질문 유형에 따라 적절한 레이어를 선택:
- 내용/개념 질문 → Concept 노드와 개념 간 관계(part_of, implements 등) + MENTIONS로 연결된 Segment.text + APPEARS_IN으로 연결된 Slide.slide_text 조회
- 구조/위치 질문 → Slide, Scene, Segment 위주로 조회
- 복합 질문 → 두 레이어를 JOIN해서 조회

Cypher 작성 규칙:
- RETURN은 쿼리 맨 끝에 한 번만
- UNION 사용 시 각 서브쿼리에 동일한 컬럼명 사용
- WHERE 절에서 개념 검색 시 CONTAINS 사용 (정확한 이름 모를 때)

{GRAPH_SCHEMA}

Slide 노드의 slide_text 프로퍼티에는 슬라이드 원문 텍스트가 있음.
Segment 노드의 text 프로퍼티에는 해당 구간 음성 전사 텍스트가 있음.
내용 질문에는 이 텍스트도 적극 활용할 것.

출력 형식 (반드시 준수):
CYPHER:
```cypher
<쿼리>
```

ANSWER:
<한국어 답변. 슬라이드 번호·시간 등 출처 포함.>
"""


def generate_cypher(question: str) -> str:
    text = _call_gemini(f"질문: {question}\n\n단계1: Cypher만 생성해줘.", COMBINED_SYSTEM_PROMPT)
    m = re.search(r'```(?:cypher)?\s*(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_cypher(cypher: str) -> list[dict]:
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def run_cypher_safe(cypher: str, question: str) -> tuple[list[dict], str]:
    """Cypher 실행 + 문법 오류 시 Gemini에 수정 요청 후 재시도."""
    try:
        return run_cypher(cypher), cypher
    except Exception as e:
        if 'SyntaxError' not in str(e) and 'TypeError' not in str(e):
            raise
        fix_prompt = (
            f"다음 Cypher 쿼리에 문법 오류가 있다:\n```cypher\n{cypher}\n```\n"
            f"오류: {e}\n\n올바른 Cypher로 수정해줘. 코드 블록(```cypher```)만 출력."
        )
        fixed_text = _call_gemini(fix_prompt, COMBINED_SYSTEM_PROMPT)
        m = re.search(r'```(?:cypher)?\s*(.*?)```', fixed_text, re.DOTALL)
        fixed_cypher = m.group(1).strip() if m else fixed_text.strip()
        return run_cypher(fixed_cypher), fixed_cypher


# ── 질문 유형 분류 ────────────────────────────────────────────────────────────

STRUCTURAL_KEYWORDS = ['몇 번', '슬라이드', '시간', '타임', '언제', '어디서', '강조',
                       '몇 초', '구간', '씬', '장면', 'scene', '몇 분']

def classify_question(question: str) -> str:
    """'structural' | 'content' 반환"""
    for kw in STRUCTURAL_KEYWORDS:
        if kw in question:
            return 'structural'
    return 'content'


STOPWORDS = {
    '무엇', '무슨', '어떤', '왜', '어떻게', '언제', '어디', '누가', '몇',
    '이유', '방법', '설명', '의미', '정의', '개념', '차이', '관계',
    '있나요', '있어요', '인가요', '인지', '하는', '하나요', '해줘', '알려줘',
    '대해', '대한', '위한', '위해', '그리고', '또는', '하지만', '그러나',
}

def extract_keywords_from_question(question: str) -> list[str]:
    """
    Gemini 없이 규칙 기반으로 모든 키워드 추출.
    1순위: 따옴표/강조 표시 안 단어
    2순위: 2글자 이상 명사 (조사 제거)
    """
    keywords = []

    # 1. 따옴표 안 단어 (모두 포함)
    quoted = re.findall(
        r"['\u2018\u2019\u201c\u201d\u300c\u300d]"
        r"([^'\u2018\u2019\u201c\u201d\u300c\u300d]+)"
        r"['\u2018\u2019\u201c\u201d\u300c\u300d]",
        question
    )
    keywords.extend(quoted)

    # 2. 조사/어미 제거 후 2글자 이상 토큰
    clean = re.sub(r"['\u2018\u2019\u201c\u201d\u300c\u300d]", ' ', question)
    clean = re.sub(r'[?？!！.,，。·]', ' ', clean)
    tokens = clean.split()
    for tok in tokens:
        # 한국어 조사 제거
        tok = re.sub(r'(은|는|이|가|을|를|의|에|에서|에게|로|으로|와|과|도|만|까지|부터|처럼|이란|란|이란|란|으로서|로서)$', '', tok)
        tok = tok.strip()
        if len(tok) >= 2 and tok not in STOPWORDS and tok not in keywords:
            keywords.append(tok)

    # 중복 제거, 순서 유지
    seen = set()
    result = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result


# ── 고정 쿼리 템플릿 (내용 질문용) ───────────────────────────────────────────

def _relevance_score(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def run_content_queries(keywords: list[str]) -> tuple[dict, str]:
    """
    내용 질문용 멀티 쿼리. 키워드 수 제한 없이 전체 실행.
    결과는 관련성(키워드 포함 개수) 기준 정렬 후 상위만 컨텍스트 전달.
    """
    results: dict = {}
    cyphers: list = []
    seen_segs: set = set()
    seen_slides: set = set()

    for kw in keywords:
        c1 = f"MATCH (sub:Concept)-[:part_of|is_a|implements]->(c:Concept) WHERE c.name CONTAINS '{kw}' RETURN sub.name AS sub_concept, c.name AS parent_concept LIMIT 20"
        for r in run_cypher(c1):
            results.setdefault('sub_concepts', []).append(r)
        cyphers.append(c1)

        c2 = (f"MATCH (slide:Slide)-[:HAS_SCENE]->(scene:Scene)-[:HAS_SEGMENT]->(seg:Segment)"
              f"-[:MENTIONS]->(c:Concept) WHERE c.name CONTAINS '{kw}' "
              f"RETURN seg.text AS segment_text, seg.start AS start, seg.end AS end, "
              f"slide.slide_number AS slide_number, c.name AS concept ORDER BY seg.start LIMIT 15")
        for r in run_cypher(c2):
            key = (r.get('slide_number'), r.get('start'))
            if key not in seen_segs:
                seen_segs.add(key)
                results.setdefault('segments', []).append(r)
        cyphers.append(c2)

        c3 = (f"MATCH (slide:Slide)-[:APPEARS_IN]->(c:Concept) WHERE c.name CONTAINS '{kw}' "
              f"RETURN slide.slide_number AS slide_number, slide.title AS title, "
              f"slide.slide_text AS slide_text ORDER BY slide.slide_number LIMIT 5")
        for r in run_cypher(c3):
            if r.get('slide_number') not in seen_slides:
                seen_slides.add(r.get('slide_number'))
                results.setdefault('slides', []).append(r)
        cyphers.append(c3)

        c4 = (f"MATCH (slide:Slide) WHERE slide.slide_text CONTAINS '{kw}' "
              f"RETURN slide.slide_number AS slide_number, slide.title AS title, "
              f"slide.slide_text AS slide_text ORDER BY slide.slide_number LIMIT 5")
        for r in run_cypher(c4):
            if r.get('slide_number') not in seen_slides:
                seen_slides.add(r.get('slide_number'))
                results.setdefault('slides', []).append(r)
        cyphers.append(c4)

        c5 = (f"MATCH (slide:Slide)-[:HAS_SCENE]->(scene:Scene)-[:HAS_SEGMENT]->(seg:Segment) "
              f"WHERE seg.text CONTAINS '{kw}' "
              f"RETURN seg.text AS segment_text, seg.start AS start, seg.end AS end, "
              f"slide.slide_number AS slide_number ORDER BY seg.start LIMIT 15")
        for r in run_cypher(c5):
            key = (r.get('slide_number'), r.get('start'))
            if key not in seen_segs:
                seen_segs.add(key)
                results.setdefault('segments', []).append(r)
        cyphers.append(c5)

    if results.get('segments'):
        results['segments'].sort(
            key=lambda r: _relevance_score(r.get('segment_text', ''), keywords),
            reverse=True
        )
    if results.get('slides'):
        results['slides'].sort(
            key=lambda r: _relevance_score(r.get('slide_text', '') + r.get('title', ''), keywords),
            reverse=True
        )

    return results, '\n\n'.join(dict.fromkeys(cyphers))


def generate_answer(question: str, raw_results: list[dict], cypher: str,
                    structured: dict = None, keywords: list[str] = None) -> str:
    keywords = keywords or []
    if structured:
        context_parts = [
            f"질문: {question}\n",
            "아래는 지식 그래프에서 조회한 관련 데이터다. 이를 바탕으로 질문에 직접 답해.\n"
        ]
        if structured.get('sub_concepts'):
            subs = list(dict.fromkeys(r.get('sub_concept','') for r in structured['sub_concepts']))
            context_parts.append(f"[구성 개념] {', '.join(subs[:15])}")

        if structured.get('segments'):
            # 관련성 높은 상위 8개만
            top_segs = structured['segments'][:8]
            context_parts.append("\n[음성 발췌]")
            for r in top_segs:
                t = r.get('start', 0)
                context_parts.append(f"  {int(t)//60}:{int(t)%60:02d} | {r.get('segment_text','')}")

        if structured.get('slides'):
            # 관련성 높은 상위 3개만, slide_text는 400자로 제한
            context_parts.append("\n[슬라이드 본문]")
            for r in structured['slides'][:3]:
                context_parts.append(
                    f"  슬라이드 {r.get('slide_number')} '{r.get('title','')}'\n"
                    f"  {r.get('slide_text','')[:400]}"
                )
        context = '\n'.join(context_parts)
    else:
        context = f"질문: {question}\n\n조회 결과:\n{json.dumps(raw_results[:20], ensure_ascii=False, indent=2)}"

    return _call_gemini(context, ANSWER_SYSTEM_PROMPT)


def extract_timestamps(results: list[dict]) -> list[dict]:
    """결과에서 start/end 타임스탬프 추출."""
    timestamps = []
    for row in results:
        for key, val in row.items():
            if isinstance(val, dict):
                start = val.get('start') or val.get('start_sec')
                end   = val.get('end')   or val.get('end_sec')
                label = val.get('text') or val.get('title') or val.get('name') or key
                if start is not None:
                    timestamps.append({'label': str(label)[:50], 'start': start, 'end': end or start})
            elif key in ('start', 'start_sec') and val is not None:
                end_val = row.get('end') or row.get('end_sec') or val
                label   = row.get('text') or row.get('title') or row.get('name') or '구간'
                timestamps.append({'label': str(label)[:50], 'start': val, 'end': end_val})
    # 중복 제거 및 정렬
    seen = set()
    unique = []
    for t in sorted(timestamps, key=lambda x: x['start']):
        key = (t['start'], t['end'])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique[:20]


def build_graph_data(results: list[dict]) -> dict:
    """결과에서 vis.js용 nodes/edges 생성."""
    nodes = {}
    edges = []

    NODE_COLORS = {
        'Concept':           '#FF6B6B',
        'Slide':             '#4ECDC4',
        'Scene':             '#45B7D1',
        'Segment':           '#96CEB4',
        'AnnotationEmphasis':'#FFEAA7',
        'Video':             '#DDA0DD',
        'default':           '#B0C4DE',
    }

    def add_node(node_id: str, label: str, node_type: str = 'default', props: dict = None):
        if node_id not in nodes:
            color = NODE_COLORS.get(node_type, NODE_COLORS['default'])
            title = json.dumps(props, ensure_ascii=False) if props else ''
            nodes[node_id] = {'id': node_id, 'label': label[:30], 'color': color,
                               'title': title, 'type': node_type}

    for row in results:
        for key, val in row.items():
            if isinstance(val, dict) and 'id' in val:
                nid    = val['id']
                ntype  = val.get('labels', ['default'])[0] if val.get('labels') else 'default'
                nlabel = val.get('name') or val.get('title') or val.get('text', '')[:20] or nid
                add_node(nid, nlabel, ntype, val)
            elif isinstance(val, str) and val.startswith(('concept/', 'slide_', 'segment/', 'annotation/')):
                prefix = val.split('/')[0]
                type_map = {'concept': 'Concept', 'slide': 'Slide',
                            'segment': 'Segment', 'annotation': 'AnnotationEmphasis'}
                ntype = type_map.get(prefix, 'default')
                add_node(val, val, ntype)
            elif isinstance(val, (int, float, str)) and key in ('name', 'title', 'text'):
                pass  # 단순 스칼라는 노드로 추가 안 함

        # 관계 추출 시도
        keys = list(row.keys())
        if len(keys) >= 2:
            for i in range(len(keys) - 1):
                src = row.get(keys[i])
                tgt = row.get(keys[i + 1])
                if isinstance(src, dict) and isinstance(tgt, dict):
                    if 'id' in src and 'id' in tgt:
                        edges.append({'from': src['id'], 'to': tgt['id'],
                                      'label': keys[i + 1]})

    return {'nodes': list(nodes.values()), 'edges': edges}


# ============================================================================
#  FastAPI 앱
# ============================================================================

app = FastAPI(title="GraphLEC Query System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")
    try:
        q_type = classify_question(req.question)
        print(f"\n[TYPE] {q_type} | {req.question}")

        if q_type == 'content':
            keywords = extract_keywords_from_question(req.question)
            print(f"[KEYWORDS] {keywords}")
            structured_results, cypher = run_content_queries(keywords)
            raw_results = []
            for v in structured_results.values():
                raw_results.extend(v)
            print(f"[RESULTS] {len(raw_results)}개\n")
            answer = generate_answer(req.question, raw_results, cypher,
                                     structured=structured_results, keywords=keywords)
        else:
            cypher = generate_cypher(req.question)
            raw_results, cypher = run_cypher_safe(cypher, req.question)
            print(f"[RESULTS] {len(raw_results)}개\n")
            answer = generate_answer(req.question, raw_results, cypher)
        timestamps = extract_timestamps(raw_results)
        graph      = build_graph_data(raw_results)
        timestamps  = extract_timestamps(raw_results)
        graph       = build_graph_data(raw_results)

        return QueryResponse(
            answer=answer,
            cypher=cypher,
            timestamps=timestamps,
            graph=graph,
            raw_results=raw_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema")
async def schema():
    return {"schema": GRAPH_SCHEMA}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# ============================================================================
#  프론트엔드
# ============================================================================

def _load_html() -> str:
    html_path = Path(__file__).parent / 'templates' / 'index.html'
    return html_path.read_text(encoding='utf-8')

HTML_PAGE = _load_html()


# ============================================================================
#  메인
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GraphLEC 질의 시스템")
    parser.add_argument("--stem",   default="lecture", help="강의 stem")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  GraphLEC Query System")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()