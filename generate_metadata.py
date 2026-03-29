"""
generate_metadata.py
────────────────────
강의 메타데이터 추출 스크립트

입력:
  - {stem}_nodes.parquet
  - {stem}_edges.parquet
  - {stem}_fused.json          (config.output_paths 기준)

출력:
  - metadata/{stem}_metadata.json

사용법:
  python generate_metadata.py --stem os1-1
  python generate_metadata.py --stem os1-1 --title "운영체제의 정의" --instructor "황기태"
  python generate_metadata.py --stem os1-1 --output_dir output --data_dir output
"""

import json
import re
import argparse
from pathlib import Path

import pandas as pd
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ============================================================================
#  STOPWORDS — fusion.py Config.STOPWORDS와 동일하게 유지
# ============================================================================

STOPWORDS: frozenset = frozenset({
    "이", "그", "저", "은", "는", "가", "을", "를", "의", "에", "도",
    "와", "과", "하고", "이고", "이며", "그리고", "그래서", "하지만",
    "또한", "즉", "따라서", "그러나", "또는", "및",
    "있습니다", "있어요", "합니다", "해요", "됩니다", "돼요",
    "입니다", "이에요", "이다", "한다", "된다",
    "것", "거", "수", "때", "더", "많이", "같은", "이런", "그런",
    "개념", "정의", "목표", "목적", "기능", "시작", "발전", "차이",
    "종류", "특징", "핵심", "단어", "강의", "내용", "설명", "이해",
    "개요", "소개", "정리", "비교", "분석", "예시", "문제",
    "실행", "요청", "종료", "생각", "과정", "사용", "제공",
    "처리", "수행", "동작", "발생", "설치", "구현", "관련",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "and", "or", "for", "with", "that", "this", "be", "by",
    "키보드", "마우스", "콘솔", "램", "캐시", "입출력 장치", "모니터", "프린터",
    "컴퓨터", "사용자", "하드웨어", "소프트웨어",
})


# ============================================================================
#  그룹 A — 기본 정보
# ============================================================================

def infer_title(
    top_concepts: list[dict],
    top_keywords: list[str],
    topic_segments: list[dict],
) -> str:
    """top_concepts/keywords/segments로 강의 제목 자동 생성"""
    concept_names = [c["name"] for c in top_concepts[:5]]
    segment_titles = [s["title"] for s in topic_segments[:4]]
    prompt = f"""다음 정보를 바탕으로 강의 제목을 한 줄로 작성해줘.

핵심 개념: {', '.join(concept_names)}
핵심 키워드: {', '.join(top_keywords[:6])}
강의 흐름: {' → '.join(segment_titles)}

조건:
- 10자 내외의 간결한 강의 제목
- '~의 정의', '~의 개념', '~의 기초' 형식 허용
- 제목만 출력 (따옴표, 설명 없이)"""
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.0},
    )
    return response.text.strip().strip('"').strip("'")


def infer_domain(
    nodes: "pd.DataFrame",
    top_concepts: list[dict],
    top_keywords: list[str],
) -> str:
    """
    KG Domain 노드 → broad category 추출 후,
    top_concepts/keywords로 세부 도메인을 Gemini가 추론.
    """
    # KG에서 broad domain 추출
    domain_nodes = nodes[nodes["label"] == "Domain"]
    broad = ""
    if not domain_nodes.empty:
        props = json.loads(domain_nodes.iloc[0]["properties_json"])
        subdomain = props.get("subdomain", "")
        name = props.get("name", "")
        broad = f"{name}/{subdomain}" if subdomain else name

    concept_names = [c["name"] for c in top_concepts[:7]]
    prompt = f"""다음 강의의 도메인을 추론해줘.

KG 추출 카테고리: {broad}
핵심 개념: {', '.join(concept_names)}
핵심 키워드: {', '.join(top_keywords[:8])}

규칙:
- 아래 형식 중 하나로만 출력: cs/operating_system, cs/network, cs/data_structure,
  cs/algorithm, cs/database, cs/software_engineering, math/linear_algebra,
  math/statistics, math/calculus, ml/deep_learning, ml/machine_learning, other
- 형식 문자열만 출력 (설명 없이)"""
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.0},
    )
    return response.text.strip()


def extract_basic_info(fused: dict, overrides: dict) -> dict:
    slides = fused.get("slides", [])
    duration_sec = max((s.get("end_sec", 0) for s in slides), default=0)
    return {
        "video_id":    overrides.get("video_id", Path(fused.get("video_path", "")).stem),
        "instructor":  overrides.get("instructor", ""),
        "duration_sec": round(duration_sec),
        "language":    overrides.get("language", "ko"),
    }


# ============================================================================
#  그룹 B — KG 기반 정보
# ============================================================================

def extract_fused_keyword_scores(fused: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for slide in fused.get("slides", []):
        for kw in slide.get("emphasized_keywords", []):
            k = kw["keyword"]
            scores[k] = scores.get(k, 0) + (
                kw.get("audio_score", 0)
                + kw.get("visual_score", 0)
                + kw.get("annotation_score", 0)
                + kw.get("slide_text_score", 0)
            )
    return scores


def extract_top_concepts(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    fused_kw_scores: dict[str, float],
    top_n: int = 10,
) -> list[dict]:
    concept_nodes = nodes[nodes["label"] == "Concept"]["node_id"].tolist()
    concept_names = {nid: nid.replace("concept/", "") for nid in concept_nodes}

    in_degree  = edges[edges["rel_type"] == "MENTIONS"]["tgt_id"].value_counts().to_dict()
    max_degree = max(in_degree.values(), default=1)
    max_score  = max(fused_kw_scores.values(), default=1)

    results = []
    for nid, name in concept_names.items():
        centrality = round(in_degree.get(nid, 0) / max_degree, 3)
        weight     = round(fused_kw_scores.get(name, 0) / max_score * 5, 3)
        results.append({"name": name, "weight": weight, "centrality": centrality})

    results.sort(key=lambda x: x["weight"] + x["centrality"], reverse=True)
    return results[:top_n]


PREREQUISITE_RELATIONS = {"is_a", "part_of", "uses", "implements", "extends"}
PREREQ_VALID_ENTITY_TYPES = {
    "system", "artifact", "method", "agent", "phenomenon", "metric", "event"
}

def extract_prerequisites(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    top_concept_names: set[str],
    max_prereqs: int = 5,
    min_mention: int = 8,
) -> list[str]:
    concept_nodes = nodes[nodes["label"] == "Concept"].copy()
    concept_nodes["entity_type"] = concept_nodes["properties_json"].apply(
        lambda x: json.loads(x).get("entity_type", "") if x else ""
    )
    valid_node_ids = set(
        concept_nodes[concept_nodes["entity_type"].isin(PREREQ_VALID_ENTITY_TYPES)]["node_id"]
    )

    semantic_edges    = edges[edges["rel_type"].isin(PREREQUISITE_RELATIONS)]
    prereq_candidates = semantic_edges[
        semantic_edges["tgt_id"].isin(valid_node_ids)
    ]["tgt_id"].value_counts()

    mention_counts = edges[edges["rel_type"] == "MENTIONS"]["tgt_id"].value_counts().to_dict()

    results = []
    for nid, ref_count in prereq_candidates.items():
        name = nid.replace("concept/", "")
        if name in top_concept_names:
            continue
        if mention_counts.get(nid, 0) < min_mention:
            continue
        results.append((name, ref_count))

    results.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in results[:max_prereqs]]


def extract_kg_stats(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    return {
        "concept_node_count":  len(nodes[nodes["label"] == "Concept"]),
        "total_node_count":    len(nodes),
        "semantic_edge_count": len(edges[edges["rel_type"].isin(PREREQUISITE_RELATIONS)]),
        "mention_edge_count":  len(edges[edges["rel_type"] == "MENTIONS"]),
    }


def compute_knowledge_density(kg_stats: dict, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return round(min(kg_stats["concept_node_count"] / (duration_sec / 60) / 10.0, 1.0), 3)


# ============================================================================
#  그룹 C — 요약/검색용
# ============================================================================

_JOSA_PATTERN = re.compile(r"(의|와|과|에서|에|은|는|이|가|을|를|로|으로|도|만)$")

def _strip_josa(word: str) -> str:
    return _JOSA_PATTERN.sub("", word)


def extract_top_keywords(fused: dict, top_n: int = 10) -> list[str]:
    TITLE_BOOST = 10.0

    scores: dict[str, dict] = {}
    for slide in fused.get("slides", []):
        for kw in slide.get("emphasized_keywords", []):
            k = kw["keyword"]
            if k not in scores:
                scores[k] = {"total": 0.0, "is_both": False}
            scores[k]["total"] += (
                kw.get("audio_score", 0)
                + kw.get("visual_score", 0)
                + kw.get("annotation_score", 0)
                + kw.get("slide_text_score", 0)
            )
            if set(kw.get("sources", [])) >= {"audio", "visual"}:
                scores[k]["is_both"] = True

    title_word_counts: dict[str, int] = {}
    for slide in fused.get("slides", []):
        for w in re.split(r"[\s,.\-·]+", slide.get("title", "")):
            normalized = _strip_josa(w)
            if len(normalized) >= 2:
                title_word_counts[normalized] = title_word_counts.get(normalized, 0) + 1

    for k in scores:
        scores[k]["total"] += title_word_counts.get(k, 0) * TITLE_BOOST

    for word, count in title_word_counts.items():
        if word in STOPWORDS:
            continue
        if word not in scores:
            scores[word] = {"total": count * TITLE_BOOST, "is_both": False}

    sorted_kw = sorted(
        [(k, v) for k, v in scores.items() if k not in STOPWORDS],
        key=lambda x: (x[1]["is_both"], x[1]["total"]),
        reverse=True,
    )
    return [k for k, _ in sorted_kw[:top_n]]


def extract_topic_segments(fused: dict) -> list[dict]:
    return [
        {
            "slide_idx": s["slide_number"],
            "title":     s["title"],
            "start_sec": round(s.get("start_sec", 0)),
            "end_sec":   round(s.get("end_sec", 0)),
            "role":      s.get("role"),
        }
        for s in fused.get("slides", [])
        if s.get("role") in ("core", "elaborated")
    ]


def generate_summary(
    title: str,
    top_keywords: list[str],
    top_concepts: list[dict],
    topic_segments: list[dict],
) -> str:
    prompt = f"""다음 정보를 바탕으로 강의 요약을 작성해줘.

강의 제목: {title}
핵심 개념: {', '.join(c['name'] for c in top_concepts[:7])}
핵심 키워드: {', '.join(top_keywords[:10])}
강의 흐름: {' → '.join(s['title'] for s in topic_segments)}

조건:
- 3~5문장으로 작성
- 학습자가 이 강의에서 무엇을 배울 수 있는지 명확히 전달
- 강의 흐름(도입→핵심→마무리)을 반영
- 한국어로 작성
- 요약문만 출력 (다른 텍스트 없이)"""

    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


# ============================================================================
#  통합 실행
# ============================================================================

def generate_metadata(
    nodes_path: str,
    edges_path: str,
    fused_path: str,
    output_path: str,
    basic_info_overrides: dict,
    top_concepts_n: int = 10,
    top_keywords_n: int = 10,
) -> dict:

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    with open(fused_path, encoding="utf-8") as f:
        fused = json.load(f)

    print("[1/6] 기본 정보 추출...")
    basic = extract_basic_info(fused, basic_info_overrides)

    print("[2/6] 키워드 점수 집계...")
    fused_kw_scores = extract_fused_keyword_scores(fused)

    print(f"[3/6] 핵심 개념 추출 (top {top_concepts_n})...")
    top_concepts = extract_top_concepts(nodes, edges, fused_kw_scores, top_concepts_n)

    print("[4/6] 선수 지식 추출...")
    prerequisites = extract_prerequisites(nodes, edges, {c["name"] for c in top_concepts})

    print("[5/6] 요약/검색 정보 추출...")
    kg_stats          = extract_kg_stats(nodes, edges)
    knowledge_density = compute_knowledge_density(kg_stats, basic["duration_sec"])
    top_keywords      = extract_top_keywords(fused, top_keywords_n)
    topic_segments    = extract_topic_segments(fused)

    print("[5.5/6] 제목·도메인 자동 추론 (Gemini)...")
    title  = basic_info_overrides.get("title") or infer_title(top_concepts, top_keywords, topic_segments)
    domain = basic_info_overrides.get("domain") or infer_domain(nodes, top_concepts, top_keywords)
    print(f"  title  = {title}")
    print(f"  domain = {domain}")

    print("[6/6] 요약 생성 (Gemini)...")
    summary = generate_summary(
        title=title,
        top_keywords=top_keywords,
        top_concepts=top_concepts,
        topic_segments=topic_segments,
    )

    metadata = {
        "video_id":    basic["video_id"],
        "title":       title,
        "instructor":  basic["instructor"],
        "domain":      domain,
        "duration_sec": basic["duration_sec"],
        "language":    basic["language"],
        "top_concepts":      top_concepts,
        "prerequisites":     prerequisites,
        "knowledge_density": knowledge_density,
        "kg_stats":          kg_stats,
        "summary":          summary,
        "top_keywords":     top_keywords,
        "topic_segments":   topic_segments,
        "difficulty_level": "unclassified",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 메타데이터 저장 완료: {output_path}")
    return metadata


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="강의 메타데이터 생성기")
    parser.add_argument("--stem",         required=True, help="강의 파일 stem (예: os1-1)")
    parser.add_argument("--output_dir",   default="output",       help="parquet/fused 파일 위치")
    parser.add_argument("--slides_dir",   default="output_slides", help="슬라이드 이미지 디렉토리")
    parser.add_argument("--metadata_dir", default="metadata",     help="메타데이터 저장 위치")
    parser.add_argument("--title",        default="",  help="강의명 (미입력 시 Gemini 자동 생성)")
    parser.add_argument("--instructor",   default="",  help="교수자명")
    parser.add_argument("--domain",       default="",  help="도메인 (미입력 시 Gemini 자동 추론)")
    parser.add_argument("--language",     default="ko")
    parser.add_argument("--top_concepts", type=int, default=10)
    parser.add_argument("--top_keywords", type=int, default=10)
    args = parser.parse_args()

    # config.py 패턴: output_paths()가 있으면 사용, 없으면 직접 경로 구성
    try:
        from config import output_paths
        paths = output_paths(args.stem, Path(args.output_dir), Path(args.slides_dir))
        nodes_path = str(Path(args.output_dir) / f"{args.stem}_nodes.parquet")
        edges_path = str(Path(args.output_dir) / f"{args.stem}_edges.parquet")
        fused_path = str(paths["fused"])
    except (ImportError, KeyError):
        nodes_path = str(Path(args.output_dir) / f"{args.stem}_nodes.parquet")
        edges_path = str(Path(args.output_dir) / f"{args.stem}_edges.parquet")
        fused_path = str(Path(args.output_dir) / f"{args.stem}_fused.json")

    output_path = str(Path(args.metadata_dir) / f"{args.stem}_metadata.json")

    metadata = generate_metadata(
        nodes_path  = nodes_path,
        edges_path  = edges_path,
        fused_path  = fused_path,
        output_path = output_path,
        basic_info_overrides = {
            "video_id":   args.stem,
            "title":      args.title,
            "instructor": args.instructor,
            "domain":     args.domain,
            "language":   args.language,
        },
        top_concepts_n = args.top_concepts,
        top_keywords_n = args.top_keywords,
    )

    preview = {k: v for k, v in metadata.items() if k not in ("topic_segments", "summary")}
    preview["topic_segments"] = f"[{len(metadata['topic_segments'])}개]"
    preview["summary"] = metadata["summary"][:80] + "..."
    print(json.dumps(preview, ensure_ascii=False, indent=2))