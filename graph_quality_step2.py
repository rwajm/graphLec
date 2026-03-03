"""
지식그래프 추출 정밀도 검증 스크립트 (Step 2)

"추출된 개념이 실제 슬라이드에 존재하는가?"를 LLM Judge로 검증합니다.

Input:
  - knowledge_graph.json

샘플링 전략 (Hybrid):
  A. 희소 개념 (Hallucination Hunt) — frequency ≤ 2, 50%
  B. 고밀도 슬라이드 (Noise Check)  — 연결 개념 상위 슬라이드, 30%
  C. 균등 분포 (Baseline)            — 전체 무작위, 20%

출력:
  - 그룹별 정밀도 (Precision)
  - 오탐(False Positive) 유형 분석
  - quality_report_step2.json

Usage:
  python graph_quality_step2.py -g ./output/knowledge_graph.json --api-key $GOOGLE_API_KEY
  python graph_quality_step2.py -g ./output/knowledge_graph.json --api-key $GOOGLE_API_KEY -n 50
  python graph_quality_step2.py -g ./output/knowledge_graph.json --api-key $GOOGLE_API_KEY -o ./output/step2_quality_report.json
"""

import json
import argparse
import os
import time
import random
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
#  샘플링                                                                        #
# ============================================================================ #

def _get_slide_map(graph_data: Dict) -> Dict[str, Dict]:
    """slide_id → slide 데이터 (t3, title 포함)"""
    slides = graph_data.get("slides", [])
    return {s["slide_id"]: s for s in slides}


def _get_concept_node_map(graph: Dict) -> Dict[str, Dict]:
    """concept_id → node"""
    return {n["id"]: n for n in graph.get("nodes", []) if n["type"] == "concept"}


def _get_slide_concept_counts(graph: Dict) -> Dict[str, int]:
    """slide_id → 연결된 개념 수"""
    counts = defaultdict(int)
    for edge in graph.get("edges", []):
        if edge["type"] == "contains":
            counts[edge["from"]] += 1
    return dict(counts)


def sample_pairs(
    graph: Dict,
    n_total: int = 50,
    ratio_a: float = 0.5,
    ratio_b: float = 0.3,
    seed: int = 42
) -> List[Dict]:
    """
    Hybrid 샘플링으로 (concept_id, slide_id, group) 쌍 생성.

    A. 희소 개념 — frequency ≤ 2 (부족 시 ≤ 3까지 자동 확장)
    B. 고밀도 슬라이드 — 개념 수 상위 슬라이드에서 무작위 개념 2개씩
    C. 균등 분포 — 나머지 슬라이드에서 무작위
    """
    random.seed(seed)

    concept_nodes = {n["id"]: n for n in graph.get("nodes", []) if n["type"] == "concept"}
    slide_nodes   = {n["id"]: n for n in graph.get("nodes", []) if n["type"] == "slide"}
    edges         = graph.get("edges", [])

    # slide → concept 매핑
    slide_to_concepts: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if edge["type"] == "contains" and edge["from"] in slide_nodes:
            slide_to_concepts[edge["from"]].append(edge["to"])

    n_a = int(n_total * ratio_a)
    n_b = int(n_total * ratio_b)
    n_c = n_total - n_a - n_b

    samples = []
    used_pairs = set()

    def add_sample(concept_id, slide_id, group):
        key = (concept_id, slide_id)
        if key in used_pairs:
            return False
        used_pairs.add(key)
        node = concept_nodes.get(concept_id, {})
        samples.append({
            "concept_id": concept_id,
            "slide_id": slide_id,
            "group": group,
            "frequency": node.get("frequency", 0),
        })
        return True

    # ── 그룹 A: 희소 개념 ─────────────────────────────────────────────────────
    # frequency ≤ 2 부족 시 ≤ 3까지 자동 확장
    for max_freq in [2, 3, 5]:
        rare = [
            (cid, node) for cid, node in concept_nodes.items()
            if node.get("frequency", 0) <= max_freq
        ]
        if len(rare) >= n_a:
            break

    rare_pool = []
    for cid, node in rare:
        for sid in node.get("slides", []):
            rare_pool.append((cid, sid))
    random.shuffle(rare_pool)

    added_a = 0
    for cid, sid in rare_pool:
        if added_a >= n_a:
            break
        if add_sample(cid, sid, "A_hallucination_hunt"):
            added_a += 1

    # ── 그룹 B: 고밀도 슬라이드 ───────────────────────────────────────────────
    slide_concept_counts = {sid: len(cids) for sid, cids in slide_to_concepts.items()}
    top_slides = sorted(slide_concept_counts, key=lambda s: slide_concept_counts[s], reverse=True)
    # 상위 슬라이드에서 개념 2개씩 (슬라이드 수 × 2 >= n_b 되도록)
    n_top_slides = max(1, (n_b + 1) // 2)
    top_slides = top_slides[:n_top_slides]

    added_b = 0
    for sid in top_slides:
        if added_b >= n_b:
            break
        concepts = slide_to_concepts[sid][:]
        random.shuffle(concepts)
        for cid in concepts[:2]:
            if added_b >= n_b:
                break
            if add_sample(cid, sid, "B_noise_check"):
                added_b += 1

    # ── 그룹 C: 균등 분포 ─────────────────────────────────────────────────────
    all_pairs = []
    for sid, cids in slide_to_concepts.items():
        if sid not in top_slides:  # B 그룹과 중복 슬라이드 제외
            for cid in cids:
                all_pairs.append((cid, sid))
    random.shuffle(all_pairs)

    added_c = 0
    for cid, sid in all_pairs:
        if added_c >= n_c:
            break
        if add_sample(cid, sid, "C_baseline"):
            added_c += 1

    print(f"  샘플 구성: A(희소)={added_a}개, B(고밀도)={added_b}개, C(균등)={added_c}개  합계={len(samples)}개")
    return samples


# ============================================================================ #
#  LLM Judge                                                                    #
# ============================================================================ #

# [변경 1] implied 기준 엄격화
# 기존: "내용상 강하게 암시되면 implied"  →  판단 기준이 모호해 관대한 판정 유발
# 변경: yes = 개념어 또는 동의어가 슬라이드에 직접 등장
#        implied = 개념어는 없지만 슬라이드 제목/핵심 문장이 해당 개념을 정의·설명
#        no = 위 두 경우 모두 해당 없음
# 효과: 100% 정밀도처럼 보이던 과대평가를 줄이고 실제 환각을 포착할 가능성 증가
BATCH_JUDGE_PROMPT = """당신은 강의 자료 품질 검증 전문가입니다.
아래 하나의 슬라이드에 대해 여러 개념이 실제로 존재하는지 판정하세요.

[슬라이드 정보]
슬라이드 ID : {slide_id}
제목        : {title}

[슬라이드 내용]
{content}

[검증 대상 개념 목록]
{concepts_numbered}

각 개념에 대해 아래 기준으로 판정하세요:
- "yes"    : 해당 개념어(또는 명백한 동의어)가 슬라이드 본문에 직접 등장하거나 명시적으로 설명됨
- "implied": 개념어 자체는 없으나, 슬라이드 제목 또는 핵심 문장이 해당 개념을 정의·설명하고 있음
             (단순히 관련 분야라는 이유만으로는 implied 불가)
- "no"     : 슬라이드에서 해당 개념을 찾을 수 없음

반드시 아래 JSON 배열 형식으로만 출력하세요 (다른 텍스트 금지):
[
  {{"index": 1, "verdict": "yes", "evidence": "슬라이드 내 관련 문구 20자 이내", "confidence": "high"}},
  {{"index": 2, "verdict": "no",  "evidence": "없는 이유 한 줄", "confidence": "high"}},
  ...
]"""


def _parse_batch_response(text: str) -> List[Dict]:
    """배치 응답 JSON 파싱 (마크다운 펜스 제거 포함)"""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


def run_judge_batch(
    samples: List[Dict],
    slide_map: Dict[str, Dict],
    api_key: str,
    model: str = "gemini-2.5-flash",
    batch_size: int = 10,
) -> Tuple[List[Dict], Dict]:
    """
    [변경 2] 배치 호출: 같은 슬라이드의 개념들을 한 번에 묶어서 판정
    기존: 샘플 수만큼 개별 호출 (49회)
    변경: 슬라이드별로 묶은 후 batch_size 단위로 호출 (약 5~6회)
    효과: API 호출 횟수 ~90% 감소 → 소요 시간 207초 → 약 20~30초 예상

    [변경 3] Temperature=0 적용
    기존: 기본 temperature (약 1.0) → 동일 입력에 다른 판정 가능
    변경: temperature=0 → 결정론적 출력, 재현 가능한 결과
    효과: 동일 샘플 재실행 시 동일 결과 보장, 판정 일관성 향상
    """
    if not GENAI_AVAILABLE:
        print("❌ google-genai 없음")
        return [], {}

    client = genai.Client(api_key=api_key)

    # 슬라이드별로 샘플 그룹핑
    slide_groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        slide_groups[s["slide_id"]].append(s)

    # 슬라이드 그룹을 batch_size 단위 청크로 분할
    # (하나의 슬라이드에 개념이 많아도 batch_size 초과 시 분할)
    chunks: List[List[Dict]] = []
    current_chunk: List[Dict] = []
    for sid, group in slide_groups.items():
        for item in group:
            current_chunk.append(item)
            if len(current_chunk) >= batch_size:
                chunks.append(current_chunk)
                current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    # index → sample 매핑 (배치 내 순서 추적용)
    results_map: Dict[tuple, Dict] = {}
    total_input = total_output = 0
    t_start = time.time()
    call_count = 0

    for chunk_idx, chunk in enumerate(chunks):
        # 이 청크가 같은 슬라이드면 슬라이드 텍스트를 공유, 다르면 첫 슬라이드 기준
        # → 슬라이드별로 그룹핑했으므로 청크 내 슬라이드가 섞일 수 있음
        # 슬라이드별로 다시 분리해서 호출
        sub_groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in chunk:
            sub_groups[item["slide_id"]].append(item)

        for sid, items in sub_groups.items():
            slide = slide_map.get(sid, {})
            title   = slide.get("title", "")
            content = slide.get("t3", slide.get("t1", ""))[:2000]

            concepts_numbered = "\n".join(
                f"{i+1}. {item['concept_id']}"
                for i, item in enumerate(items)
            )

            prompt = BATCH_JUDGE_PROMPT.format(
                slide_id=sid,
                title=title,
                content=content,
                concepts_numbered=concepts_numbered,
            )

            call_count += 1
            try:
                # [변경 3] Temperature=0
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )

                if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                    m = resp.usage_metadata
                    total_input  += getattr(m, "prompt_token_count", 0)
                    total_output += getattr(m, "candidates_token_count", 0)

                verdicts = _parse_batch_response(resp.text)
                verdict_map = {v["index"]: v for v in verdicts}

                for i, item in enumerate(items):
                    v = verdict_map.get(i + 1, {})
                    verdict = v.get("verdict", "unknown")
                    judged = {
                        **item,
                        "verdict":    verdict,
                        "evidence":   v.get("evidence", ""),
                        "confidence": v.get("confidence", ""),
                        "error":      None,
                    }
                    results_map[(item["concept_id"], item["slide_id"])] = judged
                    icon = {"yes": "✅", "no": "❌", "implied": "🔶", "error": "💥"}.get(verdict, "❓")
                    print(f"  {icon} {item['concept_id'][:35]:<35} [{sid}]  {v.get('evidence','')[:40]}")

            except Exception as e:
                for item in items:
                    judged = {**item, "verdict": "error", "evidence": str(e),
                              "confidence": "", "error": str(e)}
                    results_map[(item["concept_id"], item["slide_id"])] = judged
                    print(f"  💥 {item['concept_id'][:35]:<35} [{sid}]  ERROR: {str(e)[:40]}")

        print(f"  ── 배치 {chunk_idx+1}/{len(chunks)} 완료 (누적 호출 {call_count}회)")

    # 원래 샘플 순서 복원
    results = []
    for s in samples:
        key = (s["concept_id"], s["slide_id"])
        results.append(results_map.get(key, {
            **s, "verdict": "error", "evidence": "결과 없음",
            "confidence": "", "error": "결과 없음"
        }))

    elapsed = time.time() - t_start
    usage = {
        "total_api_calls": call_count,
        "total_samples":   len(results),
        "elapsed_sec":     round(elapsed, 2),
        "input_tokens":    total_input,
        "output_tokens":   total_output,
        "total_tokens":    total_input + total_output,
    }
    return results, usage


# ============================================================================ #
#  정밀도 계산                                                                   #
# ============================================================================ #

def compute_precision(results: List[Dict]) -> Dict:
    """
    전체 및 그룹별 정밀도 계산.
    yes + implied = 정밀도 분자 (개념이 실제로 존재하거나 암시됨)
    """
    groups = ["A_hallucination_hunt", "B_noise_check", "C_baseline"]
    group_results: Dict[str, List] = defaultdict(list)
    for r in results:
        group_results[r["group"]].append(r)

    def precision_of(items):
        valid = [r for r in items if r["verdict"] != "error"]
        if not valid:
            return None
        positive = sum(1 for r in valid if r["verdict"] in ("yes", "implied"))
        return round(positive / len(valid), 4)

    group_stats = {}
    for g in groups:
        items = group_results.get(g, [])
        verdicts = Counter(r["verdict"] for r in items)
        group_stats[g] = {
            "sample_count": len(items),
            "precision":    precision_of(items),
            "verdicts":     dict(verdicts),
        }

    # 오탐 유형 분석: verdict=no인 샘플의 evidence 패턴
    false_positives = [r for r in results if r["verdict"] == "no"]
    fp_by_group = defaultdict(list)
    for r in false_positives:
        fp_by_group[r["group"]].append({
            "concept":  r["concept_id"],
            "slide_id": r["slide_id"],
            "reason":   r.get("evidence", ""),
            "frequency": r.get("frequency", 0)
        })

    return {
        "overall_precision":   precision_of(results),
        "overall_sample_count": len([r for r in results if r["verdict"] != "error"]),
        "overall_verdicts":    dict(Counter(r["verdict"] for r in results)),
        "group_stats":         group_stats,
        "false_positive_count": len(false_positives),
        "false_positives_by_group": dict(fp_by_group),
    }


# ============================================================================ #
#  리포트 출력                                                                   #
# ============================================================================ #

def print_report(precision: Dict, usage: Dict, sampling_config: Dict):
    sep = "=" * 62
    print(f"\n{sep}")
    print("🔍 추출 정밀도 검증 리포트 (Step 2)")
    print(sep)

    p = precision
    overall = p["overall_precision"]
    status = "✅ 양호" if overall and overall >= 0.8 else "⚠️  주의" if overall else "—"
    print(f"\n[전체 정밀도]  {status}")
    print(f"  샘플 수    : {p['overall_sample_count']}개")
    print(f"  정밀도     : {overall*100:.1f}%" if overall else "  정밀도     : 계산 불가")
    v = p["overall_verdicts"]
    print(f"  판정 분포  : yes={v.get('yes',0)}  implied={v.get('implied',0)}  "
          f"no={v.get('no',0)}  error={v.get('error',0)}")

    print(f"\n[그룹별 정밀도]")
    group_labels = {
        "A_hallucination_hunt": "A. 희소 개념 (환각 탐지)",
        "B_noise_check":        "B. 고밀도 슬라이드 (노이즈 체크)",
        "C_baseline":           "C. 균등 분포 (베이스라인)",
    }
    for gkey, glabel in group_labels.items():
        gs = p["group_stats"].get(gkey, {})
        prec = gs.get("precision")
        n    = gs.get("sample_count", 0)
        vd   = gs.get("verdicts", {})
        prec_str = f"{prec*100:.1f}%" if prec is not None else "—"
        status_g = "✅" if prec and prec >= 0.8 else "⚠️" if prec else "—"
        print(f"\n  {status_g} {glabel}")
        print(f"     샘플 수 : {n}  |  정밀도 : {prec_str}")
        print(f"     판정    : yes={vd.get('yes',0)}  implied={vd.get('implied',0)}  "
              f"no={vd.get('no',0)}  error={vd.get('error',0)}")

    # 오탐 분석
    print(f"\n[오탐 분석 (verdict=no)]")
    fp_total = p["false_positive_count"]
    print(f"  오탐 총계 : {fp_total}개")
    for gkey, fps in p["false_positives_by_group"].items():
        if not fps:
            continue
        print(f"\n  [{group_labels.get(gkey, gkey)}]")
        for fp in fps[:5]:
            freq_note = f"(등장 {fp['frequency']}회)"
            print(f"    ✗ '{fp['concept']}' @ {fp['slide_id']} {freq_note}")
            print(f"      이유: {fp['reason'][:80]}")
        if len(fps) > 5:
            print(f"    ... 외 {len(fps)-5}개 (전체 결과는 JSON 참조)")

    # 인사이트
    print(f"\n[인사이트]")
    gs_a = p["group_stats"].get("A_hallucination_hunt", {})
    gs_b = p["group_stats"].get("B_noise_check", {})
    prec_a = gs_a.get("precision")
    prec_b = gs_b.get("precision")
    if prec_a is not None and prec_a < 0.7:
        print(f"  ⚠️  희소 개념 정밀도 낮음({prec_a*100:.0f}%) → 추출 프롬프트에서 환각 방지 강화 필요")
    if prec_b is not None and prec_b < 0.7:
        print(f"  ⚠️  고밀도 슬라이드 정밀도 낮음({prec_b*100:.0f}%) → 최대 개념 수 제한 검토 필요")
    if overall and overall >= 0.85:
        print(f"  ✅ 전체 정밀도 {overall*100:.0f}% — 추출 품질 양호")

    # 성능
    print(f"\n[API 사용량]")
    print(f"  소요 시간  : {usage.get('elapsed_sec', 0)}초")
    print(f"  API 호출   : {usage.get('total_api_calls', 0)}회  (샘플 {usage.get('total_samples',0)}개)")
    print(f"  토큰       : 입력 {usage.get('input_tokens',0)} / "
          f"출력 {usage.get('output_tokens',0)} / "
          f"합계 {usage.get('total_tokens',0)}")

    print(f"\n{sep}")


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="지식그래프 추출 정밀도 검증 (Step 2)")
    parser.add_argument("-g", "--graph",      default="./output/knowledge_graph.json")
    parser.add_argument("-o", "--output",     default=None, help="JSON 리포트 저장 경로")
    # [변경 4] 기본 샘플 수 50 → 100
    # 효과: 통계적 신뢰도 향상 (표본 오차 ±14% → ±10%), 희소 개념 풀이 작아도 충분한 A그룹 확보
    parser.add_argument("-n", "--samples",    type=int, default=100, help="총 샘플 수 (기본 100)")
    parser.add_argument("--api-key",          default=os.getenv("GOOGLE_API_KEY", ""))
    parser.add_argument("--model",            default="gemini-2.5-flash")
    parser.add_argument("--seed",             type=int, default=42, help="샘플링 랜덤 시드")
    parser.add_argument("--ratio-a",          type=float, default=0.5, help="희소 개념 비율 (기본 0.5)")
    parser.add_argument("--ratio-b",          type=float, default=0.3, help="고밀도 슬라이드 비율 (기본 0.3)")
    parser.add_argument("--batch-size",       type=int, default=10, help="슬라이드당 배치 크기 (기본 10)")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ --api-key 또는 GOOGLE_API_KEY 환경변수가 필요합니다.")
        return

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"❌ 파일 없음: {graph_path}")
        return

    # knowledge_graph_fixed.json이 있으면 우선 사용
    fixed_graph_path = graph_path.parent / "knowledge_graph_fixed.json"
    if fixed_graph_path.exists():
        print(f"📂 로드 중: {fixed_graph_path}  (fixed 버전 감지)")
        with open(fixed_graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph_path = fixed_graph_path  # 리포트 source 경로도 fixed로 통일
    else:
        print(f"📂 로드 중: {graph_path}")
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    graph     = data.get("graph", {})
    slide_map = _get_slide_map(data)

    if not slide_map:
        print("❌ knowledge_graph.json에 'slides' 배열이 없습니다. 벡터 포함 버전을 사용하세요.")
        return

    print(f"  노드 {len(graph.get('nodes',[]))}개  |  슬라이드 컨텍스트 {len(slide_map)}개 로드 완료")

    # 샘플링
    print(f"\n⚙️  샘플링 중 (n={args.samples})...")
    samples = sample_pairs(
        graph,
        n_total=args.samples,
        ratio_a=args.ratio_a,
        ratio_b=args.ratio_b,
        seed=args.seed
    )

    # LLM 판정
    print(f"\n🤖 LLM Judge 실행 중 ({len(samples)}쌍, batch_size={args.batch_size})...")
    results, usage = run_judge_batch(
        samples, slide_map, args.api_key, args.model,
        batch_size=args.batch_size
    )

    # 정밀도 계산
    precision = compute_precision(results)

    sampling_config = {
        "n_total":  args.samples,
        "ratio_a":  args.ratio_a,
        "ratio_b":  args.ratio_b,
        "ratio_c":  round(1 - args.ratio_a - args.ratio_b, 2),
        "seed":     args.seed,
        "model":    args.model,
    }

    # 리포트 출력
    print_report(precision, usage, sampling_config)

    # JSON 저장
    report = {
        "source":          str(graph_path),
        "sampling_config": sampling_config,
        "precision":       precision,
        "api_usage":       usage,
        "raw_results":     results,
    }

    out_path = Path(args.output) if args.output else graph_path.parent / "quality_report_step2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 리포트 저장: {out_path}")


if __name__ == "__main__":
    main()