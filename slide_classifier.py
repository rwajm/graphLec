"""
슬라이드 역할 분류기 (slide_classifier.py)

slide_textualized.json의 모든 필드를 계승하면서
각 슬라이드에 역할(role)과 중복 방문 정보를 추가한
slide_classified.json을 생성한다.

Input:
  - output/slide_textualized.json  : 텍스트화 결과
  - output_slides/metadata.json    : 슬라이드 타임스탬프 + 중복 그룹 정보
  - output/silence.json            : 침묵 구간 정보

Output:
  - output/slide_classified.json

분류 체계:
  role:
    "core"        : 해당 슬라이드의 핵심 설명 구간
    "elaborated"  : 동일 슬라이드의 추가/보충 설명 구간
    "transitional": 스쳐지나가는 구간 (체류 짧음 + 침묵 비율 높음)
    "silent_new"  : 중복 아닌 단독 슬라이드인데 체류가 매우 짧고 침묵이 많음

  revisited (bool)  : 같은 슬라이드 그룹에서 core/elaborated가 2개 이상 — 재방문 슬라이드
  revisit_count (int): 그룹 내 non-transitional 슬라이드 수 (revisited=True일 때만 의미 있음)

분류 점수 (중복 그룹 내 역할 결정용):
  score = speech_ratio × 0.4 + dwell_score × 0.3 + order_score × 0.3
    - speech_ratio : 발화 시간 / 체류 시간  (1 - 침묵 비율)
    - dwell_score  : 체류 시간을 그룹 내 [0, 1] 정규화
    - order_score  : 등장 순서를 그룹 내 [0, 1] 정규화 (나중 등장일수록 높음)

Transitional 판정 기준:
  체류 < 5초                          → transitional 강한 후보
  5~15초이고 silence_ratio > 0.6      → transitional
  15초 초과이고 silence_ratio > 0.8   → transitional
  (그룹 내 score 최하위 + 위 조건 중 하나라도 충족 시 우선 transitional)
"""

import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

# 역할 결정 가중치
W_SPEECH = 0.4
W_DWELL  = 0.3
W_ORDER  = 0.3

# elaborated 최소 점수 (이 이하면 transitional)
ELABORATED_MIN_SCORE = 0.3

# transitional 판정 임계값
TRANSITIONAL_HARD_SEC   = 5.0   # 체류 < 5초: 조건 무관 transitional 후보
TRANSITIONAL_MID_RATIO  = 0.6   # 5~15초 구간: silence_ratio > 0.6 → transitional
TRANSITIONAL_MID_SEC    = 15.0
TRANSITIONAL_LONG_RATIO = 0.8   # 15초 초과: silence_ratio > 0.8 → transitional

# silent_new 판정 (단독 슬라이드)
SILENT_NEW_MAX_SEC   = 5.0
SILENT_NEW_MIN_RATIO = 0.6

# continuous 판정: 첫 방문 종료 ~ 재방문 시작 사이 gap 침묵이 이 값 미만이면 continuous
GAP_CONTINUOUS_MAX_SEC = 1.0


# ============================================================================ #
#  유틸                                                                         #
# ============================================================================ #

def overlap_duration(a_start: float, a_end: float,
                     b_start: float, b_end: float) -> float:
    """두 구간의 겹침 시간 반환."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_duplicate_groups(base_entries: Dict[int, dict]) -> List[Set[int]]:
    """
    metadata의 duplicate_of 관계를 union-find로 묶어 중복 그룹 리스트 반환.
    단독 슬라이드(중복 없음)는 포함하지 않는다.
    """
    parent: Dict[int, int] = {idx: idx for idx in base_entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for idx, entry in base_entries.items():
        for dup_idx in entry.get("duplicate_of", []):
            if dup_idx in parent:
                union(idx, dup_idx)

    # 같은 root끼리 묶기
    groups: Dict[int, Set[int]] = defaultdict(set)
    for idx in base_entries:
        groups[find(idx)].add(idx)

    # 크기 1(단독)은 제외
    return [g for g in groups.values() if len(g) > 1]


def compute_silence_stats(start: float, end: float,
                          silences: List[dict]) -> Tuple[float, float]:
    """
    슬라이드 구간 [start, end]에 겹치는 침묵 총 시간과 비율 반환.
    Returns: (silence_sec, silence_ratio)
    """
    dwell = end - start
    if dwell <= 0:
        return 0.0, 1.0
    total_silence = sum(
        overlap_duration(start, end, s["start"], s["end"])
        for s in silences
    )
    return total_silence, min(total_silence / dwell, 1.0)


def is_transitional(dwell: float, silence_ratio: float) -> bool:
    """체류 시간과 침묵 비율로 transitional 여부 판정."""
    if dwell < TRANSITIONAL_HARD_SEC:
        return True
    if dwell <= TRANSITIONAL_MID_SEC and silence_ratio > TRANSITIONAL_MID_RATIO:
        return True
    if dwell > TRANSITIONAL_MID_SEC and silence_ratio > TRANSITIONAL_LONG_RATIO:
        return True
    return False


# ============================================================================ #
#  분류기                                                                       #
# ============================================================================ #

class SlideClassifier:
    """
    슬라이드별 체류 시간 + 침묵 겹침 + 등장 순서를 종합해 역할을 분류.
    """

    def __init__(self, silences: List[dict]):
        self.silences = silences

    def _slide_stats(self, entry: dict) -> dict:
        """슬라이드 1개의 기본 통계 계산."""
        start = entry["slide_start_sec"]
        end   = entry["slide_end_sec"]
        dwell = end - start
        silence_sec, silence_ratio = compute_silence_stats(start, end, self.silences)
        speech_ratio = max(0.0, 1.0 - silence_ratio)
        return {
            "dwell_sec":     round(dwell, 3),
            "silence_sec":   round(silence_sec, 3),
            "silence_ratio": round(silence_ratio, 4),
            "speech_ratio":  round(speech_ratio, 4),
        }

    def classify_group(self, group: Set[int],
                       base_entries: Dict[int, dict]) -> Dict[int, dict]:
        """
        중복 그룹 내 각 슬라이드에 role / score / revisited / revisit_count 부여.

        revisited 정의:
          비-transitional 슬라이드 중 시간상 첫 번째가 아닌 것 = revisited=True.
          즉 그룹에 core + elaborated가 있을 때,
            - 시간상 앞선 쪽: 첫 방문 (revisited=False)
            - 시간상 뒤쪽 : 재방문 (revisited=True)
          transitional 슬라이드는 revisited 판단 대상 외.

        Returns: {slide_index: classification_dict}
        """
        # 각 슬라이드 통계 계산 (order = slide_start_sec, 시간 순서 기준)
        members = []
        for idx in sorted(group):
            if idx not in base_entries:
                continue
            entry = base_entries[idx]
            stats = self._slide_stats(entry)
            members.append({
                "slide_index":   idx,
                "order":         entry["slide_start_sec"],
                "slide_end_sec": entry["slide_end_sec"],
                **stats,
            })

        if not members:
            return {}

        # 정규화용 최대값
        max_dwell   = max(m["dwell_sec"] for m in members) or 1.0
        min_order   = min(m["order"] for m in members)
        max_order   = max(m["order"] for m in members)
        order_range = (max_order - min_order) or 1.0

        # score 계산
        for m in members:
            dwell_score = m["dwell_sec"] / max_dwell
            order_score = (m["order"] - min_order) / order_range
            m["score"]  = round(
                m["speech_ratio"] * W_SPEECH
                + dwell_score     * W_DWELL
                + order_score     * W_ORDER,
                4
            )

        # ── 1단계: score 기준으로 역할 결정 (transitional 우선 판정) ───────── #
        members.sort(key=lambda x: -x["score"])

        roles: Dict[int, str] = {}
        core_assigned = False

        for m in members:
            trans = is_transitional(m["dwell_sec"], m["silence_ratio"])
            idx   = m["slide_index"]
            if trans:
                roles[idx] = "transitional"
            elif not core_assigned:
                roles[idx] = "core"
                core_assigned = True
            elif m["score"] >= ELABORATED_MIN_SCORE:
                roles[idx] = "elaborated"
            else:
                roles[idx] = "transitional"

        # ── 2단계: revisited — 시간 순 기준, 비-transitional 중 첫 등장 외 ─── #
        members.sort(key=lambda x: x["order"])
        first_non_trans_seen = False
        revisited_map: Dict[int, bool] = {}

        for m in members:
            idx = m["slide_index"]
            if roles[idx] == "transitional":
                revisited_map[idx] = False   # transitional은 재방문 개념 없음
            elif not first_non_trans_seen:
                revisited_map[idx] = False   # 시간상 첫 번째 비-transitional
                first_non_trans_seen = True
            else:
                revisited_map[idx] = True    # 이후 등장 = 재방문

        revisit_count = sum(1 for v in revisited_map.values() if v)

        # ── 3단계: continuous — 첫 방문 종료 ~ 재방문 시작 사이 gap 침묵 계산 ── #
        # revisited=True인 슬라이드에 대해, 바로 직전 비-transitional 방문의 종료
        # 시점부터 현재 슬라이드 시작 시점까지의 gap에 침묵이 GAP_CONTINUOUS_MAX_SEC
        # 미만이면 continuous=True (말이 끊기지 않고 이어진 설명).
        continuous_map: Dict[int, bool] = {}
        prev_non_trans_end: float = None   # 직전 비-transitional 슬라이드의 종료 시점

        for m in members:
            idx = m["slide_index"]
            if roles[idx] == "transitional":
                continuous_map[idx] = False
                continue

            if not revisited_map[idx]:
                # 첫 방문 — continuous 해당 없음
                continuous_map[idx] = False
                prev_non_trans_end = m["slide_end_sec"]
            else:
                # 재방문 — gap 구간 침묵 계산
                if prev_non_trans_end is not None:
                    gap_silence, _ = compute_silence_stats(
                        prev_non_trans_end, m["order"], self.silences
                    )
                    continuous_map[idx] = gap_silence < GAP_CONTINUOUS_MAX_SEC
                else:
                    continuous_map[idx] = False
                prev_non_trans_end = m["slide_end_sec"]

        # ── 4단계: 결과 조립 ───────────────────────────────────────────────── #
        result: Dict[int, dict] = {}
        for m in members:
            idx = m["slide_index"]
            result[idx] = {
                "role":          roles[idx],
                "score":         m["score"],
                "revisited":     revisited_map[idx],
                "revisit_count": revisit_count,
                "continuous":    continuous_map[idx],
            }

        return result

    def classify_standalone(self, idx: int, entry: dict) -> dict:
        """단독 슬라이드(중복 그룹 없음) 분류."""
        stats = self._slide_stats(entry)
        if (stats["dwell_sec"] < SILENT_NEW_MAX_SEC
                and stats["silence_ratio"] > SILENT_NEW_MIN_RATIO):
            role = "silent_new"
        else:
            role = "core"
        return {
            "role":          role,
            "score":         None,
            "revisited":     False,
            "revisit_count": 0,
            "continuous":    False,
        }


# ============================================================================ #
#  파이프라인                                                                   #
# ============================================================================ #

class ClassificationPipeline:

    def __init__(
        self,
        textualized_path: Path,
        metadata_path:    Path,
        silence_path:     Path,
        output_path:      Path,
    ):
        self.textualized_path = textualized_path
        self.metadata_path    = metadata_path
        self.silence_path     = silence_path
        self.output_path      = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🗂️  슬라이드 역할 분류기 (slide_classifier)")
        print("="*70)
        print(f"📄 textualized : {self.textualized_path}")
        print(f"📄 metadata    : {self.metadata_path}")
        print(f"📄 silence     : {self.silence_path}")
        print(f"📂 output      : {self.output_path}")

        # ── 로드 ─────────────────────────────────────────────────────────── #
        print("\n" + "-"*70)
        print("Stage 1: 데이터 로드")
        print("-"*70)

        with open(self.textualized_path, encoding="utf-8") as f:
            tex_data = json.load(f)
        with open(self.metadata_path, encoding="utf-8") as f:
            raw_meta = json.load(f)
        with open(self.silence_path, encoding="utf-8") as f:
            silence_data = json.load(f)

        silences = silence_data.get("silences", [])

        # base 항목만 추출 (slide_index 기준 중복 제거 — base가 여러 개면 첫 번째 우선)
        base_entries: Dict[int, dict] = {}
        for entry in raw_meta:
            if entry.get("capture_type") == "base":
                idx = entry["slide_index"]
                if idx not in base_entries:
                    base_entries[idx] = entry

        logger.info(
            f"✓ 슬라이드: {len(tex_data['slides'])}개 | "
            f"base entries: {len(base_entries)}개 | "
            f"침묵 구간: {len(silences)}개"
        )

        # ── 중복 그룹 구성 ────────────────────────────────────────────────── #
        print("\n" + "-"*70)
        print("Stage 2: 중복 그룹 구성 및 역할 분류")
        print("-"*70)

        dup_groups  = build_duplicate_groups(base_entries)
        grouped_idx: Set[int] = set(idx for g in dup_groups for idx in g)

        logger.info(f"  중복 그룹: {len(dup_groups)}개 "
                    f"(참여 슬라이드: {len(grouped_idx)}개)")

        # ── 분류 ─────────────────────────────────────────────────────────── #
        classifier  = SlideClassifier(silences)
        class_map:  Dict[int, dict] = {}   # slide_index → classification

        for group in dup_groups:
            result = classifier.classify_group(group, base_entries)
            class_map.update(result)

        for idx, entry in base_entries.items():
            if idx not in grouped_idx:
                class_map[idx] = classifier.classify_standalone(idx, entry)

        # ── 통합 출력 ─────────────────────────────────────────────────────── #
        print("\n" + "-"*70)
        print("Stage 3: slide_textualized 계승 + 분류 결과 병합")
        print("-"*70)

        classified_slides = []
        role_counter: Dict[str, int] = defaultdict(int)

        for slide in tex_data["slides"]:
            num   = slide["slide_number"]
            cls   = class_map.get(num, {
                "role":          "core",
                "score":         None,
                "revisited":     False,
                "revisit_count": 0,
                "continuous":    False,
            })

            classified_slide = {
                **slide,                          # textualized 전체 필드 계승
                "role":          cls["role"],
                "score":         cls.get("score"),
                "revisited":     cls.get("revisited", False),
                "revisit_count": cls.get("revisit_count", 0),
                "continuous":    cls.get("continuous", False),
            }
            classified_slides.append(classified_slide)
            role_counter[cls["role"]] += 1

            logger.info(
                f"  slide_{num:03d} | {cls['role']:<12} | "
                f"score={str(round(cls['score'], 3)) if cls.get('score') is not None else '-':>5} | "
                f"revisited={cls.get('revisited', False)} | "
                f"continuous={cls.get('continuous', False)}"
            )

        # ── 저장 ─────────────────────────────────────────────────────────── #
        print("\n" + "-"*70)
        print("Stage 4: 결과 저장")
        print("-"*70)

        total_time = time.time() - start_time
        result = {
            "metadata": {
                **tex_data.get("metadata", {}),
                "silence_path":     str(self.silence_path),
                "processing_time":  total_time,
                "role_counts":      dict(role_counter),
                "thresholds": {
                    "transitional_hard_sec":   TRANSITIONAL_HARD_SEC,
                    "transitional_mid_ratio":  TRANSITIONAL_MID_RATIO,
                    "transitional_mid_sec":    TRANSITIONAL_MID_SEC,
                    "transitional_long_ratio": TRANSITIONAL_LONG_RATIO,
                    "elaborated_min_score":    ELABORATED_MIN_SCORE,
                    "gap_continuous_max_sec":   GAP_CONTINUOUS_MAX_SEC,
                    "weights": {"speech": W_SPEECH, "dwell": W_DWELL, "order": W_ORDER},
                },
            },
            "slides": classified_slides,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {self.output_path}")

        print("\n" + "="*70)
        print("✅ 분류 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 전체 슬라이드: {len(classified_slides)}개")
        for role in ["core", "elaborated", "transitional", "silent_new"]:
            print(f"  • {role:<12}: {role_counter.get(role, 0)}개")
        revisited_count = sum(1 for s in classified_slides if s.get("revisited"))
        print(f"  • revisited    : {revisited_count}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {self.output_path}")
        print(f"\n⏱️  처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  공개 API (main.py에서 호출)                                                  #
# ============================================================================ #

def classify_slides(
    textualized_path: str,
    metadata_path: str,
    silences_path: str,
    output_path: str,
) -> list:
    """ClassificationPipeline 래퍼 — main.py에서 단일 함수로 호출."""
    result = ClassificationPipeline(
        textualized_path=Path(textualized_path),
        metadata_path=Path(metadata_path),
        silence_path=Path(silences_path),
        output_path=Path(output_path),
    ).run()
    return result.get("slides", [])


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    from config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(description="슬라이드 역할 분류기")
    parser.add_argument(
        "--textualized", "-t",
        default=str(DEFAULT_OUTPUT_DIR / "slide_textualized.json"),
        help=f"slide_textualizer.py 출력 경로 (default: {DEFAULT_OUTPUT_DIR}/slide_textualized.json)"
    )
    parser.add_argument(
        "--metadata", "-m",
        default=str(DEFAULT_SLIDES_DIR / "metadata.json"),
        help=f"slide_extractor.py 메타데이터 경로 (default: {DEFAULT_SLIDES_DIR}/metadata.json)"
    )
    parser.add_argument(
        "--silence", "-s",
        required=True,
        help="침묵 구간 JSON 경로 (예: output/lecture_silences.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR / "slide_classified.json"),
        help=f"출력 경로 (default: {DEFAULT_OUTPUT_DIR}/slide_classified.json)"
    )

    args = parser.parse_args()

    for path, label in [
        (args.textualized, "slide_textualized.json"),
        (args.metadata,    "metadata.json"),
        (args.silence,     "silences.json"),
    ]:
        if not Path(path).exists():
            print(f"❌ {label} not found: {path}")
            return

    ClassificationPipeline(
        textualized_path = Path(args.textualized),
        metadata_path    = Path(args.metadata),
        silence_path     = Path(args.silence),
        output_path      = Path(args.output),
    ).run()


if __name__ == "__main__":
    main()