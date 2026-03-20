"""
슬라이드 통합 텍스트 생성 파이프라인

Input:
  - output/slide_textualized.json : slide_textualizer.py 출력 (t1, t1_structure)
  - output/annotation_analysis.json: annotation_analyzer.py 출력 (강조 키워드)

Output:
  - slide_integrated.json: 통합 슬라이드 텍스트

통합 항목:
  - t1                : 슬라이드 원본 텍스트
  - t1_structure      : 다이어그램/표/화살표 구조
  - emphasized        : 강조된 텍스트 목록
      {text, type, confidence, emphasis_count,
       emphasis_weight_sum, cross_slide_discount}
      - emphasis_count      : 강조 등장 횟수 (정수) — t1 마킹 레벨 결정용
      - emphasis_weight_sum : 가중 누적 점수 (float)
                              callout 타입은 0.3, 나머지는 1.0 적용
      - cross_slide_discount: 전체 슬라이드 대비 등장 비율 기반 감쇠 계수
                              BACKGROUND_THRESHOLD 초과 시 < 1.0
                              fusion.py에서 emphasis_weight_sum * cross_slide_discount 사용
  - t1_integrated     : t1 텍스트에 강조 마킹이 반영된 최종 통합 텍스트

Usage:
    python slide_integrator.py
    python slide_integrator.py --extracted output/slide_textualized.json \\
                               --annotated output/annotation_analysis.json \\
                               --output    output/slide_integrated.json
"""

import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

# 통합 대상에서 제외할 confidence 수준
EXCLUDE_CONFIDENCE = {"low"}

# target_type이 이것 이외인 경우 강조 대상에서 제외
VALID_TARGET_TYPES = {"text", "diagram_element"}

# cross-slide 배경 키워드 discount 임계값
# 전체 슬라이드 중 이 비율을 초과해 등장하는 키워드는 discount 적용
# (예: 0.4 → 40% 초과 시부터 선형 감쇠)
BACKGROUND_THRESHOLD = 0.4


# ============================================================================ #
#  로더                                                                         #
# ============================================================================ #

class ExtractedSlideLoader:
    """slide_textualized.json 로드"""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Dict:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"✓ slide_textualized.json 로드: {len(data['slides'])}개 슬라이드")
        return data


class AnnotationLoader:
    """
    annotation_analysis.json 로드 후 slide_index 기준으로 그룹화.
    한 슬라이드에 annot이 여러 개일 수 있으므로 리스트로 묶음.

    반환: { slide_index: [annotation_result, ...] }
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Dict[int, List[Dict]]:
        with open(self.path, encoding="utf-8") as f:
            raw: List[Dict] = json.load(f)

        grouped: Dict[int, List[Dict]] = {}
        for entry in raw:
            idx = entry.get("slide_index")
            if idx is None:
                continue
            grouped.setdefault(idx, []).append(entry)

        total_annots = sum(
            len(e.get("annotations", []))
            for entries in grouped.values()
            for e in entries
        )
        logger.info(
            f"✓ annotation_analysis.json 로드: "
            f"{len(grouped)}개 슬라이드, 총 {total_annots}개 annotation"
        )
        return grouped


# ============================================================================ #
#  통합 로직                                                                    #
# ============================================================================ #

class SlideIntegrator:
    """
    slide_extracted + annotation_analysis → 통합 슬라이드 텍스트 생성.

    강조 출처 구분:
      - "scene" : annotation_analyzer.py 출력 — 교수가 강의 중 직접 그린 필기/강조
      - "slide" : slide_textualizer.py 출력 — 슬라이드 제작 시 삽입된 색상/굵기 강조

    동일 텍스트가 두 채널에서 모두 강조되면 emphasis_count 합산
    (두 채널이 동시에 가리키면 중요도가 더 높다는 신호).
    """

    def integrate(
        self,
        slide: Dict,
        annot_entries: Optional[List[Dict]],
    ) -> Dict:
        t1 = slide.get("t1", "")

        # scene 강조: annotation_analysis.json
        scene_emphasized = self._collect_scene_emphasized(annot_entries)
        # slide 강조: slide_textualized.json의 slide_emphasis
        slide_emphasized = self._collect_slide_emphasized(slide.get("slide_emphasis", []))

        # 두 채널 병합 (동일 텍스트면 emphasis_count 합산)
        emphasized = self._merge_emphasized(scene_emphasized, slide_emphasized)

        t1_integrated = self._mark_emphasized(t1, emphasized)

        return {
            "slide_id":            slide["slide_id"],
            "slide_number":        slide["slide_number"],
            "timestamp":           slide["timestamp"],
            "timestamp_formatted": slide.get("timestamp_formatted", ""),
            "image_path":          slide.get("image_path", ""),
            "title":               slide.get("title", ""),
            "t1":                  t1,
            "t1_structure":        slide.get("t1_structure", ""),
            "emphasized":          emphasized,
            "t1_integrated":       t1_integrated,
            "has_annotation":      len(emphasized) > 0,
        }

    # ── scene 강조 수집 ──────────────────────────────────────────────────── #

    def _collect_scene_emphasized(self, annot_entries):
        """
        annotation_analysis.json에서 scene 강조 수집.
        동일 텍스트 반복 등장 시 emphasis_count 집계.
        bbox(annotation_bbox, target_bbox)도 첫 번째 등장 기준으로 보존.

        scene 강조는 교수가 직접 그린 필기이므로 emphasis_weight = 1.0 고정.
        """
        if not annot_entries:
            return {}

        buckets = {}
        for entry in sorted(annot_entries, key=lambda x: x.get("annot_index", 0)):
            for ann in entry.get("annotations", []):
                if ann.get("confidence") in EXCLUDE_CONFIDENCE:
                    continue
                if ann.get("target_type") not in VALID_TARGET_TYPES:
                    continue
                target = ann.get("target_content")
                if not target:
                    continue

                text = target.strip()
                key  = " ".join(text.lower().split())

                if key not in buckets:
                    buckets[key] = {
                        "text":               text,
                        "source":             "scene",
                        "type":               ann.get("type", "other"),
                        "confidence":         ann.get("confidence", "medium"),
                        "intent":             ann.get("emphasis_intent", ""),
                        "emphasis_count":     0,
                        "emphasis_weight_sum": 0.0,
                        "annotation_bbox":    ann.get("annotation_bbox"),
                        "target_bbox":        ann.get("target_bbox"),
                    }
                buckets[key]["emphasis_count"]      += 1
                buckets[key]["emphasis_weight_sum"] += 1.0   # scene은 항상 1.0

        return buckets

    # ── slide 강조 수집 ──────────────────────────────────────────────────── #

    def _collect_slide_emphasized(self, slide_emphasis):
        """
        slide_textualized.json의 slide_emphasis에서 slide 강조 수집.

        emphasis_weight (slide_textualizer 후처리에서 부여):
          - callout 타입 : 0.3  (말풍선/설명 줄글 — 강조 신호 약함)
          - 그 외         : 1.0
        emphasis_count는 정수 그대로 집계하고 (마킹 레벨 결정용),
        emphasis_weight_sum에 가중치를 누적한다.
        """
        if not slide_emphasis:
            return {}

        buckets = {}
        for item in slide_emphasis:
            text = (item.get("text") or "").strip()
            if not text:
                continue

            key    = " ".join(text.lower().split())
            weight = float(item.get("emphasis_weight", 1.0))

            if key not in buckets:
                buckets[key] = {
                    "text":               text,
                    "source":             "slide",
                    "type":               item.get("type", "other"),
                    "color":              item.get("color"),
                    "confidence":         "high",   # 슬라이드 디자인 요소는 항상 high
                    "emphasis_count":     0,
                    "emphasis_weight_sum": 0.0,
                    "target_bbox":        item.get("bbox"),
                }
            buckets[key]["emphasis_count"]      += 1
            buckets[key]["emphasis_weight_sum"] += weight

        return buckets

    # ── 병합 ────────────────────────────────────────────────────────────── #

    def _merge_emphasized(self, scene_buckets, slide_buckets):
        """
        scene과 slide 강조를 병합.
        동일 키(정규화 텍스트)가 양쪽에 있으면 emphasis_count / emphasis_weight_sum 합산,
        source는 "scene+slide"로 표기.
        cross_slide_discount는 1.0으로 초기화 — IntegrationPipeline에서 2차 패스로 갱신.
        최종 정렬: emphasis_weight_sum 내림차순 → 텍스트 길이 내림차순.
        """
        merged = dict(scene_buckets)

        for key, slide_item in slide_buckets.items():
            if key in merged:
                merged[key]["emphasis_count"]      += slide_item["emphasis_count"]
                merged[key]["emphasis_weight_sum"] += slide_item["emphasis_weight_sum"]
                merged[key]["source"] = "scene+slide"
                if slide_item.get("color"):
                    merged[key]["color"] = slide_item["color"]
                if not merged[key].get("target_bbox") and slide_item.get("target_bbox"):
                    merged[key]["target_bbox"] = slide_item["target_bbox"]
            else:
                merged[key] = slide_item

        # cross_slide_discount 초기화 (2차 패스에서 갱신)
        for item in merged.values():
            item.setdefault("cross_slide_discount", 1.0)

        return sorted(
            merged.values(),
            key=lambda x: (-x["emphasis_weight_sum"], -len(x["text"]))
        )

    # ── 마킹 삽입 ───────────────────────────────────────────────────────── #

    def _mark_emphasized(self, t1, emphasized):
        """
        t1 텍스트에서 강조 대상을 찾아 emphasis_count 기반 마킹 삽입.

        마킹 레벨:
          count == 1  →  *텍스트*
          count 2~3   →  **텍스트**
          count >= 4  →  ***텍스트***

        이미 마킹된 구간은 재마킹하지 않아 중첩 방지.
        """
        if not t1 or not emphasized:
            return t1

        result = t1
        already_marked = set()

        for item in emphasized:
            target = item["text"]
            key    = " ".join(target.lower().split())
            if key in already_marked:
                continue

            count = item.get("emphasis_count", 1)
            if count >= 4:
                marker = "***"
            elif count >= 2:
                marker = "**"
            else:
                marker = "*"

            lower_result = result.lower()
            lower_target = target.lower()
            pos = lower_result.find(lower_target)

            if pos == -1:
                target_normalized = " ".join(target.split())
                pos = lower_result.find(target_normalized.lower())
                if pos != -1:
                    target = target_normalized

            if pos != -1:
                star_count = result[:pos].count("*")
                if star_count % 2 == 1:
                    logger.debug(f"  마킹 중첩 스킵: '{target}'")
                    already_marked.add(key)
                    continue

                original_span = result[pos:pos + len(target)]
                result = (
                    result[:pos]
                    + f"{marker}{original_span}{marker}"
                    + result[pos + len(target):]
                )
                already_marked.add(key)
            else:
                logger.debug(f"  매칭 실패: '{target}'")

        return result


# ============================================================================ #
#  파이프라인                                                                   #
# ============================================================================ #

class IntegrationPipeline:
    """통합 파이프라인"""

    def __init__(
        self,
        extracted_path: Path,
        annotated_path: Path,
        output_path: Path,
    ):
        self.extracted_path = extracted_path
        self.annotated_path = annotated_path
        self.output_path    = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _apply_cross_slide_discount(integrated_slides: List[Dict]) -> None:
        """
        전체 슬라이드에 걸쳐 등장하는 키워드에 cross-slide discount 적용 (in-place).

        discount 공식:
          coverage = df / N   (df: 해당 키워드가 emphasized에 등장하는 슬라이드 수)
          coverage <= BACKGROUND_THRESHOLD  →  discount = 1.0  (감쇠 없음)
          coverage >  BACKGROUND_THRESHOLD  →  discount = BACKGROUND_THRESHOLD / coverage
            예) N=20, df=16, coverage=0.8 → discount = 0.4/0.8 = 0.5
            예) N=20, df=8,  coverage=0.4 → discount = 1.0 (임계값 경계, 감쇠 없음)

        강의 주제어처럼 거의 모든 슬라이드에 등장하는 키워드는
        0.4/coverage 비율로 자연스럽게 감쇠되어 score를 독식하지 않는다.
        """
        N = len(integrated_slides)
        if N == 0:
            return

        # 전체 슬라이드 순회하며 키워드별 document frequency 집계
        df: Dict[str, int] = {}
        for slide in integrated_slides:
            for item in slide.get("emphasized", []):
                key = " ".join(item["text"].lower().split())
                df[key] = df.get(key, 0) + 1

        # discount 계산 후 각 항목에 기록
        discounts: Dict[str, float] = {}
        for key, freq in df.items():
            coverage = freq / N
            discounts[key] = (
                BACKGROUND_THRESHOLD / coverage
                if coverage > BACKGROUND_THRESHOLD
                else 1.0
            )

        # in-place 업데이트
        discounted_count = 0
        for slide in integrated_slides:
            for item in slide.get("emphasized", []):
                key = " ".join(item["text"].lower().split())
                d = discounts.get(key, 1.0)
                item["cross_slide_discount"] = round(d, 4)
                if d < 1.0:
                    discounted_count += 1

        logger.info(
            f"✓ cross-slide discount 적용: "
            f"{sum(1 for d in discounts.values() if d < 1.0)}개 키워드 감쇠 "
            f"(threshold={BACKGROUND_THRESHOLD}, total_keywords={len(df)})"
        )

    def run(self) -> Dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🔗 슬라이드 통합 텍스트 생성 파이프라인")
        print("="*70)
        print(f"📄 slide_textualized  : {self.extracted_path}")
        print(f"📄 annotation_analysis: {self.annotated_path}")
        print(f"📂 output             : {self.output_path}")

        # Stage 1: 로드
        print("\n" + "-"*70)
        print("Stage 1: 데이터 로드")
        print("-"*70)

        extracted_data  = ExtractedSlideLoader(self.extracted_path).load()
        annot_grouped   = AnnotationLoader(self.annotated_path).load()

        # Stage 2: 통합
        print("\n" + "-"*70)
        print("Stage 2: 슬라이드별 통합")
        print("-"*70)

        integrator = SlideIntegrator()
        integrated_slides = []

        for slide in extracted_data["slides"]:
            slide_num = slide["slide_number"]
            annot_entries = annot_grouped.get(slide_num)

            integrated = integrator.integrate(slide, annot_entries)
            integrated_slides.append(integrated)

            status = f"{len(integrated['emphasized'])}개 강조" if integrated["has_annotation"] else "강조 없음"
            logger.info(
                f"  [{slide_num:03d}] {integrated['title'][:30]:<30} | {status}"
            )

        # Stage 3: cross-slide discount 적용
        print("\n" + "-"*70)
        print("Stage 3: cross-slide 배경 키워드 discount 적용")
        print("-"*70)

        self._apply_cross_slide_discount(integrated_slides)

        # Stage 4: 저장
        print("\n" + "-"*70)
        print("Stage 4: 결과 저장")
        print("-"*70)

        total_time = time.time() - start_time
        annotated_count  = sum(1 for s in integrated_slides if s["has_annotation"])
        total_emphasized = sum(len(s["emphasized"]) for s in integrated_slides)
        scene_count      = sum(
            1 for s in integrated_slides
            for e in s["emphasized"] if e.get("source") in ("scene", "scene+slide")
        )
        slide_count      = sum(
            1 for s in integrated_slides
            for e in s["emphasized"] if e.get("source") in ("slide", "scene+slide")
        )

        result = {
            "metadata": {
                "textualized_path":    str(self.extracted_path),
                "annotated_path":      str(self.annotated_path),
                "processing_time":     total_time,
                "total_slides":        len(integrated_slides),
                "annotated_slides":    annotated_count,
                "total_emphasized":    total_emphasized,
                "scene_emphasized":    scene_count,
                "slide_emphasized":    slide_count,
            },
            "slides": integrated_slides,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {self.output_path}")

        print("\n" + "="*70)
        print("✅ 통합 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 전체 슬라이드:      {len(integrated_slides)}개")
        print(f"  • 강조 있는 슬라이드: {annotated_count}개")
        print(f"  • 총 강조 텍스트:     {total_emphasized}개")
        print(f"    - scene 강조:       {scene_count}개 (교수 필기)")
        print(f"    - slide 강조:       {slide_count}개 (슬라이드 디자인)")
        print(f"\n📁 생성된 파일:")
        print(f"  • {self.output_path}")
        print(f"\n⏱️  처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="슬라이드 통합 텍스트 생성")
    parser.add_argument(
        "--extracted", "-e",
        default="./output/slide_textualized.json",
        help="slide_textualizer.py 출력 경로 (default: ./output/slide_textualized.json)"
    )
    parser.add_argument(
        "--annotated", "-a",
        default="./output/annotation_analysis.json",
        help="annotation_analyzer.py 출력 경로 (default: ./output/nnotation_analysis.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output/slide_integrated.json",
        help="통합 결과 저장 경로 (default: ./output/slide_integrated.json)"
    )

    args = parser.parse_args()

    extracted_path = Path(args.extracted)
    annotated_path = Path(args.annotated)
    output_path    = Path(args.output)

    if not extracted_path.exists():
        print(f"❌ slide_textualized.json not found: {extracted_path}")
        return
    if not annotated_path.exists():
        print(f"❌ annotation_analysis.json not found: {annotated_path}")
        return

    pipeline = IntegrationPipeline(extracted_path, annotated_path, output_path)
    pipeline.run()


if __name__ == "__main__":
    main()