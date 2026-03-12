"""
슬라이드 통합 텍스트 생성 파이프라인

Input:
  - output/slide_textualized.json : slide_textualizer.py 출력 (t1, t1_structure)
  - output/annotation_analysis.json: annotation_analyzer.py 출력 (강조 키워드)

Output:
  - slide_integrated.json: 통합 슬라이드 텍스트

통합 항목:
  - t1           : 슬라이드 원본 텍스트 (video_extract.py)
  - t1_structure : 다이어그램/표/화살표 구조 (video_extract.py)
  - emphasized   : 강조된 텍스트 목록 [{text, type, confidence, emphasis_count}] (annotation_analyzer.py)
  - t1_integrated: t1 텍스트에 강조 마킹이 반영된 최종 통합 텍스트

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

    강조 텍스트 매핑 전략:
      1. annotation의 target_content를 t1 텍스트에서 substring 매칭
      2. 매칭되면 해당 구간에 **마킹** 삽입 → t1_integrated 생성
      3. 매칭 여부와 무관하게 emphasized 리스트에 수록
    """

    def integrate(
        self,
        slide: Dict,
        annot_entries: Optional[List[Dict]],
    ) -> Dict:
        """
        단일 슬라이드 통합.
        annot_entries: 해당 슬라이드의 annotation_analysis 결과 리스트 (없으면 None)
        """
        t1 = slide.get("t1", "")
        emphasized = self._collect_emphasized(annot_entries)
        t1_integrated = self._mark_emphasized(t1, emphasized)

        return {
            "slide_id":        slide["slide_id"],
            "slide_number":    slide["slide_number"],
            "timestamp":       slide["timestamp"],
            "timestamp_formatted": slide.get("timestamp_formatted", ""),
            "image_path":      slide.get("image_path", ""),
            "title":           slide.get("title", ""),
            "t1":              t1,
            "t1_structure":    slide.get("t1_structure", ""),
            "emphasized":      emphasized,           # 강조 텍스트 원본 목록
            "t1_integrated":   t1_integrated,        # 강조 마킹이 삽입된 최종 텍스트
            "has_annotation":  len(emphasized) > 0,
        }

    # ── 내부 메서드 ──────────────────────────────────────────────────────── #

    def _collect_emphasized(self, annot_entries):
        """
        annotation_analysis 결과에서 유효한 강조 항목 수집.
        동일 텍스트가 여러 annot에 반복 등장하면 emphasis_count로 집계.
        반복 강조일수록 중요도가 높다는 신호.
        """
        if not annot_entries:
            return []

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
                        "text":           text,
                        "type":           ann.get("type", "other"),
                        "confidence":     ann.get("confidence", "medium"),
                        "intent":         ann.get("emphasis_intent", ""),
                        "emphasis_count": 0,
                    }
                buckets[key]["emphasis_count"] += 1

        # emphasis_count 내림차순, 길이 내림차순 정렬 (긴 텍스트 우선 마킹)
        return sorted(buckets.values(), key=lambda x: (-x["emphasis_count"], -len(x["text"])))

    def _mark_emphasized(self, t1, emphasized):
        """
        t1 텍스트에서 강조 대상을 찾아 emphasis_count 기반 마킹 삽입.

        마킹 레벨:
          count == 1  ->  *텍스트*
          count 2~3   ->  **텍스트**
          count >= 4  ->  ***텍스트***

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
                # 이미 마킹 기호(*) 안에 있는지 확인
                star_count = result[:pos].count("*")
                if star_count % 2 == 1:
                    logger.debug(f"  마킹 중첩 스킵: '{target}'")
                    already_marked.add(key)
                    continue

                original_span = result[pos:pos + len(target)]
                result = result[:pos] + f"{marker}{original_span}{marker}" + result[pos + len(target):]
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

        # Stage 3: 저장
        print("\n" + "-"*70)
        print("Stage 3: 결과 저장")
        print("-"*70)

        total_time = time.time() - start_time
        annotated_count   = sum(1 for s in integrated_slides if s["has_annotation"])
        total_emphasized  = sum(len(s["emphasized"]) for s in integrated_slides)

        result = {
            "metadata": {
                "textualized_path": str(self.extracted_path),
                "annotated_path":   str(self.annotated_path),
                "processing_time":  total_time,
                "total_slides":     len(integrated_slides),
                "annotated_slides": annotated_count,
                "total_emphasized": total_emphasized,
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
        print(f"  • 전체 슬라이드:    {len(integrated_slides)}개")
        print(f"  • 강조 있는 슬라이드: {annotated_count}개")
        print(f"  • 총 강조 텍스트:   {total_emphasized}개")
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