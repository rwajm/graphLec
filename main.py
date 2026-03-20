"""
main.py
=======
강의 영상 분석 파이프라인 전체 실행

단계:
  Stage 1   : slide_extractor     — 영상에서 슬라이드/필기 프레임 추출
  Stage 2-1  : slide_textualizer  — 슬라이드 텍스트 + 슬라이드 강조 추출 (Gemini)
  Stage 2-2  : annotation_analyzer— 교수 필기 강조 분석 (Gemini)
  Stage 3   : slide_integrator    — scene/slide 강조 통합 텍스트 생성

Usage:
    python main.py --input lecture.mp4
    python main.py --input lecture.mp4 --output output/ --slides output_slides/
    python main.py --input lecture.mp4 --skip-extract   # Stage 1 건너뜀 (이미 추출된 경우)
    python main.py --input lecture.mp4 --debug --masks
"""

import sys
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 유틸: 단계 구분 출력
# ──────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _done(label: str, elapsed: float):
    print(f"\n  ✓ {label} 완료  ({elapsed:.1f}초)")
    print("─" * 70)


# ──────────────────────────────────────────────
# 파이프라인
# ──────────────────────────────────────────────

def run_pipeline(args):
    total_start = time.time()
    timings: dict[str, float] = {}

    slides_dir  = Path(args.slides)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    textualized_path  = output_dir / "slide_textualized.json"
    annotation_path   = output_dir / "annotation_analysis.json"
    integrated_path   = output_dir / "slide_integrated.json"

    print("\n" + "═" * 70)
    print("  강의 영상 분석 파이프라인")
    print("═" * 70)
    print(f"  입력 영상 : {args.input}")
    print(f"  슬라이드  : {slides_dir}")
    print(f"  출력      : {output_dir}")

    # ────────────────────────────────────────
    # Stage 1: 슬라이드 추출
    # ────────────────────────────────────────
    if args.skip_extract:
        log.info("Stage 1 건너뜀 (--skip-extract)")
        timings["Stage 1 슬라이드 추출"] = 0.0
    else:
        _banner("Stage 1 / 4  —  슬라이드 추출  (slide_extractor)")
        print(f"  영상: {args.input}  →  {slides_dir}/")

        from slide_extractor import extract_slides

        t0 = time.time()
        metadata = extract_slides(
            input_path=args.input,
            output_dir=str(slides_dir),
            debug=args.debug,
        )
        elapsed = time.time() - t0
        timings["Stage 1 슬라이드 추출"] = elapsed

        slide_count = len({m["slide_index"] for m in metadata})
        frame_count = len(metadata)
        _done(f"슬라이드 {slide_count}개, 프레임 {frame_count}개 추출", elapsed)

    # ────────────────────────────────────────
    # Stage 2-1: 슬라이드 텍스트화
    # ────────────────────────────────────────
    _banner("Stage 2-1 / 4  —  슬라이드 텍스트화  (slide_textualizer)")
    print(f"  {slides_dir}/  →  {textualized_path}")

    from slide_textualizer import TextualizationPipeline, Config as TextConfig

    t0 = time.time()
    text_config = TextConfig(
        slides_dir=slides_dir,
        output_dir=output_dir,
        max_retries=args.retries,
    )
    text_result = TextualizationPipeline(text_config).run()
    elapsed = time.time() - t0
    timings["Stage 2-1 슬라이드 텍스트화"] = elapsed

    n_slides = text_result["metadata"]["total_slides"]
    _done(f"슬라이드 {n_slides}개 텍스트화", elapsed)

    # ────────────────────────────────────────
    # Stage 2-2: 필기 강조 분석
    # ────────────────────────────────────────
    _banner("Stage 2-2 / 4  —  필기 강조 분석  (annotation_analyzer)")
    print(f"  {slides_dir}/  →  {annotation_path}")

    from annotation_analyzer import analyze_all

    t0 = time.time()
    annot_results = analyze_all(
        slides_dir=str(slides_dir),
        output_path=str(annotation_path),
        save_masks=args.masks,
    )
    elapsed = time.time() - t0
    timings["Stage 2-2 필기 강조 분석"] = elapsed

    n_analyzed   = len(annot_results)
    n_annots     = sum(len(r.get("annotations", [])) for r in annot_results)
    _done(f"annot {n_analyzed}개 분석, 총 {n_annots}개 강조 추출", elapsed)

    # ────────────────────────────────────────
    # Stage 3: 통합
    # ────────────────────────────────────────
    _banner("Stage 3 / 4  —  통합 텍스트 생성  (slide_integrator)")
    print(f"  {textualized_path}")
    print(f"  {annotation_path}")
    print(f"  →  {integrated_path}")

    from slide_integrator import IntegrationPipeline

    t0 = time.time()
    integ_result = IntegrationPipeline(
        extracted_path=textualized_path,
        annotated_path=annotation_path,
        output_path=integrated_path,
    ).run()
    elapsed = time.time() - t0
    timings["Stage 3 통합 텍스트 생성"] = elapsed

    n_integrated = integ_result["metadata"]["total_slides"]
    n_emphasized = integ_result["metadata"]["total_emphasized"]
    _done(f"슬라이드 {n_integrated}개 통합, 총 {n_emphasized}개 강조", elapsed)

    # ────────────────────────────────────────
    # 최종 요약
    # ────────────────────────────────────────
    total_elapsed = time.time() - total_start

    print("\n" + "═" * 70)
    print("  ✅ 파이프라인 완료")
    print("═" * 70)
    print("\n  단계별 소요 시간:")
    for stage, t in timings.items():
        if t == 0.0 and "건너뜀" not in stage:
            label = f"  (건너뜀)"
        else:
            label = f"  {t:>7.1f}초"
        print(f"    {stage:<30} {label}")
    print(f"\n    {'총 소요 시간':<30}  {total_elapsed:>7.1f}초")

    print("\n  생성된 파일:")
    for path in [textualized_path, annotation_path, integrated_path]:
        exists = "✓" if path.exists() else "✗"
        print(f"    {exists}  {path}")
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="강의 영상 분석 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --input lecture.mp4
  python main.py --input lecture.mp4 --output out/ --slides slides/
  python main.py --input lecture.mp4 --skip-extract   # 슬라이드 재추출 생략
  python main.py --input lecture.mp4 --debug --masks  # 디버그 + 마스크 저장
        """
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="입력 강의 영상 경로 (.mp4)")
    parser.add_argument("--slides", "-s", default="./output_slides",
                        help="슬라이드 프레임 저장 디렉토리 (default: ./output_slides)")
    parser.add_argument("--output", "-o", default="./output",
                        help="분석 결과 저장 디렉토리 (default: ./output)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Stage 1 건너뜀 — 이미 슬라이드가 추출된 경우")
    parser.add_argument("--retries", type=int, default=3,
                        help="Gemini API 재시도 횟수 (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Stage 1 디버그 로그 출력")
    parser.add_argument("--masks", action="store_true",
                        help="Stage 2-2 diff 마스크 이미지 저장")

    args = parser.parse_args()

    if not args.skip_extract and not Path(args.input).exists():
        print(f"❌ 입력 영상 없음: {args.input}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()