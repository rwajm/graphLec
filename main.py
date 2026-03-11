"""
실행 순서:
  Stage 0: 영상 → 슬라이드 이미지 추출         (main_0.py  - MSESlideDetector)
  Stage 1: 슬라이드 이미지 → t1 + t1_structure  (video_extract.py - ExtractionPipeline)
  Stage 2: t1 + t2(오디오) → t3 + text_vector   (integrate_text.py - IntegrationPipeline)
  Stage 3: t3 + 벡터 → 지식그래프               (multimodal_graph.py - GraphPipeline)

사용법:
  python main.py --video lecture.mp4 --audio audio.json
  python main.py --video lecture.mp4 --audio audio.json --output ./output --threshold 500
  python main.py --skip-stage0 --output ./output  # 슬라이드가 이미 있는 경우
  python main.py --only 3 --synonyms ./cs_synonyms.json  # Stage 3만, 커스텀 동의어 사전
"""

import argparse
import time
import sys
from pathlib import Path

from config import PipelineConfig


def run_stage0(cfg: PipelineConfig):
    """Stage 0: MSE 기반 슬라이드 변화 감지 및 이미지 저장"""
    print("\n" + "=" * 70)
    print("🎬 Stage 0: 영상 → 슬라이드 이미지 추출")
    print("=" * 70)

    from main_0 import MSESlideDetector

    if not Path(cfg.video_path).exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {cfg.video_path}")
        sys.exit(1)

    detector = MSESlideDetector(cfg.video_path)
    result = detector.run(threshold=cfg.mse_threshold, output_dir=cfg.slides_dir)

    print(f"\n✅ Stage 0 완료: {result['count']}개 슬라이드 → {cfg.slides_dir}/")
    return result


def run_stage1(cfg: PipelineConfig):
    """Stage 1: 슬라이드 이미지 → t1 텍스트 + t1_structure"""
    print("\n" + "=" * 70)
    print("🖼️  Stage 1: 슬라이드 이미지 데이터 추출")
    print("=" * 70)

    from video_extract import ExtractionPipeline, Config as Stage1Config

    stage_cfg = Stage1Config(
        google_api_key=cfg.google_api_key,
        slides_dir=Path(cfg.slides_dir),
        output_dir=Path(cfg.output_dir),
        gemini_model=cfg.gemini_model,
    )

    pipeline = ExtractionPipeline(stage_cfg)
    result = pipeline.run()

    print(f"\n✅ Stage 1 완료: {cfg.output_dir}/slide_extracted.json")
    return result


def run_stage2(cfg: PipelineConfig):
    """Stage 2: t1 + 오디오 전사 → t3 통합 텍스트 + text_vector"""
    print("\n" + "=" * 70)
    print("🔗 Stage 2: 텍스트 통합 (슬라이드 + 오디오)")
    print("=" * 70)

    from integrate_text import IntegrationPipeline, Config as Stage2Config

    if not Path(cfg.audio_json).exists():
        print(f"❌ 오디오 JSON 파일을 찾을 수 없습니다: {cfg.audio_json}")
        sys.exit(1)

    # Stage 1 출력인 slide_extracted.json을 직접 읽음
    stage_cfg = Stage2Config(
        google_api_key=cfg.google_api_key,
        slide_json=Path(cfg.output_dir) / "slide_extracted.json",
        audio_json=Path(cfg.audio_json),
        output_dir=Path(cfg.output_dir),
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        alpha=cfg.alpha,
        match_threshold=cfg.match_threshold,
        min_segment_length=cfg.min_segment_length,
        alpha_short_threshold=cfg.alpha_short_threshold,
        alpha_short=cfg.alpha_short,
    )

    pipeline = IntegrationPipeline(stage_cfg)
    result = pipeline.run()

    print(f"\n✅ Stage 2 완료: {cfg.output_dir}/integrated_text.json")
    return result


def run_stage3(cfg: PipelineConfig):
    """Stage 3: t3 + 벡터 → 지식그래프 (JSON + HTML 시각화)"""
    print("\n" + "=" * 70)
    print("🕸️  Stage 3: 지식그래프 생성")
    print("=" * 70)

    from multimodal_graph import GraphPipeline, Config as Stage3Config

    stage_cfg = Stage3Config(
        google_api_key=cfg.google_api_key,
        integrated_text_json=Path(cfg.output_dir) / "integrated_text.json",
        slide_extracted_json=Path(cfg.output_dir) / "slide_extracted.json",
        output_dir=Path(cfg.output_dir),
        gemini_model=cfg.gemini_model,
        synonyms_path=Path(cfg.synonyms_path) if cfg.synonyms_path else None,
    )

    pipeline = GraphPipeline(stage_cfg)
    result = pipeline.run()

    print(f"\n✅ Stage 3 완료: {cfg.output_dir}/knowledge_graph.json")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="강의 영상 → 지식그래프 전체 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v", "--video",
        default=None,
        help="입력 영상 파일 경로 (예: lecture.mp4)"
    )
    parser.add_argument(
        "-a", "--audio",
        default="./audio.json",
        help="Whisper 전사 JSON 파일 경로 (기본값: ./audio.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="결과 저장 폴더 (기본값: ./output)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="MSE 슬라이드 변화 감지 임계값 (기본값: 500, 권장 범위: 500~2000)"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.55,
        help="오디오-슬라이드 매칭 최소 점수 (기본값: 0.55)"
    )
    parser.add_argument(
        "--synonyms",
        default=None,
        help="동의어 사전 JSON 경로 (없으면 기본 사전 사용)"
    )
    parser.add_argument(
        "--skip-stage0",
        action="store_true",
        help="Stage 0 건너뜀 (슬라이드 이미지가 이미 slides/ 폴더에 있는 경우)"
    )
    parser.add_argument(
        "--only",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="특정 스테이지만 실행 (0/1/2/3)"
    )

    args = parser.parse_args()

    cfg = PipelineConfig(
        video_path=args.video or "",
        audio_json=args.audio,
        output_dir=args.output,
        mse_threshold=args.threshold,
        match_threshold=args.match_threshold,
        synonyms_path=args.synonyms,
    )

    total_start = time.time()

    print("\n" + "=" * 70)
    print("🎓 강의 지식그래프 파이프라인")
    print("=" * 70)
    print(f"  영상  : {cfg.video_path or '(건너뜀)'}")
    print(f"  오디오: {cfg.audio_json}")
    print(f"  출력  : {cfg.output_dir}")
    if cfg.synonyms_path:
        print(f"  동의어: {cfg.synonyms_path}")

    only = args.only

    if only is not None:
        stage_fn = {0: run_stage0, 1: run_stage1, 2: run_stage2, 3: run_stage3}
        stage_fn[only](cfg)
    else:
        if not args.skip_stage0:
            if not cfg.video_path:
                print("❌ --video 인자가 필요합니다. (--skip-stage0 옵션으로 건너뛸 수 있습니다)")
                sys.exit(1)
            run_stage0(cfg)

        run_stage1(cfg)
        run_stage2(cfg)
        run_stage3(cfg)

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("🏁 전체 파이프라인 완료!")
    print("=" * 70)
    print(f"  ⏱️  총 처리 시간: {total_time:.1f}초")
    print(f"\n  📁 생성된 주요 파일:")
    print(f"     {cfg.output_dir}/slide_extracted.json      ← t1 + t1_structure")
    print(f"     {cfg.output_dir}/integrated_text.json      ← t3 + text_vector")
    print(f"     {cfg.output_dir}/knowledge_graph.json      ← 지식그래프")
    print(f"     {cfg.output_dir}/knowledge_graph.html      ← 시각화")
    print(f"\n  💬 Q&A 실행:")
    print(f"     python graph_qa.py -g {cfg.output_dir}/knowledge_graph.json")


if __name__ == "__main__":
    main()