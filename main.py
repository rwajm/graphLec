"""
강의 영상 → 지식그래프 전체 파이프라인 (전처리 기반)

실행 순서:
  Preprocessing: 영상 → Groq Whisper 전사 → pHash 슬라이드 감지 → Gemini 교정 + 병합
  Stage 2:       t1 + t2(오디오) → t3 + text_vector   (integrate_text.py)
  Stage 3:       t3 + 벡터 → 지식그래프               (multimodal_graph.py)

사용법:
  python main.py --video lecture.mp4
  python main.py --video lecture.mp4 --pptx lecture.pptx
  python main.py --video lecture.mp4 --output ./output
"""

import argparse
import json
import time
import sys
from pathlib import Path

from config import PipelineConfig


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    """초 → HH:MM:SS.ss 형식 문자열"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: 전사 → 슬라이드 감지 → 병합
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(cfg: PipelineConfig):
    """
    preprocessing 모듈 3단계를 순차 실행:
      1) transcriber.py  — Groq Whisper로 영상 전사
      2) slide_detector.py — pHash 기반 슬라이드 감지 및 이미지 저장
      3) make_merged_from_detector.py — 전사 교정 + 슬라이드-전사 병합
    """
    print("\n" + "=" * 70)
    print("📋 Preprocessing: 전사 + 슬라이드 감지 + 병합")
    print("=" * 70)

    video_path = Path(cfg.video_path)
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    stem = video_path.stem
    work_dir = Path(".")

    # ── Step 1: Groq Whisper 전사 ──────────────────────────────────────────
    transcript_path = work_dir / f"{stem}_transcribed.json"
    if transcript_path.exists():
        print(f"\n✅ 전사 파일이 이미 존재합니다: {transcript_path} (건너뜀)")
    else:
        print(f"\n🎙️ Step 1/3: Groq Whisper 전사 시작")
        from preprocessing.transcriber import transcribe

        segments = transcribe(str(video_path))
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 전사 완료: {len(segments)}개 세그먼트 → {transcript_path}")

    # ── Step 2: pHash 슬라이드 감지 ─────────────────────────────────────────
    detector_log = Path(cfg.output_dir) / stem / f"{stem}_log.json"
    if detector_log.exists():
        print(f"\n✅ 슬라이드 감지 로그가 이미 존재합니다: {detector_log} (건너뜀)")
    else:
        print(f"\n🖼️ Step 2/3: pHash 슬라이드 감지")
        from preprocessing.slide_detector import run as detect_slides

        detect_slides(str(video_path), cfg.output_dir)
        print(f"  ✅ 슬라이드 감지 완료 → {Path(cfg.output_dir) / stem}/")

    # ── Step 3: 전사 교정 + 병합 ────────────────────────────────────────────
    merged_path = work_dir / f"{stem}_merged.json"
    if merged_path.exists():
        print(f"\n✅ 병합 파일이 이미 존재합니다: {merged_path} (건너뜀)")
    else:
        print(f"\n🔗 Step 3/3: 전사 교정 + 슬라이드-전사 병합")
        from preprocessing.make_merged_from_detector import process

        pptx = cfg.pptx_path if cfg.pptx_path else None
        process(stem, pptx_path=pptx, output_dir=cfg.output_dir, work_dir=".")
        print(f"  ✅ 병합 완료 → {merged_path}")

    return merged_path


# ─────────────────────────────────────────────────────────────────────────────
# 포맷 변환: merged.json → Stage 2/3 입력 형식
# ─────────────────────────────────────────────────────────────────────────────

def convert_merged(merged_path: Path, cfg: PipelineConfig):
    """
    merged.json을 기존 파이프라인(Stage 2, 3)이 소비하는 형식으로 변환.

    생성 파일:
      - slide_extracted_light.json  (Stage 2 입력: t1 + 타임스탬프)
      - slide_extracted.json        (Stage 3 입력: image_vector=null)
      - audio.json                  (Stage 2 입력: 전사 세그먼트 배열)
    """
    print("\n" + "=" * 70)
    print("🔄 포맷 변환: merged.json → 파이프라인 입력 형식")
    print("=" * 70)

    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(cfg.video_path).stem
    slides = merged.get("slides", [])

    # ── slide_extracted_light.json (Stage 2 입력) ──────────────────────────
    light_slides = []
    for s in slides:
        slide_num = s["slide_number"]
        ts_start = s["time_range_seconds"][0]
        slide_id = f"slide_{slide_num:03d}"
        img_dir = output_dir / stem
        img_path = str(img_dir / f"slide_{slide_num:03d}_start.jpg")

        light_slides.append({
            "slide_id": slide_id,
            "slide_number": slide_num,
            "timestamp": ts_start,
            "timestamp_formatted": _format_timestamp(ts_start),
            "image_path": img_path,
            "title": s.get("title", ""),
            "t1": s.get("slide_text", ""),
        })

    light_path = output_dir / "slide_extracted_light.json"
    with open(light_path, "w", encoding="utf-8") as f:
        json.dump({"slides": light_slides}, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {light_path} ({len(light_slides)}개 슬라이드)")

    # ── slide_extracted.json (Stage 3 입력, image_vector=null) ─────────────
    full_slides = []
    for ls in light_slides:
        full_slides.append({**ls, "image_vector": None})

    full_path = output_dir / "slide_extracted.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({"slides": full_slides}, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {full_path} ({len(full_slides)}개 슬라이드)")

    # ── audio.json (Stage 2 입력: 교정된 전사 세그먼트) ────────────────────
    all_segments = []
    for s in slides:
        for seg in s.get("transcript_segments", []):
            all_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
            })
    all_segments.sort(key=lambda x: x["start"])

    audio_path = output_dir / "audio.json"
    with open(audio_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {audio_path} ({len(all_segments)}개 세그먼트)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: 텍스트 통합
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(cfg: PipelineConfig):
    """Stage 2: t1 + 오디오 전사 → t3 통합 텍스트 + text_vector"""
    print("\n" + "=" * 70)
    print("🔗 Stage 2: 텍스트 통합 (슬라이드 + 오디오)")
    print("=" * 70)

    from integrate_text import IntegrationPipeline, Config as Stage2Config

    stage_cfg = Stage2Config(
        google_api_key=cfg.google_api_key,
        slide_json=Path(cfg.output_dir) / "slide_extracted_light.json",
        audio_json=Path(cfg.output_dir) / "audio.json",
        output_dir=Path(cfg.output_dir),
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
    )

    pipeline = IntegrationPipeline(stage_cfg)
    result = pipeline.run()

    print(f"\n✅ Stage 2 완료: {cfg.output_dir}/integrated_text.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: 지식그래프 생성
# ─────────────────────────────────────────────────────────────────────────────

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
    )

    pipeline = GraphPipeline(stage_cfg)
    result = pipeline.run()

    print(f"\n✅ Stage 3 완료: {cfg.output_dir}/knowledge_graph.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="강의 영상 → 지식그래프 전체 파이프라인 (전처리 기반)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v", "--video",
        required=True,
        help="입력 영상 파일 경로 (예: lecture.mp4)",
    )
    parser.add_argument(
        "--pptx",
        default=None,
        help="PPTX 강의 자료 경로 (선택, 없으면 이미지에서 텍스트 추출)",
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="결과 저장 폴더 (기본값: ./output)",
    )
    parser.add_argument(
        "--only",
        type=int,
        choices=[2, 3],
        default=None,
        help="특정 스테이지만 실행 (2: 텍스트 통합, 3: 지식그래프)",
    )

    args = parser.parse_args()

    # ─── 설정 구성 ────────────────────────────────────────────────────────
    cfg = PipelineConfig(
        video_path=args.video,
        output_dir=args.output,
        pptx_path=args.pptx or "",
    )

    total_start = time.time()

    print("\n" + "=" * 70)
    print("🎓 GraphLec - 강의 지식그래프 파이프라인")
    print("=" * 70)
    print(f"  영상 : {cfg.video_path}")
    print(f"  PPTX : {cfg.pptx_path or '(없음 — 이미지에서 텍스트 추출)'}")
    print(f"  출력 : {cfg.output_dir}")

    # ─── 스테이지 실행 ────────────────────────────────────────────────────
    if args.only is not None:
        # 단일 스테이지 실행 (Stage 2 또는 3만)
        stage_fn = {2: run_stage2, 3: run_stage3}
        stage_fn[args.only](cfg)
    else:
        # 전체 파이프라인: Preprocessing → Stage 2 → Stage 3
        merged_path = run_preprocessing(cfg)
        convert_merged(merged_path, cfg)
        run_stage2(cfg)
        run_stage3(cfg)

    # ─── 최종 요약 ────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("🏁 전체 파이프라인 완료!")
    print("=" * 70)
    print(f"  ⏱️  총 처리 시간: {total_time:.1f}초")
    print(f"\n  📁 생성된 주요 파일:")
    print(f"     {cfg.output_dir}/slide_extracted.json      ← t1 + 슬라이드 정보")
    print(f"     {cfg.output_dir}/integrated_text.json      ← t3 + text_vector")
    print(f"     {cfg.output_dir}/knowledge_graph.json      ← 지식그래프")
    print(f"     {cfg.output_dir}/knowledge_graph.html      ← 시각화")
    print(f"\n  💬 Q&A 실행:")
    print(f"     python graph_qa.py -g {cfg.output_dir}/knowledge_graph.json")


if __name__ == "__main__":
    main()
