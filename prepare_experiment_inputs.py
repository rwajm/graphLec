"""
실험 입력 데이터 조립 스크립트
Phase 1: G1 / G2 / G3 입력 데이터 생성

입력:
  - slide_extracted.json  : Stage 1 출력 (t1, t1_structure, timestamp, image_path)
  - audio.json            : Whisper 전사문 (start, end, text 세그먼트 배열)
  - slide_type.json       : 슬라이드 유형 태깅 (diagram_heavy / text_heavy / mixed)

출력 (output_dir 내):
  - g1_integrated_text.json + g1_slide_extracted.json  → G1: Audio Only
  - g2_integrated_text.json + g2_slide_extracted.json  → G2: Textual Hybrid
  - g3_integrated_text.json + g3_slide_extracted.json  → G3: Vision-Augmented

그룹별 변수 통제:
  G1: t3=오디오만,           t1_structure=없음, image_path=없음
  G2: t3=raw_text+오디오,    t1_structure=없음, image_path=없음
  G3: t3=raw_text+오디오,    t1_structure=원본, image_path=원본
"""

import json
import argparse
from pathlib import Path


# ============================================================================ #
#  오디오 세그먼트 → 슬라이드 매핑                                               #
# ============================================================================ #

def assign_audio_to_slides(slides: list, audio_segments: list) -> dict:
    """
    슬라이드 타임스탬프 기준으로 오디오 세그먼트를 슬라이드별로 분리.
    각 슬라이드의 구간: [slide.timestamp, next_slide.timestamp)
    마지막 슬라이드: [slide.timestamp, 마지막 세그먼트 end]
    """
    audio_map = {}  # slide_id → [segment, ...]

    for i, slide in enumerate(slides):
        slide_id = slide["slide_id"]
        start_ts = slide["timestamp"]
        end_ts = slides[i + 1]["timestamp"] if i + 1 < len(slides) else float("inf")

        segments = [
            seg for seg in audio_segments
            if seg["start"] >= start_ts and seg["start"] < end_ts
        ]
        audio_map[slide_id] = segments

    return audio_map


def segments_to_text(segments: list) -> str:
    return " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())


# ============================================================================ #
#  그룹별 입력 조립                                                              #
# ============================================================================ #

def build_integrated_text(slides: list, audio_map: dict, group: str) -> dict:
    """
    group: "g1" | "g2" | "g3"
    integrated_text.json 형식으로 반환.
    """
    result_slides = []

    for slide in slides:
        slide_id = slide["slide_id"]
        segments = audio_map.get(slide_id, [])
        audio_text = segments_to_text(segments)
        raw_text = slide.get("t1", "").strip()
        title = slide.get("title", "")
        slide_number = slide.get("slide_number", 0)

        # t3 조립
        if group == "g1":
            t3 = audio_text

        elif group in ("g2", "g3"):
            parts = []
            if raw_text:
                parts.append(f"[슬라이드 텍스트]\n{raw_text}")
            if audio_text:
                parts.append(f"[강의 오디오]\n{audio_text}")
            t3 = "\n\n".join(parts)

        # image_path: G3만 원본 유지
        image_path = slide.get("image_path", "") if group == "g3" else ""

        result_slides.append({
            "slide_id": slide_id,
            "slide_number": slide_number,
            "title": title,
            "t3": t3,
            "image_path": image_path,
            "has_audio": len(segments) > 0,
            "t2_coverage": len(segments),
        })

    return {
        "metadata": {
            "group": group.upper(),
            "total_slides": len(result_slides),
        },
        "slides": result_slides,
    }


def build_slide_extracted(slides: list, group: str) -> dict:
    """
    slide_extracted.json 형식으로 반환.
    G1, G2: t1_structure 비움 / G3: 원본 유지
    """
    result_slides = []

    for slide in slides:
        result_slides.append({
            "slide_id": slide["slide_id"],
            "slide_number": slide.get("slide_number", 0),
            "timestamp": slide.get("timestamp", 0.0),
            "timestamp_formatted": slide.get("timestamp_formatted", ""),
            "image_path": slide.get("image_path", "") if group == "g3" else "",
            "title": slide.get("title", ""),
            "t1": slide.get("t1", ""),
            "t1_structure": slide.get("t1_structure", "") if group == "g3" else "",
        })

    return {
        "metadata": {
            "group": group.upper(),
            "total_slides": len(result_slides),
        },
        "slides": result_slides,
    }


# ============================================================================ #
#  통계 출력                                                                     #
# ============================================================================ #

def print_stats(group: str, integrated: dict, slide_extracted: dict, audio_map: dict):
    slides = integrated["slides"]
    with_audio = sum(1 for s in slides if s["has_audio"])
    total_segments = sum(s["t2_coverage"] for s in slides)
    has_structure = sum(
        1 for s in slide_extracted["slides"] if s.get("t1_structure")
    )
    has_image = sum(1 for s in slides if s.get("image_path"))
    avg_t3_len = sum(len(s["t3"]) for s in slides) / len(slides) if slides else 0

    print(f"\n[{group.upper()}]")
    print(f"  슬라이드 수:           {len(slides)}")
    print(f"  오디오 있는 슬라이드:  {with_audio} / {len(slides)}")
    print(f"  총 오디오 세그먼트:    {total_segments}")
    print(f"  t1_structure 유지:     {has_structure}")
    print(f"  image_path 유지:       {has_image}")
    print(f"  평균 t3 길이 (chars):  {avg_t3_len:.0f}")


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="G1/G2/G3 실험 입력 데이터 조립")
    parser.add_argument(
        "-s", "--slide",
        default="./output/slide_extracted.json",
        help="Stage 1 출력 (slide_extracted.json)"
    )
    parser.add_argument(
        "-a", "--audio",
        default="./output/audio.json",
        help="Whisper 전사문 JSON"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output/experiment",
        help="출력 디렉토리"
    )
    args = parser.parse_args()

    slide_path = Path(args.slide)
    audio_path = Path(args.audio)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 파일 존재 확인
    for p in [slide_path, audio_path]:
        if not p.exists():
            print(f"❌ 파일 없음: {p}")
            return

    # 로드
    with open(slide_path, "r", encoding="utf-8") as f:
        slide_data = json.load(f)
    slides = slide_data.get("slides", [])

    with open(audio_path, "r", encoding="utf-8") as f:
        audio_segments = json.load(f)

    print(f"✓ 슬라이드: {len(slides)}장")
    print(f"✓ 오디오 세그먼트: {len(audio_segments)}개")

    # 오디오 → 슬라이드 매핑
    audio_map = assign_audio_to_slides(slides, audio_segments)
    mapped = sum(1 for v in audio_map.values() if v)
    print(f"✓ 오디오 매핑된 슬라이드: {mapped} / {len(slides)}")

    # G1 / G2 / G3 생성 및 저장
    for group in ["g1", "g2", "g3"]:
        integrated = build_integrated_text(slides, audio_map, group)
        slide_ext = build_slide_extracted(slides, group)

        int_path = output_dir / f"{group}_integrated_text.json"
        sld_path = output_dir / f"{group}_slide_extracted.json"

        with open(int_path, "w", encoding="utf-8") as f:
            json.dump(integrated, f, indent=2, ensure_ascii=False)

        with open(sld_path, "w", encoding="utf-8") as f:
            json.dump(slide_ext, f, indent=2, ensure_ascii=False)

        print_stats(group, integrated, slide_ext, audio_map)

    print(f"\n✅ 완료. 출력 디렉토리: {output_dir}")
    print("\n다음 단계 (Phase 2) — 지식그래프 3회 생성:")
    for group in ["g1", "g2", "g3"]:
        print(
            f"  python multimodal_graph.py "
            f"-t {output_dir}/{group}_integrated_text.json "
            f"-s {output_dir}/{group}_slide_extracted.json "
            f"-o {output_dir}/{group}_output"
        )


if __name__ == "__main__":
    main()