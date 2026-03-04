#!/usr/bin/env python3
"""
슬라이드 전환 감지 및 추출기

- 4fps 샘플링으로 빠른 분석 (전체 프레임 디코딩 불필요)
- 1.5초 미만 구간 자동 스킵
- pHash 기반 중복 슬라이드 제거 (뒤로 돌아가는 경우 포함)
- 각 슬라이드마다 시작 프레임(깨끗한) + 종료 프레임(변경분 포함) 저장

사용법:
  python slide_detector.py <video_path>
  python slide_detector.py <video_path> -o ./output --min-duration 2.0
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# ── 파라미터 ──────────────────────────────────────────────
SAMPLE_FPS       = 4      # 분석용 샘플링 FPS
MIN_DURATION     = 1.5    # 이 초 미만 구간은 스킵
TRANSITION_THR   = 8      # pHash 해밍 거리: 슬라이드 전환 판정
DUPLICATE_THR    = 6      # pHash 해밍 거리: 중복 슬라이드 판정
START_OFFSET     = 0.5    # 전환 직후 이 초 뒤 프레임을 "시작"으로 저장
END_OFFSET       = 0.3    # 다음 전환 이 초 전 프레임을 "종료"으로 저장


# ── 영상 메타데이터 ───────────────────────────────────────

def get_video_meta(v_path: str) -> tuple[int, int, float]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", v_path]
    meta = json.loads(subprocess.check_output(cmd))
    v_s = next(s for s in meta["streams"] if s["codec_type"] == "video")
    w, h = int(v_s["width"]), int(v_s["height"])
    num, den = map(int, v_s["avg_frame_rate"].split("/"))
    if den == 0:
        raise ValueError("avg_frame_rate 분모가 0입니다.")
    return w, h, num / den


# ── pHash 시퀀스 추출 ─────────────────────────────────────

def sample_phashes(v_path: str, w: int, h: int) -> list:
    """4fps 샘플링으로 pHash 배열 반환"""
    proc = subprocess.Popen(
        ["ffmpeg", "-i", v_path,
         "-vf", f"fps={SAMPLE_FPS}",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    hashes = []
    sz = w * h
    try:
        while True:
            data = proc.stdout.read(sz)
            if not data or len(data) < sz:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
            small = Image.fromarray(frame).resize((w // 4, h // 4))
            hashes.append(imagehash.phash(small))
            print(f"\r  pHash 계산: {len(hashes)}프레임 "
                  f"({len(hashes) / SAMPLE_FPS:.1f}초)", end="", flush=True)
    finally:
        proc.terminate()
        proc.wait()
    print()
    return hashes


# ── 안정 구간 탐지 ────────────────────────────────────────

def detect_stable_periods(hashes: list) -> list[tuple[int, int]]:
    """
    연속된 pHash 시퀀스에서 안정 구간을 탐지.
    반환: [(start_idx, end_idx), ...] (샘플 프레임 인덱스 기준)
    """
    if len(hashes) < 2:
        return [(0, len(hashes) - 1)]

    periods = []
    start = 0

    for i in range(1, len(hashes)):
        dist = hashes[i] - hashes[i - 1]
        if dist > TRANSITION_THR:
            periods.append((start, i - 1))
            start = i

    periods.append((start, len(hashes) - 1))
    return periods


# ── 프레임 저장 ───────────────────────────────────────────

def save_frame(v_path: str, timestamp: float, out_path: str) -> None:
    """특정 타임스탬프의 프레임을 고품질 JPEG로 저장"""
    timestamp = max(0.0, timestamp)
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", f"{timestamp:.3f}",
         "-i", v_path,
         "-frames:v", "1",
         "-q:v", "2",
         out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ── 메인 ─────────────────────────────────────────────────

def run(v_path: str, output_dir: str, min_duration: float = MIN_DURATION) -> None:
    v_path = str(Path(v_path).resolve())
    v_name = Path(v_path).stem
    out_dir = Path(output_dir) / v_name

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # [1] 메타데이터
    print(f"\n[1/3] 영상 정보 분석: {v_name}")
    w, h, fps = get_video_meta(v_path)
    print(f"  해상도: {w}x{h}, FPS: {fps:.2f}")

    # [2] pHash 샘플링
    print(f"\n[2/3] {SAMPLE_FPS}fps 샘플링 + pHash 계산")
    hashes = sample_phashes(v_path, w, h)
    total_sec = len(hashes) / SAMPLE_FPS
    print(f"  총 {len(hashes)}샘플 완료 (영상 길이: {int(total_sec)//60}분 {int(total_sec)%60}초)")

    # [3] 구간 탐지 + 저장
    print(f"\n[3/3] 슬라이드 구간 탐지 (전환 기준: pHash>{TRANSITION_THR}, "
          f"최소 {min_duration}초)")
    periods = detect_stable_periods(hashes)
    print(f"  탐지된 구간: {len(periods)}개\n")

    seen: dict[int, dict] = {}
    timeline = []   # 강의 흐름 순서 기록 (짧음 스킵 제외)
    log_rows = []
    slide_no = 0

    for start_idx, end_idx in periods:
        duration  = (end_idx - start_idx + 1) / SAMPLE_FPS
        start_ts  = start_idx / SAMPLE_FPS
        end_ts    = end_idx   / SAMPLE_FPS

        # ── 1.5초 미만 스킵
        if duration < min_duration:
            log_rows.append({
                "status":    "SKIP_SHORT",
                "start_sec": round(start_ts, 2),
                "end_sec":   round(end_ts,   2),
                "duration":  round(duration,  2),
            })
            print(f"  [{_fmt(start_ts)} ~ {_fmt(end_ts)}] ({duration:.1f}s) → 짧음, 스킵")
            continue

        # ── 중복 체크 (start hash + end hash 둘 다 비교)
        h_start = hashes[start_idx]
        h_end   = hashes[end_idx]
        dup_no = next(
            (no for no, info in seen.items()
             if h_start - info["hash_start"] <= DUPLICATE_THR
             or h_start - info["hash_end"]   <= DUPLICATE_THR
             or h_end   - info["hash_start"] <= DUPLICATE_THR
             or h_end   - info["hash_end"]   <= DUPLICATE_THR),
            None,
        )

        if dup_no is not None:
            # 누적 시간 합산 + 이번 등장 기록
            seen[dup_no]["total_duration"] += duration
            seen[dup_no]["occurrences"].append({
                "start_sec": round(start_ts, 2),
                "end_sec":   round(end_ts,   2),
                "duration":  round(duration,  2),
            })

            # end 프레임 갱신: 이번 구간이 더 길면 덮어쓰기
            end_updated = False
            if duration > seen[dup_no]["best_duration"]:
                seen[dup_no]["best_duration"] = duration
                seen[dup_no]["hash_end"] = h_end
                save_end_ts = max(start_ts, end_ts - END_OFFSET)
                f_end = str(out_dir / f"slide_{dup_no:03d}_end.jpg")
                save_frame(v_path, save_end_ts, f_end)
                end_updated = True

            # 타임라인 기록
            timeline.append({
                "slide_no":  dup_no,
                "start_sec": round(start_ts, 2),
                "end_sec":   round(end_ts,   2),
                "duration":  round(duration,  2),
                "is_dup":    True,
            })

            suffix = " ★end 갱신" if end_updated else ""
            print(f"  [{_fmt(start_ts)} ~ {_fmt(end_ts)}] ({duration:.1f}s) "
                  f"→ 중복 (슬라이드 {dup_no:03d}){suffix}")

            log_rows.append({
                "status":         "DUP",
                "slide_no":       dup_no,
                "start_sec":      round(start_ts, 2),
                "end_sec":        round(end_ts,   2),
                "duration":       round(duration,  2),
                "total_duration": round(seen[dup_no]["total_duration"], 2),
                "end_updated":    end_updated,
            })
            continue

        # ── 새 슬라이드 저장
        slide_no += 1
        seen[slide_no] = {
            "hash_start":     h_start,
            "hash_end":       h_end,
            "total_duration": duration,
            "best_duration":  duration,
            "occurrences":    [{"start_sec": round(start_ts, 2),
                                "end_sec":   round(end_ts,   2),
                                "duration":  round(duration,  2)}],
        }

        save_start_ts = start_ts + START_OFFSET
        save_end_ts   = max(start_ts, end_ts - END_OFFSET)

        f_start = str(out_dir / f"slide_{slide_no:03d}_start.jpg")
        f_end   = str(out_dir / f"slide_{slide_no:03d}_end.jpg")
        save_frame(v_path, save_start_ts, f_start)
        save_frame(v_path, save_end_ts,   f_end)

        # 타임라인 기록
        timeline.append({
            "slide_no":  slide_no,
            "start_sec": round(start_ts, 2),
            "end_sec":   round(end_ts,   2),
            "duration":  round(duration,  2),
            "is_dup":    False,
        })

        log_rows.append({
            "status":          "SAVED",
            "slide_no":        slide_no,
            "start_sec":       round(start_ts,      2),
            "end_sec":         round(end_ts,         2),
            "duration":        round(duration,       2),
            "total_duration":  round(duration,       2),
            "saved_start_sec": round(save_start_ts, 2),
            "saved_end_sec":   round(save_end_ts,   2),
        })
        print(f"  [{_fmt(start_ts)} ~ {_fmt(end_ts)}] ({duration:.1f}s) "
              f"→ 슬라이드 {slide_no:03d} 저장")

    # ── 슬라이드별 occurrences를 log_rows(SAVED)에 병합
    for row in log_rows:
        if row["status"] == "SAVED":
            no = row["slide_no"]
            row["total_duration"] = round(seen[no]["total_duration"], 2)
            row["occurrences"]    = seen[no]["occurrences"]

    # ── 로그 저장
    log_path = out_dir / f"{v_name}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "slides":   log_rows,
            "timeline": timeline,
        }, f, ensure_ascii=False, indent=2)

    skipped_short = sum(1 for r in log_rows if r["status"] == "SKIP_SHORT")
    skipped_dup   = sum(1 for r in log_rows if r["status"] == "DUP")

    print(f"\n{'='*50}")
    print(f"  저장된 슬라이드 : {slide_no}개")
    print(f"  스킵 (짧음)     : {skipped_short}개")
    print(f"  중복 등장       : {skipped_dup}개")
    print(f"  출력 경로       : {out_dir}")
    print(f"  로그            : {log_path}")
    print(f"{'='*50}")
    print(f"\n강의 흐름 (timeline):")
    for t in timeline:
        dup_mark = " [재등장]" if t["is_dup"] else ""
        print(f"  [{_fmt(t['start_sec'])} ~ {_fmt(t['end_sec'])}] "
              f"슬라이드 {t['slide_no']:03d}{dup_mark} ({t['duration']:.0f}s)")


def _fmt(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m:02d}:{s:02d}"


# ── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="슬라이드 전환 감지 및 추출기")
    parser.add_argument("video", help="영상 파일 경로")
    parser.add_argument("-o", "--output", default="./output", help="출력 디렉토리 (기본: ./output)")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION,
                        help=f"최소 슬라이드 지속 시간 초 (기본: {MIN_DURATION})")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    run(args.video, args.output, args.min_duration)
