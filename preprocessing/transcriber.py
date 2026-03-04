#!/usr/bin/env python3
"""
Groq Whisper 전사기 (단독)
- MP4/영상 파일을 Groq Whisper large-v3-turbo로 전사
- Gemini 없이 Groq만 사용

사용법: python groq_transcriber.py "1절 운영체제의 개념.mp4"
설치: pip install groq python-dotenv
필요: ffmpeg
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_groq_client = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY 환경변수를 설정해주세요")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def get_duration(video_path: str) -> float:
    """ffprobe로 영상 길이(초) 반환"""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def transcribe(video_path: str) -> list[dict]:
    """영상을 10분 청크로 분할, Groq Whisper로 전사"""
    duration = get_duration(video_path)
    print(f"  영상 길이: {duration/60:.1f}분")

    CHUNK_SEC = 600
    segments = []
    total_chunks = max(1, -(-int(duration) // CHUNK_SEC))

    for i in range(total_chunks):
        offset = i * CHUNK_SEC
        chunk_path = f"temp_chunk_{i}.wav"

        # 오디오 청크 추출
        subprocess.run(
            ["ffmpeg", "-i", video_path,
             "-ss", str(offset), "-t", str(CHUNK_SEC),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             "-y", chunk_path],
            capture_output=True,
        )

        chunk = Path(chunk_path)
        if not chunk.exists() or chunk.stat().st_size == 0:
            continue

        if total_chunks > 1:
            end = min(offset + CHUNK_SEC, duration)
            print(f"  [{i+1}/{total_chunks}] {offset/60:.0f}분~{end/60:.0f}분 처리 중...")

        # Groq Whisper 전사
        with open(chunk_path, "rb") as f:
            transcription = _get_client().audio.transcriptions.create(
                file=(chunk_path, f.read()),
                model="whisper-large-v3-turbo",
                language="ko",
                response_format="verbose_json",
            )

        for seg in transcription.segments:
            segments.append({
                "start": round(seg["start"] + offset, 2),
                "end": round(seg["end"] + offset, 2),
                "text": seg["text"].strip(),
            })

        chunk.unlink(missing_ok=True)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Groq Whisper 전사기")
    parser.add_argument("video", help="영상 파일 경로 (mp4 등)")
    parser.add_argument("-o", "--output", help="출력 JSON 경로 (기본: {name}_transcribed.json)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.with_name(video_path.stem + "_transcribed.json")

    print(f"Groq Whisper 전사 시작: {video_path.name}")
    segments = transcribe(str(video_path))
    print(f"  전사 완료: {len(segments)}개 세그먼트")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"  저장: {output_path}")


if __name__ == "__main__":
    main()
