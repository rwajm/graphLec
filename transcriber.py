"""
Groq Whisper 음성 전사
"""

import subprocess
from pathlib import Path

from config import groq_client


def transcribe_video(video_path: str, duration: float, output_dir=None) -> list[dict]:
    """영상에서 음성 추출 후 Groq Whisper로 전사"""
    segments = []
    chunk_duration = 600  # 10분 단위
    base_dir = Path(output_dir) if output_dir else Path(".")

    # 청크 수 계산
    total_chunks = max(1, int(duration / chunk_duration) + (1 if duration % chunk_duration > 0 else 0))

    for i in range(total_chunks):
        chunk_start = i * chunk_duration
        chunk_path = str(base_dir / f"temp_chunk_{i}.wav")

        # 영상에서 직접 오디오 청크 추출
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(chunk_start), "-t", str(chunk_duration),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", chunk_path
        ], capture_output=True)

        if not Path(chunk_path).exists() or Path(chunk_path).stat().st_size == 0:
            continue

        if total_chunks > 1:
            chunk_end = min((i + 1) * chunk_duration, duration)
            print(f"  [{i+1}/{total_chunks}] {chunk_start//60:.0f}분~{chunk_end//60:.0f}분 처리 중...")

        # Groq API 호출
        with open(chunk_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(chunk_path, f.read()),
                model="whisper-large-v3-turbo",
                language="ko",
                response_format="verbose_json",
            )

        # 타임스탬프 보정 후 추가
        for seg in transcription.segments:
            segments.append({
                "start": seg["start"] + chunk_start,
                "end": seg["end"] + chunk_start,
                "text": seg["text"].strip()
            })

        # 임시 파일 삭제
        Path(chunk_path).unlink(missing_ok=True)

    return segments
