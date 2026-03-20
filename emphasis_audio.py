"""
오디오 신호 기반 강조 구간 감지 (볼륨 + 피치 표준편차)
"""

import librosa
import numpy as np


def detect_emphasis_by_std(y, sr, segments: list[dict]) -> list[dict]:
    """
    표준편차 기반 강조 구간 감지
    - 평균 + 1.5 표준편차
    - 임계값 50점
    """
    print("  [표준편차 기반] 분석 중...")

    # RMS 에너지 계산
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    volume_threshold = rms_mean + 1.5 * rms_std

    # 피치
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else 0)

    valid_pitches = [p for p in pitch_values if p > 0]
    pitch_std = np.std(valid_pitches) if valid_pitches else 0

    emphasis_segments = []

    for seg in segments:
        start_time = seg['start']
        end_time = seg['end']

        seg_rms_indices = np.where((rms_times >= start_time) & (rms_times <= end_time))[0]
        if len(seg_rms_indices) == 0:
            continue

        seg_rms = rms[seg_rms_indices]
        seg_rms_max = np.max(seg_rms)

        pitch_start_idx = int(start_time * sr / 512)
        pitch_end_idx = int(end_time * sr / 512)
        if pitch_end_idx > len(pitch_values):
            pitch_end_idx = len(pitch_values)

        seg_pitches = [p for p in pitch_values[pitch_start_idx:pitch_end_idx] if p > 0]
        pitch_variation = np.std(seg_pitches) if len(seg_pitches) > 0 else 0

        volume_score = 0.0
        pitch_score = 0.0

        if seg_rms_max > volume_threshold:
            volume_ratio = seg_rms_max / volume_threshold
            volume_score = float(min((volume_ratio - 1) * 60, 60))
        else:
            volume_ratio = seg_rms_max / volume_threshold if volume_threshold > 0 else 0.0

        if pitch_variation > pitch_std * 2:
            pitch_ratio = pitch_variation / (pitch_std * 2) if pitch_std > 0 else 0.0
            pitch_score = float(min(pitch_ratio * 40, 40))

        emphasis_score = float(volume_score + pitch_score)

        if emphasis_score >= 50:
            emphasis_segments.append({
                'start': start_time,
                'end': end_time,
                'text': seg['text'],
                'emphasis_score': emphasis_score,
                'volume_score': volume_score,
                'pitch_score': pitch_score,
                'volume_ratio': float(seg_rms_max / rms_mean) if rms_mean > 0 else 0.0,
                'pitch_variation': float(pitch_variation),
                'detection_method': 'std_based',
            })

    print(f"    -> 표준편차 기반: {len(emphasis_segments)}개 (전체의 {len(emphasis_segments)/len(segments)*100:.1f}%)")
    return emphasis_segments