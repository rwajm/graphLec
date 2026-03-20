"""
표준편차 기반 강조 구간 감지 (수정 버전 - 비교용)
"""

import librosa
import numpy as np
from difflib import SequenceMatcher


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
    
    # ✅ 표준편차 기반 임계값
    volume_threshold = rms_mean + 1.5 * rms_std
    
    # 피치
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else 0)
    
    valid_pitches = [p for p in pitch_values if p > 0]
    pitch_mean = np.mean(valid_pitches) if valid_pitches else 0
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
        
        # 피치
        pitch_start_idx = int(start_time * sr / 512)
        pitch_end_idx = int(end_time * sr / 512)
        if pitch_end_idx > len(pitch_values):
            pitch_end_idx = len(pitch_values)
        
        seg_pitches = [p for p in pitch_values[pitch_start_idx:pitch_end_idx] if p > 0]
        pitch_variation = np.std(seg_pitches) if len(seg_pitches) > 0 else 0
        
        # 점수 계산
        volume_score = 0.0
        pitch_score = 0.0
        
        # ✅ 임계값 초과만 점수
        if seg_rms_max > volume_threshold:
            volume_ratio = seg_rms_max / volume_threshold
            volume_score = float(min((volume_ratio - 1) * 60, 60))
        else:
            volume_ratio = seg_rms_max / volume_threshold if volume_threshold > 0 else 0.0
        
        # ✅ 피치도 표준편차 기반
        if pitch_variation > pitch_std * 2:
            pitch_ratio = pitch_variation / (pitch_std * 2) if pitch_std > 0 else 0.0
            pitch_score = float(min(pitch_ratio * 40, 40))
        
        emphasis_score = float(volume_score + pitch_score)
        
        # ✅ 임계값 50점
        if emphasis_score >= 50:
            emphasis_segments.append({
                'start': start_time,
                'end': end_time,
                'text': seg['text'],
                # 전체 오디오 강조 점수 (볼륨+피치)
                'emphasis_score': emphasis_score,
                # 볼륨/피치 세부 점수 및 지표
                'volume_score': volume_score,
                'pitch_score': pitch_score,
                'volume_ratio': float(seg_rms_max / rms_mean) if rms_mean > 0 else 0.0,
                'pitch_variation': float(pitch_variation),
                'detection_method': 'std_based'
            })
    
    print(f"    -> 표준편차 기반: {len(emphasis_segments)}개 (전체의 {len(emphasis_segments)/len(segments)*100:.1f}%)")
    return emphasis_segments


# 가중치 키워드 (확인용 get_keywords_weighted / 감지용 공용)
KEYWORDS_WEIGHTED = {
    'strong': (['중요', '핵심', '반드시', '꼭', '필수'], 30),
    'summary': (['정리하면', '요약하면', '다시 말하면', '즉'], 25),
    'exam': (['시험', '문제', '출제', '나옵니다'], 40)
}


# ---------------------------------------------------------------------------
# (현재 미사용) 가중치 키워드 목록 조회
#
# 역할:
# - UI/리포트에서 "가중치 키워드 사전"을 덤프해서 확인하려고 만든 헬퍼였습니다.
#
# 현재:
# - main 파이프라인에서 별도로 덤프하지 않으므로 비활성화했습니다.
# - 필요해지면 아래 코드를 다시 함수로 되돌리면 됩니다.
# ---------------------------------------------------------------------------

"""
def get_keywords_weighted() -> dict:
    return {
        cat: {'keywords': list(kw), 'weight': w}
        for cat, (kw, w) in KEYWORDS_WEIGHTED.items()
    }
"""


def detect_emphasis_by_keywords_weighted(segments: list[dict]) -> list[dict]:
    """가중치 적용 키워드 감지 (습관 필터링 없음)"""
    print("  [가중치 키워드] 분석 중...")
    emphasis_keywords = KEYWORDS_WEIGHTED
    emphasis_segments = []

    for seg in segments:
        text = seg['text']
        matched_keywords = []
        keyword_score = 0

        for category, (keywords, weight) in emphasis_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    matched_keywords.append(keyword)
                    keyword_score += weight
        
        # ✅ 점수 40 이상만
        if keyword_score >= 40:
            emphasis_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': text,
                'emphasis_score': float(keyword_score),
                'matched_keywords': matched_keywords,
                'detection_method': 'keyword_weighted'
            })
    
    print(f"    -> 가중치 키워드: {len(emphasis_segments)}개 (전체의 {len(emphasis_segments)/len(segments)*100:.1f}%)")
    return emphasis_segments
