"""
강조 구간 감지 모듈 비교 및 통합
"""

import json


def _collect_keywords_from_det(det: dict) -> tuple[list[str], list[str], str]:
    """
    감지 결과에서 키워드 목록과 방법 이름 추출.
    반환: (topic_keywords, importance_keywords, method_name)
    """
    method = det.get("detection_method", "")
    topic_keywords: list[str] = []
    importance_keywords: list[str] = []
    # 주제 키워드 반복 모듈
    if method == "topic_keyword_repeat" and det.get("repeated_topic_keywords"):
        topic_keywords = list(det["repeated_topic_keywords"])
    # 고정 가중치 중요 키워드 모듈
    if method == "keyword_weighted" and det.get("matched_keywords"):
        importance_keywords = list(det["matched_keywords"])
    return topic_keywords, importance_keywords, method


def combine_emphasis_simple(audio_emphasis, keyword_emphasis, segments):
    """간단한 통합 (평균/표준편차 버전용). 선정 키워드 포함."""
    segment_emphasis_info = {}
    
    for seg in segments:
        start = seg['start']
        segment_emphasis_info[start] = {
            'segment': seg,
            'methods': [],
            'scores': [],
            'scores_by_method': {},
            'keywords_by_method': {},
            'topic_keywords': [],
            'importance_keywords': [],
            'audio_detail': None,
        }
    
    for det in audio_emphasis + keyword_emphasis:
        start = det['start']
        if start in segment_emphasis_info:
            info = segment_emphasis_info[start]
            method = det.get('detection_method')
            score = float(det.get('emphasis_score', 0.0) or 0.0)
            if method:
                info['methods'].append(method)
                info['scores'].append(score)
                prev = info['scores_by_method'].get(method, 0.0)
                if score > prev:
                    info['scores_by_method'][method] = score
            topic_kws, importance_kws, method = _collect_keywords_from_det(det)
            if method:
                if topic_kws:
                    info['keywords_by_method'][method] = topic_kws
                    info['topic_keywords'].extend(topic_kws)
                if importance_kws:
                    info['keywords_by_method'][method] = importance_kws
                    info['importance_keywords'].extend(importance_kws)
            # 오디오(표준편차 기반) 세부 정보
            if method == 'std_based':
                info['audio_detail'] = {
                    'score': score,
                    'volume_score': float(det.get('volume_score', 0.0) or 0.0),
                    'pitch_score': float(det.get('pitch_score', 0.0) or 0.0),
                    'volume_ratio': float(det.get('volume_ratio', 0.0) or 0.0),
                    'pitch_variation': float(det.get('pitch_variation', 0.0) or 0.0),
                }
    
    annotated_segments = []
    emphasis_only_sections = []
    
    for start, info in segment_emphasis_info.items():
        seg = info['segment'].copy()
        
        if info['methods']:
            max_score = max(info['scores'])
            kw_by_method = info.get('keywords_by_method') or {}

            # 키워드 분리: 주제/내용 키워드 vs 고정 가중치 중요 키워드
            def _uniq(seq: list[str]) -> list[str]:
                return list(dict.fromkeys(seq))

            topic_keywords = _uniq(info.get('topic_keywords', []))
            importance_keywords = _uniq(info.get('importance_keywords', []))
            # 전체 키워드: 주제 → 중요 순으로, 중복 제거
            merged_keywords = topic_keywords + [k for k in importance_keywords if k not in set(topic_keywords)]

            seg['emphasis'] = '강조'
            seg['emphasis_detected'] = True
            seg['emphasis_score'] = round(float(max_score), 1)
            seg['emphasis_methods'] = list(set(info['methods']))
            seg['detection_count'] = len(info['methods'])

            if merged_keywords:
                seg['emphasis_keywords'] = merged_keywords
                seg['emphasis_keywords_by_method'] = kw_by_method

            # 상세 구조: 오디오 + 키워드 별도 노출
            seg['emphasis_detail'] = {
                'score': seg['emphasis_score'],
                'score_by_method': info.get('scores_by_method', {}),
                'audio': info.get('audio_detail'),
                'keywords': {
                    'topic_keywords': topic_keywords,
                    'importance_keywords': importance_keywords,
                    'all_keywords': merged_keywords,
                    'by_method': kw_by_method,
                },
                'methods': seg['emphasis_methods'],
                'detection_count': seg['detection_count'],
            }

            emphasis_only_sections.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'emphasis': '강조',
                'emphasis_score': seg['emphasis_score'],
                'methods': seg['emphasis_methods'],
                'detection_count': seg['detection_count'],
                'keywords': merged_keywords,
                'keywords_by_method': kw_by_method,
                'emphasis_detail': seg['emphasis_detail'],
            })
        else:
            seg['emphasis'] = None
            seg['emphasis_detected'] = False
            seg['emphasis_score'] = 0
        
        annotated_segments.append(seg)
    
    emphasis_only_sections.sort(key=lambda x: x['emphasis_score'], reverse=True)
    
    return annotated_segments, emphasis_only_sections


# ---------------------------------------------------------------------------
# (현재 미사용) 모듈 비교 분석
#
# 역할:
# - 평균/표준편차/적응형 3개 모듈의 강조 구간을 비교해서
#   겹침(합의)과 고유 감지 구간을 통계로 요약하는 분석 함수였습니다.
#
# 현재:
# - main 파이프라인에서 비교 분석 결과물을 생성하지 않으므로 비활성화했습니다.
# - 필요해지면 아래의 "보관된 원본 코드"를 다시 함수로 되돌리면 됩니다.
# ---------------------------------------------------------------------------

"""
def compare_emphasis_results(
    mean_result: dict,
    std_result: dict,
    adaptive_result: dict,
    segments: list[dict],
    duration: float,
    groups: list[dict] | None = None,
) -> dict:
    ...
"""
