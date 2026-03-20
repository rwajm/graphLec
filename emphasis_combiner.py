"""
강조 감지 결과 통합

오디오(emphasis_audio.py)와 키워드(emphasis_keyword.py) 감지 결과를
세그먼트 단위로 합산해 최종 강조 구간을 생성한다.
"""


def _collect_keywords_from_det(det: dict) -> tuple[list[str], list[str], str]:
    """
    감지 결과에서 키워드 목록과 방법 이름 추출.
    반환: (topic_keywords, importance_keywords, method_name)
    """
    method = det.get("detection_method", "")
    topic_keywords: list[str] = []
    importance_keywords: list[str] = []
    if method == "topic_keyword_repeat" and det.get("repeated_topic_keywords"):
        topic_keywords = list(det["repeated_topic_keywords"])
    if method == "keyword_weighted" and det.get("matched_keywords"):
        importance_keywords = list(det["matched_keywords"])
    return topic_keywords, importance_keywords, method


def combine_emphasis_simple(
    audio_emphasis: list[dict],
    keyword_emphasis: list[dict],
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    오디오 감지 결과 + 키워드 감지 결과를 세그먼트 단위로 통합.

    반환:
    - annotated_segments: 원본 segments에 emphasis 필드가 추가된 리스트
    - emphasis_only_sections: 강조 판정된 구간만, emphasis_score 내림차순 정렬
    """
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
        if start not in segment_emphasis_info:
            continue
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
        if method == 'std_based':
            info['audio_detail'] = {
                'score': score,
                'volume_score': float(det.get('volume_score', 0.0) or 0.0),
                'pitch_score': float(det.get('pitch_score', 0.0) or 0.0),
                'volume_ratio': float(det.get('volume_ratio', 0.0) or 0.0),
                'pitch_variation': float(det.get('pitch_variation', 0.0) or 0.0),
            }

    def _uniq(seq: list[str]) -> list[str]:
        return list(dict.fromkeys(seq))

    annotated_segments = []
    emphasis_only_sections = []

    for start, info in segment_emphasis_info.items():
        seg = info['segment'].copy()

        if info['methods']:
            max_score = max(info['scores'])
            kw_by_method = info.get('keywords_by_method') or {}

            topic_keywords = _uniq(info.get('topic_keywords', []))
            importance_keywords = _uniq(info.get('importance_keywords', []))
            merged_keywords = topic_keywords + [k for k in importance_keywords if k not in set(topic_keywords)]

            seg['emphasis'] = '강조'
            seg['emphasis_detected'] = True
            seg['emphasis_score'] = round(float(max_score), 1)
            seg['emphasis_methods'] = list(set(info['methods']))
            seg['detection_count'] = len(info['methods'])

            if merged_keywords:
                seg['emphasis_keywords'] = merged_keywords
                seg['emphasis_keywords_by_method'] = kw_by_method

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