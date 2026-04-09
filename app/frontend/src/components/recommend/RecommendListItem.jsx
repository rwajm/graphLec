import { useState } from 'react'

/**
 * 추천 강의 목록 아이템
 * 레이아웃: [썸네일] [태그·제목·강의자] [점수% / 자세히 보기▼]
 *
 * @param {object}   lecture  - 강의 및 AI 추천 분석 데이터
 * @param {function} onPlay   - 시청 버튼 클릭 시 호출 (lectureId 전달)
 */
export default function RecommendListItem({ lecture, onPlay }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className={`rec-item${isExpanded ? ' rec-item--expanded' : ''}`}>

      {/* ── 메인 행 ── */}
      <div className="rec-item-main" onClick={() => setIsExpanded(v => !v)}>

        {/* 썸네일 */}
        <div className="rec-col">
          <div className="rec-thumb">
            <span className="rec-thumb-icon">{lecture.categoryIcon ?? '🎬'}</span>
            {lecture.duration && (
              <span className="rec-duration">{lecture.duration}</span>
            )}
          </div>
        </div>

        {/* 메타: 태그 → 제목+시간 → 강의자·날짜 */}
        <div className="rec-col rec-col-meta">
          {lecture.tags?.length > 0 && (
            <div className="rec-col-tags">
              {lecture.tags.map((tag, i) => (
                <span key={i} className="rec-tag">#{tag}</span>
              ))}
            </div>
          )}
          <div className="rec-title-row">
            <span className="rec-title">{lecture.title}</span>
            {lecture.duration && (
              <span className="rec-duration-inline">· {lecture.duration}</span>
            )}
          </div>
          <div className="rec-sub">
            {lecture.instructor_name}
            {lecture.date && ` | ${lecture.date}`}
          </div>
        </div>

        {/* 점수 + 토글 */}
        <div className="rec-col rec-col-score">
          {lecture.overallScore != null && (
            <div className="rec-overall-score">{lecture.overallScore}%</div>
          )}
          <button className="rec-toggle-btn">
            {isExpanded ? '접기 ▲' : '자세히 ▼'}
          </button>
        </div>
      </div>

      {/* ── 아코디언 ── */}
      {isExpanded && (
        <div className="rec-item-details">
          <div className="rec-details-content">
            {lecture.scores && (
              <>
                <h4 className="rec-details-title">AI 세부 분석 지표</h4>
                <div className="rec-scores-grid">
                  <DetailScoreBar label="내용 일치"        score={lecture.scores.content}    />
                  <DetailScoreBar label="난이도 적합"      score={lecture.scores.difficulty}  />
                  <DetailScoreBar label="키워드 연관"      score={lecture.scores.keyword}     />
                  <DetailScoreBar label="강사/도메인 선호" score={lecture.scores.preference}  />
                </div>
              </>
            )}
            <div className="rec-details-action">
              <button
                className="rec-play-btn"
                onClick={e => { e.stopPropagation(); onPlay?.(lecture.id) }}
              >
                ▶ 이 강의 시청하기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function DetailScoreBar({ label, score }) {
  return (
    <div className="rec-score-row">
      <span className="rec-score-label">{label}</span>
      <div className="rec-score-track">
        <div className="rec-score-fill" style={{ width: `${score}%` }} />
      </div>
      <span className="rec-score-num">{score}</span>
    </div>
  )
}