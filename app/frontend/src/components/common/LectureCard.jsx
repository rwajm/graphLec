const THUMB_COLORS = {
  '컴퓨터 과학':    '#1e3a5f',
  '데이터 사이언스': '#1a3d2b',
  '소프트웨어 공학': '#2d1f3d',
  '수학':           '#3d2a1a',
}

const STATUS_MAP = {
  done:       { label: '분석 완료', cls: 'sDone' },
  processing: { label: '분석 중',   cls: 'sProc' },
  pending:    { label: '대기',       cls: 'sWait' },
  error:      { label: '오류',       cls: 'sErr'  },
}

/**
 * @param {object}   lecture
 * @param {boolean}  showStatus
 * @param {function} onSelect  - 카드 클릭 시 호출. QueryPage 추천 목록 등에서 전달.
 */
export default function LectureCard({ lecture, showStatus = false, onSelect }) {
  const status  = STATUS_MAP[lecture.status] ?? STATUS_MAP.pending
  const thumbBg = THUMB_COLORS[lecture.category] ?? '#1e2333'

  function handleClick() {
    onSelect?.(lecture)
  }

  return (
    <div className="card" onClick={handleClick}>
      <div className="lecture-card-thumb" style={{ background: thumbBg }}>
        <div className="lecture-card-thumb-icon">
          {{ '컴퓨터 과학': '🧠', '데이터 사이언스': '📊', '소프트웨어 공학': '⚙️', '수학': '📐' }[lecture.category] ?? '🎬'}
        </div>
        {showStatus && (
          <div className={`lecture-card-status-badge ${status.cls}`}>
            {status.label}
          </div>
        )}
      </div>

      <div className="lecture-card-info">
        <div className="lecture-card-category">{lecture.category}</div>
        <div className="lecture-card-title">{lecture.title}</div>
        <div className="lecture-card-meta">
          {lecture.instructor_name}
          {showStatus && (
            <span className="date">
              {' · '}{new Date(lecture.created_at).toLocaleDateString('ko-KR')}
            </span>
          )}
        </div>
        {lecture.tags?.length > 0 && (
          <div className="lecture-card-tags">
            {lecture.tags.map(tag => (
              <span key={tag} className="lecture-card-tag">{tag}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
