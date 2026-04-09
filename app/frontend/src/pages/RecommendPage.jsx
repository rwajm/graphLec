import { useState, useRef } from 'react'
import RecommendListItem from '../components/recommend/RecommendListItem'
import { DUMMY_RECOMMEND_ANSWERS } from '../data/dummy'

const SUGGESTIONS = [
  '선형대수 행렬 강의',
  '30분 내외의 설명이 쉬운 기초 파이썬',
  '머신러닝 배우기 전에 들으면 좋은 수학 강의',
  '딥러닝 CNN 실습 위주 강의',
]

export default function RecommendPage({ onNavigate }) {
  const [query,       setQuery]       = useState('')
  const [submitted,   setSubmitted]   = useState(false)
  const [loading,     setLoading]     = useState(false)
  const [results,     setResults]     = useState(null)
  const [searchLabel, setSearchLabel] = useState('')
  const inputRef = useRef(null)

  async function handleSearch(overrideQuery) {
    const text = (overrideQuery ?? query).trim()
    if (!text || loading) return

    setQuery(text)
    setSearchLabel(text)
    setSubmitted(true)   // 클래스 토글 → CSS transition 시작
    setLoading(true)
    setResults(null)

    try {
      await new Promise(r => setTimeout(r, 700))
      const keys  = Object.keys(DUMMY_RECOMMEND_ANSWERS)
      const match = keys.find(k => k.includes(text) || text.includes(k.slice(0, 4)))
      const res   = DUMMY_RECOMMEND_ANSWERS[match ?? '최근 업로드한 강의를 추천해주세요']
      setResults(res?.lectures ?? [])
    } finally {
      setLoading(false)
    }
  }

  // 로고 클릭 → 초기화면으로 복귀
  function handleReset() {
    setSubmitted(false)
    setQuery('')
    setResults(null)
    setSearchLabel('')
    setTimeout(() => inputRef.current?.focus(), 350) // transition 끝난 뒤 포커스
  }

  function handlePlay(lectureId) {
    onNavigate?.({ page: 'lecture', lectureId })
  }

  return (
    <div className={`rec-page${submitted ? ' rec-page--searched' : ''}`}>

      {/* 히어로: 로고(클릭 시 초기화) + 문구 */}
      <div className="rec-hero">
        <div className="rec-logo" onClick={handleReset}>
          Graph<span>Lec</span>
        </div>
        <p className="rec-headline">어떤 강의 영상을 보고 싶은지 설명해 주세요!</p>
      </div>

      {/* 검색바 — 항상 동일한 디자인 */}
      <div className="rec-bar-wrap content-max">
        <div className="rec-searchbar">
          <span className="rec-searchbar-icon">🔍</span>
          <input
            ref={inputRef}
            className="rec-searchbar-input"
            placeholder="강의 내용, 주제, 난이도 등을 자유롭게 입력하세요"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSearch() }
            }}
            disabled={loading}
            autoFocus
          />
          {query && (
            <button className="rec-searchbar-clear" onClick={() => setQuery('')} tabIndex={-1}>
              ✕
            </button>
          )}
          <button
            className="rec-searchbar-btn"
            onClick={() => handleSearch()}
            disabled={loading || !query.trim()}
          >
            {loading ? '…' : '검색'}
          </button>
        </div>
      </div>

      {/* 예시 칩 */}
      <div className="rec-chips content-max">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            className="rec-chip"
            onClick={() => { setQuery(s); handleSearch(s) }}
          >
            {s}
          </button>
        ))}
      </div>

      {/* 결과 목록 */}
      <div className="rec-results">
        <div className="content-max" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {loading ? (
            <div className="rec-loading">
              <span className="rec-loading-dot" />
              <span className="rec-loading-dot" />
              <span className="rec-loading-dot" />
            </div>
          ) : results === null ? null : results.length === 0 ? (
            <div className="rec-empty">
              <p className="rec-empty-title">검색 결과가 없습니다</p>
              <p className="rec-empty-sub">다른 키워드로 다시 시도해 보세요</p>
            </div>
          ) : (
            <>
              <div className="rec-result-label">
                <strong>"{searchLabel}"</strong>에 대한 추천 강의 {results.length}개
              </div>
              <div className="rec-list">
                {results.map(lec => (
                  <RecommendListItem key={lec.id} lecture={lec} onPlay={handlePlay} />
                ))}
              </div>
            </>
          )}
        </div>
      </div>

    </div>
  )
}