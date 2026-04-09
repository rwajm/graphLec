import { useEffect, useRef } from 'react'

const BADGE = {
  slide:    { cls: 'tl-badge--slide',    label: '슬라이드' },
  emphasis: { cls: 'tl-badge--emphasis', label: '⚡ 강조'  },
  demo:     { cls: 'tl-badge--demo',     label: '💻 데모'  },
}

const EMOJI = {
  slide: '📋', emphasis: '⚡', demo: '💻',
}

export default function SceneList({ scenes, currentScene, onSceneChange }) {
  const activeRef = useRef(null)

  // 자동 스크롤 기능 제거됨

  if (!scenes || scenes.length === 0) {
    return (
      <div className="tl-empty">
        <p>강의를 선택하면 타임라인이 표시됩니다</p>
      </div>
    )
  }

  return (
    <div className="tl-list">
      {scenes.map((s, i) => {
        const badge   = BADGE[s.type] ?? BADGE.slide
        const isEmph  = s.type === 'emphasis'
        const isActive = i === currentScene

        return (
          <div
            key={i}
            ref={isActive ? activeRef : null}
            className={`tl-row ${isActive ? 'tl-row--active' : ''} ${isEmph ? 'tl-row--emphasis' : ''}`}
            onClick={() => onSceneChange(i)}
          >
            <div className="tl-thumb">
              {s.image_url ? (
                <img src={s.image_url} alt={`Slide ${s.slide_number}`} className="tl-thumb-img" />
              ) : (
                EMOJI[s.type] ?? '🎬'
              )}
            </div>
            <div className="tl-info">
              <div className="tl-header">
                <div className="tl-ts">{s.timestamp}</div>
                <span className={`tl-badge ${badge.cls}`}>{badge.label}</span>
              </div>
              <div className="tl-text">{s.text}</div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
