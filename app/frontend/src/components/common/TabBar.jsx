import React from 'react'

const TABS = [
  { id: 'lectures',  icon: '🎬', label: '강의'   },
  { id: 'recommend', icon: '💡', label: '추천'   },
  { id: 'report',    icon: '📊', label: '통계'   },
]

export default function TabBar({ activeIdx, onTabClick }) {
  return (
    <nav className="tabbar">
      {TABS.map((t, i) => (
        <button
          key={t.id}
          className={`tabbar-btn${activeIdx === i ? ' tabbar-btn--active' : ''}`}
          onClick={() => onTabClick(i)}
        >
          <span className="tabbar-icon">{t.icon}</span>
          {t.label}
        </button>
      ))}
    </nav>
  )
}
