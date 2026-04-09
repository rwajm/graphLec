import { useState } from 'react'
import SceneList from './SceneList'
import GraphViewer from './GraphViewer'

const TABS = [
  { id: 'timeline', label: '타임라인' },
  { id: 'graph',    label: '그래프' },
  { id: 'info',     label: '강의 정보' },
]

export default function LecturePanel({ lecture, scenes, currentScene, onSceneChange }) {
  const [tab, setTab] = useState('timeline')

  const keywords = lecture?.keywords ?? []
  const emphasis = scenes?.filter(s => s.type === 'emphasis') ?? []

  return (
    <div className="sp-wrap">
      <div className="sp-tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`sp-tab ${tab === t.id ? 'sp-tab--active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="sp-content">
        {tab === 'timeline' && (
          <SceneList
            scenes={scenes}
            currentScene={currentScene}
            onSceneChange={onSceneChange}
          />
        )}

        {tab === 'graph' && (
          <GraphViewer lectureId={lecture?.id} />
        )}

        {tab === 'info' && (
          <div className="sp-info-merged">
            {/* 요약 섹션 */}
            <div className="sp-block">
              <h4>📌 핵심 요약</h4>
              <p>{lecture?.summary || '강의 분석이 완료되면 요약이 표시됩니다.'}</p>
            </div>

            {emphasis.length > 0 && (
              <div className="sp-block">
                <h4>⚡ 강조 구간 (음성 분석)</h4>
                <p>
                  {emphasis.map((s, i) => (
                    <span key={i}>
                      <strong style={{ color: 'var(--amber)' }}>{s.timestamp}</strong>
                      {' '}{s.text}
                      {i < emphasis.length - 1 ? ' · ' : ''}
                    </span>
                  ))}
                </p>
              </div>
            )}

            {/* 키워드 섹션 */}
            <div className="sp-block">
              <h4>🔑 주요 키워드</h4>
              <div className="sp-kw-cloud">
                {keywords.length === 0
                  ? <span className="sp-empty">분석 완료 후 표시됩니다.</span>
                  : keywords.map((kw, i) => (
                      <span key={i} className="sp-kw">{kw}</span>
                    ))
                }
              </div>
            </div>

            {/* 강의 메타 정보 섹션 */}
            <div className="sp-block">
              <h4>ℹ️ 강의 상세 정보</h4>
              <div className="sp-stat-row">
                <div className="sp-stat-box">
                  <div className="sp-stat-num" style={{ color: 'var(--blue)' }}>{scenes?.length || 0}</div>
                  <div className="sp-stat-lbl">장면 전환</div>
                </div>
                <div className="sp-stat-box">
                  <div className="sp-stat-num" style={{ color: 'var(--amber)' }}>{emphasis.length}</div>
                  <div className="sp-stat-lbl">강조 구간</div>
                </div>
                <div className="sp-stat-box">
                  <div className="sp-stat-num" style={{ color: 'var(--green)' }}>
                    {lecture ? '94%' : '—'}
                  </div>
                  <div className="sp-stat-lbl">STT 신뢰도</div>
                </div>
              </div>
              {lecture && (
                <div className="sp-info-text" style={{ marginTop: '12px' }}>
                  <strong>강의자:</strong> {lecture.instructor_name}<br />
                  <strong>카테고리:</strong> {lecture.category}<br />
                  <strong>업로드:</strong> {new Date(lecture.created_at).toLocaleDateString('ko-KR')}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
