import { useState } from 'react'

const TABS = [
  { id: 'usage',    label: '이용 통계'   },
  { id: 'lectures', label: '강의 통계'   },
  { id: 'query',    label: '질의 분석'   },
]

// 더미 데이터 — 백엔드 연동 전
const DUMMY_USAGE = {
  totalLectures:  24,
  totalQueries:   1482,
  totalUsers:     318,
  avgQueryPerDay: 49,
}

const DUMMY_CATEGORY = [
  { label: '컴퓨터 과학',    count: 10, pct: 42 },
  { label: '데이터 사이언스', count: 7,  pct: 29 },
  { label: '소프트웨어 공학', count: 5,  pct: 21 },
  { label: '수학',           count: 2,  pct: 8  },
]

const DUMMY_QUERY_INTENTS = [
  { label: '강의 추천', count: 834, pct: 56 },
  { label: '내용 질의', count: 462, pct: 31 },
  { label: '일반 질문', count: 186, pct: 13 },
]

const DUMMY_TOP_QUERIES = [
  '운영체제에서 데드락이란?',
  '동적 프로그래밍 개념 설명',
  '선형대수 관련 강의 추천',
  '최근 업로드한 강의 추천',
  'CNN 강의에서 풀링 레이어란?',
]

export default function ReportPage() {
  const [tab, setTab] = useState('usage')

  return (
    <div className="report-page">
      <div className="report-tab-bar">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`report-tab-btn${tab === t.id ? ' report-tab-btn--active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="report-content">

        {/* ── 이용 통계 ── */}
        {tab === 'usage' && (
          <>
            <div className="report-stat-grid content-max">
              <StatCard label="전체 강의" val={DUMMY_USAGE.totalLectures} color="var(--blue)"   sub="업로드 완료" />
              <StatCard label="전체 질의" val={DUMMY_USAGE.totalQueries.toLocaleString()}  color="var(--green)"  sub="누적 질의 수" />
              <StatCard label="전체 사용자" val={DUMMY_USAGE.totalUsers}  color="var(--purple)" sub="가입자 수" />
              <StatCard label="일평균 질의" val={DUMMY_USAGE.avgQueryPerDay} color="var(--amber)" sub="최근 30일" />
            </div>
          </>
        )}

        {/* ── 강의 통계 ── */}
        {tab === 'lectures' && (
          <div className="report-section content-max">
            <div className="report-sec-title">카테고리별 강의 분포</div>
            <div className="report-bar-list">
              {DUMMY_CATEGORY.map((c, i) => (
                <div key={i} className="report-bar-row">
                  <div className="report-bar-label">{c.label}</div>
                  <div className="report-bar-track">
                    <div className="report-bar-fill" style={{ width: `${c.pct}%` }} />
                  </div>
                  <div className="report-bar-val">{c.count}개 ({c.pct}%)</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── 질의 분석 ── */}
        {tab === 'query' && (
          <>
            <div className="report-section content-max">
              <div className="report-sec-title">질의 유형 분포</div>
              <div className="report-bar-list">
                {DUMMY_QUERY_INTENTS.map((q, i) => (
                  <div key={i} className="report-bar-row">
                    <div className="report-bar-label">{q.label}</div>
                    <div className="report-bar-track">
                      <div className="report-bar-fill" style={{ width: `${q.pct}%` }} />
                    </div>
                    <div className="report-bar-val">{q.count.toLocaleString()}회 ({q.pct}%)</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="report-section content-max">
              <div className="report-sec-title">자주 묻는 질문 TOP 5</div>
              <ol className="report-top-list">
                {DUMMY_TOP_QUERIES.map((q, i) => (
                  <li key={i} className="report-top-item">
                    <span className="report-top-rank">{i + 1}</span>
                    <span>{q}</span>
                  </li>
                ))}
              </ol>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function StatCard({ label, val, color, sub }) {
  return (
    <div className="report-stat-card">
      <div className="report-stat-lbl">{label}</div>
      <div className="report-stat-val" style={{ color }}>{val}</div>
      <div className="report-stat-sub">{sub}</div>
    </div>
  )
}
