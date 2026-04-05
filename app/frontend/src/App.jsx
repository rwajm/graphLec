import { useCallback } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import LecturesPage  from './pages/LecturesPage'
import RecommendPage from './pages/RecommendPage'
import ReportPage    from './pages/ReportPage'
import LecturePage   from './pages/LecturePage'
import TabBar        from './components/common/TabBar'

const PAGES = ['lectures', 'recommend', 'report']

function pathToIdx(pathname) {
  if (pathname === '/' || pathname.startsWith('/lectures')) {
    // /lectures/:id 도 index 0으로 처리 (TabBar 표시 여부는 별도 판단)
    return 0
  }
  if (pathname.startsWith('/recommend')) return 1
  if (pathname.startsWith('/report'))    return 2
  return 0
}

/** /lectures/:id 여부 — 탭 바를 숨겨야 하는 독립 페이지 */
function isLectureDetailPage(pathname) {
  return /^\/lectures\/\w+/.test(pathname) && !pathname.endsWith('/lectures') && pathname !== '/lectures/'
}

function MainLayout() {
  const navigate = useNavigate()
  const location = useLocation()

  const handleNavigate = useCallback(({ page, lectureId = null }) => {
    if (page === 'lecture' && lectureId != null) {
      navigate(`/lectures/${lectureId}`)
      return
    }
    navigate(`/${page}`)
  }, [navigate])

  const activeIdx = pathToIdx(location.pathname)
  const hideTabBar = isLectureDetailPage(location.pathname)

  return (
    <div className="app-shell" style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div className="main-content" style={{ flex: 1, overflowY: 'auto' }}>
        <Routes>
          <Route path="/"            element={<LecturesPage onNavigate={handleNavigate} />} />
          <Route path="/lectures"    element={<LecturesPage onNavigate={handleNavigate} />} />
          <Route path="/lectures/:id" element={<LecturePage onNavigate={handleNavigate} />} />
          <Route path="/recommend"   element={<RecommendPage onNavigate={handleNavigate} />} />
          <Route path="/report"      element={<ReportPage />} />
          {/* 404 fallback */}
          <Route path="*"            element={<LecturesPage onNavigate={handleNavigate} />} />
        </Routes>
      </div>

      {!hideTabBar && (
        <TabBar 
          activeIdx={activeIdx} 
          onTabClick={(idx) => navigate(`/${PAGES[idx]}`)} 
        />
      )}
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <MainLayout />
    </BrowserRouter>
  )
}
