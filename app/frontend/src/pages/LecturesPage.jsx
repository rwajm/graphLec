import { useEffect, useRef, useState } from 'react'
import { getLectureStatus, listLectures, uploadLecture, deleteLecture, retryLecture } from '../lib/api'

const STAGE_LABELS = ['장면 감지', '음성 분석', 'STT 전사', '통합', '요약 생성']
const STAGE_KEYS   = ['scene', 'voice', 'stt', 'integrate', 'summarize']

const STATUS_MAP = {
  done:       { label: '분석 완료', cls: 'status-done' },
  processing: { label: '분석 중',   cls: 'status-proc' },
  pending:    { label: '대기',       cls: 'status-wait' },
  error:      { label: '오류',       cls: 'status-err'  },
}

const THUMB_COLOR = {
  '컴퓨터 과학': '#1e3a5f', '데이터 사이언스': '#1a3d2b',
  '소프트웨어 공학': '#2d1f3d', '수학': '#3d2a1a',
}
const THUMB_ICON = {
  '컴퓨터 과학': '🧠', '데이터 사이언스': '📊',
  '소프트웨어 공학': '⚙️', '수학': '📐',
}

export default function LecturesPage({ onNavigate }) {
  const [lectures,    setLectures]    = useState([])
  const [title,       setTitle]       = useState('')
  const [category,    setCategory]    = useState('컴퓨터 과학')
  const [description, setDescription] = useState('')
  const [file,        setFile]        = useState(null)
  const [dragOver,    setDragOver]    = useState(false)
  const [uploading,   setUploading]   = useState(null)
  const [error,       setError]       = useState('')

  const fileRef    = useRef()
  const scrollRef  = useRef()   // 페이지 스크롤 컨테이너
  const listRef    = useRef()   // 강의 목록 섹션
  const eventSources = useRef({}) // { job_id: EventSource }

  // 주기적으로(또는 처음 로드 시) 진행 중인 작업에 대해 SSE 연결을 맺는 함수
  const setupSSEForJob = (job_id) => {
    if (eventSources.current[job_id]) return // 이미 연결되어 있음

    console.log(`--- [SSE] Connecting to stream for job ${job_id} ---`)
    const eventSource = new EventSource(`/api/jobs/${job_id}/stream`)
    eventSources.current[job_id] = eventSource

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.error) {
           console.error("SSE Error from server:", data.error)
           eventSource.close()
           delete eventSources.current[job_id]
           return
        }

        // 특정 강의 상태 및 상세 단계 업데이트
        setLectures(prev => prev.map(lec => {
          if (lec.id === data.job_id) {
            // 백엔드에서 전달받은 pipeline_stages 배열을 그대로 사용
            let stages = data.pipeline_stages
            
            // 만약 비어있다면 기본 회색 칩으로 매핑
            if (!stages || stages.length === 0) {
               stages = STAGE_KEYS.map((key) => ({ stage: key, status: 'wait' }))
            }

            return { 
              ...lec, 
              status: data.lecture_status,
              current_stage: data.current_stage,
              error_message: data.error_message || lec.error_message,
              pipeline_stages: stages
            }
          }
          return lec
        }))

        // 완료 또는 에러 상태면 연결 종료
        if (data.lecture_status === 'done' || data.lecture_status === 'error') {
           console.log(`--- [SSE] Closing stream for job ${job_id} (Terminal state) ---`)
           eventSource.close()
           delete eventSources.current[job_id]
        }

      } catch (err) {
        console.error("--- [SSE] Parse error:", err)
      }
    }

    eventSource.onerror = (err) => {
      console.error(`--- [SSE] Connection error for job ${job_id}:`, err)
      eventSource.close()
      delete eventSources.current[job_id]
    }
  }

  // 처음 로드 시 목록을 가져오고, 진행 중인 작업들에 대해 SSE 연결 시작
  useEffect(() => {
    listLectures().then(data => {
      setLectures(data)
      data.forEach(lec => {
        if (lec.status === 'running' || lec.status === 'pending') {
          setupSSEForJob(lec.id)
        }
      })
    }).catch(e => setError(String(e.message || e)))

    // 언마운트 시 모든 SSE 연결 정리
    return () => {
      Object.values(eventSources.current).forEach(source => source.close())
      eventSources.current = {}
    }
  }, [])

  function setFileWithAutoTitle(f) {
    if (!f) return
    setFile(f)
    setTitle(prev => prev.trim() ? prev : f.name.replace(/\.[^.]+$/, ''))
  }

  async function handleSubmit() {
    if (uploading || !file) return
    setError('')

    // 업로드 시작 → 목록 섹션으로 스크롤
    listRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })

    try {
      const created   = await uploadLecture({ title: title || file.name, category, description, file })
      setLectures(prev => [{ ...created, status: 'pending', pipeline_stages: [] }, ...prev])
      
      // 방금 업로드한 작업에 대해 SSE 스트리밍 연결 시작
      setupSSEForJob(created.id)
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setTitle(''); setCategory('컴퓨터 과학'); setDescription(''); setFile(null)
    }
  }

  async function handleDelete(id, e) {
    e.stopPropagation()
    if (!confirm('이 강의를 삭제하시겠습니까?')) return
    try {
      await deleteLecture(id)
      setLectures(prev => prev.filter(l => l.id !== id))
    } catch (err) { alert(`삭제 실패: ${err.message}`) }
  }

  async function handleRetry(id, e) {
    e.stopPropagation()
    if (!confirm('분석을 다시 시도하시겠습니까?')) return
    try {
      await retryLecture(id)
      // 재시도 요청 후 SSE 새로 연결
      setLectures(prev => prev.map(l => l.id === id ? { ...l, status: 'pending', pipeline_stages: [] } : l))
      setupSSEForJob(id)
    } catch (err) { alert(`재시도 실패: ${err.message}`) }
  }

  return (
    <div className="up-page" ref={scrollRef}>

      {/* ── 섹션 1: 업로드 폼 (뷰포트 전체 높이) ── */}
      <section className="up-form-section">
        <div className="up-form-inner">
          <div className="up-form-logo">
            Graph<span>Lec</span>
          </div>
          <p className="up-form-sub">강의 영상을 업로드하면 자동으로 분석합니다.</p>

          {error && <p className="up-error">{error}</p>}

          {/* 드롭존 */}
          <div
            className={`up-dropzone${dragOver ? ' up-dropzone--drag' : ''}${file ? ' up-dropzone--file' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); setFileWithAutoTitle(e.dataTransfer.files[0]) }}
          >
            <div className="up-dz-icon">{file ? '✅' : '🎬'}</div>
            {file
              ? <><p className="up-dz-text"><strong>{file.name}</strong></p><p className="up-dz-sub">{(file.size/1024/1024).toFixed(1)} MB</p></>
              : <><p className="up-dz-text"><strong>드래그하거나 클릭</strong></p><p className="up-dz-sub">MP4, MOV, AVI · 최대 2GB</p></>
            }
            <input ref={fileRef} type="file" accept="video/*" style={{ display: 'none' }}
              onChange={e => setFileWithAutoTitle(e.target.files[0])} />
          </div>

          {/* 폼 필드 */}
          <div className="up-field-group">
            <div className="up-field">
              <label className="up-label">강의 제목</label>
              <input className="up-input" value={title} onChange={e => setTitle(e.target.value)} placeholder="제목을 입력하세요" />
            </div>
            <div className="up-field">
              <label className="up-label">카테고리</label>
              <select className="up-select" value={category} onChange={e => setCategory(e.target.value)}>
                <option>컴퓨터 과학</option>
                <option>수학</option>
                <option>데이터 사이언스</option>
                <option>소프트웨어 공학</option>
              </select>
            </div>
            <div className="up-field">
              <label className="up-label">강의 설명 (선택)</label>
              <textarea className="up-textarea" value={description} onChange={e => setDescription(e.target.value)} placeholder="강의 내용을 간략히 설명하세요..." />
            </div>
          </div>

          <button className="up-submit-btn" onClick={handleSubmit} disabled={!!uploading || !file}>
            {uploading ? '업로드 중...' : '업로드 시작'}
          </button>

          {/* 목록으로 내려가는 힌트 */}
          <button className="up-scroll-hint" onClick={() => listRef.current?.scrollIntoView({ behavior: 'smooth' })}>
            강의 목록 보기 ↓
          </button>
        </div>
      </section>

      {/* ── 섹션 2: 강의 목록 ── */}
      <section className="up-list-section" ref={listRef}>
        <div className="up-list-header content-max">
          <div className="up-list-title">강의 목록</div>
          <span className="up-list-count">{lectures.length}개</span>
        </div>

        <div className="up-list content-max">
          {lectures.length === 0 && <div className="up-empty">업로드된 강의가 없습니다</div>}
          {lectures.map(lec => {
            const st        = STATUS_MAP[lec.status] ?? STATUS_MAP.pending
            const thumbBg   = THUMB_COLOR[lec.category] ?? '#1e2333'
            const thumbIcon = THUMB_ICON[lec.category] ?? '🎬'
            const isActive  = uploading?.id === lec.id

            return (
              <div key={lec.id}>
                <div
                  className={`upload-row${lec.status === 'done' ? ' upload-row--done' : ''}`}
                  onClick={() => lec.status === 'done' && onNavigate?.({ page: 'lecture', lectureId: lec.id })}
                >
                  <div className="upload-row-thumb" style={{ background: thumbBg }}>
                    <span className="upload-row-thumb-icon">{thumbIcon}</span>
                  </div>
                  <div className="upload-row-main">
                    <div className="upload-row-title">{lec.title}</div>
                    <div className="upload-row-meta">
                      <span className="upload-row-cat">{lec.category}</span>
                      {lec.tags?.slice(0, 3).map(tag => (
                        <span key={tag} className="upload-row-tag">{tag}</span>
                      ))}
                    </div>
                  </div>
                  <div className="upload-row-date">
                    {new Date(lec.created_at).toLocaleDateString('ko-KR')}
                  </div>
                  <div className="upload-row-status">
                    <span className={`upload-status-badge ${st.cls}`}>{st.label}</span>
                  </div>
                  <div className="upload-row-actions">
                    {lec.status === 'error' && (
                      <button className="upload-btn-retry" onClick={e => handleRetry(lec.id, e)}>재시도</button>
                    )}
                    {lec.status !== 'done' && (
                      <button className="upload-btn-delete" onClick={e => handleDelete(lec.id, e)}>삭제</button>
                    )}
                    {lec.status === 'done' && <span className="upload-row-arrow">→</span>}
                  </div>
                </div>

                {/* 인라인 파이프라인 진행 */}
                {(lec.status === 'running' || lec.status === 'pending' || lec.status === 'error') && lec.pipeline_stages && (
                  <div className="upload-pipe" style={{ margin: '-4px 0 4px', borderTopLeftRadius: 0, borderTopRightRadius: 0 }}>
                    <div className="upload-pipe-track">
                      {/* 파이프라인 진행 상태에 따른 진행률 바 계산 */}
                      <div className="upload-pipe-fill" style={{ 
                        width: lec.status === 'done' ? '100%' : 
                               lec.status === 'error' ? '0%' : 
                               `${Math.max(0, lec.pipeline_stages.filter(s => s.status === 'done').length * 20 + 
                                  (lec.pipeline_stages.some(s => s.status === 'run') ? 10 : 0))}%` 
                      }} />
                    </div>
                    <div className="upload-pipe-stages">
                      {STAGE_LABELS.map((lbl, i) => {
                        const stageStatus = lec.pipeline_stages.find(s => s.stage === STAGE_KEYS[i])?.status ?? 'wait'
                        return (
                          <span key={i} className={`upload-chip${
                            stageStatus === 'done' ? ' upload-chip--done' :
                            stageStatus === 'run'  ? ' upload-chip--run'  : ''
                          }`}>
                            {stageStatus === 'done' ? '✓ ' : stageStatus === 'run' ? '↻ ' : ''}{lbl}
                          </span>
                        )
                      })}
                    </div>
                    {lec.status === 'error' && (
                      <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--red)' }}>
                        오류 내용: {lec.error_message}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}