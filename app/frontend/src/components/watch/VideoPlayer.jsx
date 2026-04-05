import { useState, useRef, useEffect, useCallback } from 'react'

function secToTs(s) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`
}

function tsToSec(ts = '00:00') {
  if (typeof ts === 'number') return ts
  const [m, s] = ts.split(':').map(Number)
  return m * 60 + s
}

export default function VideoPlayer({ lecture, scenes = [], currentScene, seekTo, onSceneChange, isMini = false }) {
  const videoRef = useRef(null)
  const trackRef = useRef(null)
  
  const [playing, setPlaying]       = useState(false)
  const [progress, setProgress]     = useState(0)
  const [duration, setDuration]     = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  const togglePlay = () => {
    if (!videoRef.current) return
    if (playing) videoRef.current.pause()
    else         videoRef.current.play()
  }

  const handleTimeUpdate = () => {
    if (!videoRef.current || isDragging) return // 드래그 중일 때는 비디오의 자연 갱신 무시
    const cur = videoRef.current.currentTime
    const dur = videoRef.current.duration
    setCurrentTime(cur)
    setProgress((cur / dur) * 100 || 0)

    if (scenes.length > 0) {
      const activeSceneIndex = [...scenes].reverse().findIndex(s => tsToSec(s.timestamp) <= cur)
      if (activeSceneIndex !== -1) {
        const actualIndex = scenes.length - 1 - activeSceneIndex
        if (actualIndex !== currentScene) onSceneChange(actualIndex)
      }
    }
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration)
  }

  // 드래그 로직
  const calculateProgress = useCallback((clientX) => {
    if (!trackRef.current || !duration) return null
    const rect = trackRef.current.getBoundingClientRect()
    let pct = (clientX - rect.left) / rect.width
    pct = Math.max(0, Math.min(1, pct)) // 0~1 사이 고정
    return pct
  }, [duration])

  const handleMouseDown = (e) => {
    e.preventDefault() // 텍스트 선택 및 기본 드래그 앤 드롭 방지
    setIsDragging(true)
    const pct = calculateProgress(e.clientX)
    if (pct !== null) {
      setProgress(pct * 100)
      setCurrentTime(pct * duration)
    }
  }

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return
      e.preventDefault() // 드래그 중 텍스트 선택 방지
      const pct = calculateProgress(e.clientX)
      if (pct !== null) {
        setProgress(pct * 100)
        setCurrentTime(pct * duration)
        // (선택) 드래그 중에도 영상을 동기화하고 싶다면 주석 해제 (약간의 버벅임이 있을 수 있음)
        // if (videoRef.current) videoRef.current.currentTime = pct * duration
      }
    }

    const handleMouseUp = (e) => {
      if (isDragging) {
        setIsDragging(false)
        const pct = calculateProgress(e.clientX)
        if (pct !== null && videoRef.current) {
          videoRef.current.currentTime = pct * duration
        }
      }
    }

    if (isDragging) {
      // 드래그 중에는 전체 문서의 텍스트 선택을 막아 더 깔끔한 UX 제공
      document.body.style.userSelect = 'none'
      window.addEventListener('mousemove', handleMouseMove, { passive: false })
      window.addEventListener('mouseup', handleMouseUp)
    } else {
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup',   handleMouseUp)
    }

    return () => {
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup',   handleMouseUp)
    }
  }, [isDragging, calculateProgress, duration])

  // 외부(목록 클릭 등)에서 명시적인 점프 요청이 왔을 때 영상 이동
  useEffect(() => {
    if (!videoRef.current || !seekTo || !scenes[seekTo.index]) return
    const targetSec = tsToSec(scenes[seekTo.index].timestamp)
    videoRef.current.currentTime = targetSec
    // 점프 시 자동 재생을 원한다면: videoRef.current.play()
  }, [seekTo, scenes])

  if (!lecture) {
    return (
      <div className="vp-placeholder">
        <div className="vp-placeholder-icon">🎬</div>
        <p>강의 데이터를 불러오는 중입니다...</p>
      </div>
    )
  }

  return (
    <div className={`vp-wrap ${isMini ? 'vp-wrap--mini' : ''}`}>
      <div className="vp-video-container">
        <video
          ref={videoRef}
          src={lecture.video_url}
          className="vp-video-element"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onClick={togglePlay}
        />
        {!playing && !isDragging && (
          <div className="vp-overlay" onClick={togglePlay}>
            <div className="vp-play-icon">▶</div>
          </div>
        )}

        <div className="vp-controls">
          <div 
            className="vp-track" 
            ref={trackRef}
            onMouseDown={handleMouseDown}
            style={{ cursor: isDragging ? 'grabbing' : 'pointer' }}
          >
            <div className="vp-fill" style={{ width: `${progress}%`, transition: isDragging ? 'none' : 'width 0.1s linear' }}>
              <div className="vp-thumb" style={{ transform: isDragging ? 'scale(1.2)' : 'scale(1)' }} />
            </div>
            {scenes.map((s, i) => {
              if (!duration) return null
              const pct = (tsToSec(s.timestamp) / duration) * 100
              const cls = { slide: 'vp-marker--slide', emphasis: 'vp-marker--emphasis', demo: 'vp-marker--demo' }[s.type] ?? 'vp-marker--slide'
              return (
                <div
                  key={i}
                  className={`vp-marker ${cls} ${i === currentScene ? 'vp-marker--active' : ''}`}
                  style={{ left: `${pct}%` }}
                  title={s.text}
                />
              )
            })}
          </div>
          <div className="vp-ctrl-row">
            <button className="vp-cbtn" onClick={() => { if(videoRef.current) videoRef.current.currentTime -= 10 }}>↺</button>
            <button className="vp-cbtn" onClick={togglePlay}>{playing ? '⏸' : '▶'}</button>
            <button className="vp-cbtn" onClick={() => { if(videoRef.current) videoRef.current.currentTime += 10 }}>↻</button>
            <span className="vp-mono">{secToTs(currentTime)}</span>
            <span className="vp-muted">/ {secToTs(duration)}</span>
            <div style={{ flex: 1 }} />
            <span className="vp-scene-lbl">Scene <strong>{currentScene + 1}</strong></span>
          </div>
        </div>
      </div>
    </div>
  )
}
