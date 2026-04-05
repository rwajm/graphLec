import { useState, useRef, useEffect } from 'react'
import { askQa } from '../../lib/api'

/**
 * src/components/chat/ChatPanel.jsx
 * LecturePage 전용 — 영상 질의 사이드패널
 *
 * lecture        — 현재 강의 객체
 * onJumpToScene  — 타임스탬프 클릭 시 VideoPlayer + SceneList 동기화
 */

const INIT_MSG = {
  id: 0, role: 'assistant',
  content: '강의에 대해 질문해보세요.',
  refs: [],
}

export default function ChatPanel({ lecture, onJumpToScene, onClose }) {
  const [messages, setMessages] = useState([INIT_MSG])
  const [input,    setInput]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => { setMessages([INIT_MSG]) }, [lecture?.id])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  async function send() {
    const question = input.trim()
    if (!question || loading || !lecture?.id) return
    setInput('')
    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: question, refs: [] }])
    setLoading(true)
    
    try {
      const res = await askQa(lecture.id, question)
      
      // Extract refs from returned timestamps array if available
      const refs = (res.timestamps || []).map(t => ({
        timestamp: t.start != null ? `${Math.floor(t.start/60).toString().padStart(2,'0')}:${Math.floor(t.start%60).toString().padStart(2,'0')}` : null,
        label: t.label
      })).filter(r => r.timestamp);

      setMessages(prev => [...prev, {
        id: Date.now() + 1, role: 'assistant',
        content: res.answer || '답변을 생성하지 못했습니다.',
        refs: refs,
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1, role: 'assistant',
        content: `오류: ${e.message}`, refs: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-header-title">Chat</span>
        <button className="chat-close-btn" onClick={onClose} title="채팅창 닫기">
          ✕
        </button>
      </div>

      <div className="chat-messages">
        {messages.map(msg =>
          msg.role === 'user' ? (
            <div key={msg.id} className="chat-msg-user">
              <div className="chat-bubble-user">{msg.content}</div>
            </div>
          ) : (
            <div key={msg.id} className="chat-msg-ai">
              <div className="chat-bubble-ai">
                <span dangerouslySetInnerHTML={{ __html: msg.content }} />
                {msg.refs?.length > 0 && (
                  <div className="chat-refs">
                    {msg.refs.map((ref, i) => {
                      const idx = lecture?.scenes?.findIndex(s => s.timestamp === ref.timestamp) ?? -1
                      return (
                        <button key={i} className="chat-ref-btn"
                          disabled={idx < 0}
                          onClick={() => idx >= 0 && onJumpToScene?.(idx)}
                        >
                          ▶ {ref.timestamp}{ref.label ? ` ${ref.label}` : ''}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          )
        )}
        {loading && <div className="chat-loading">답변 생성 중...</div>}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          placeholder="강의 내용에 대해 질문해보세요..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          disabled={loading}
        />
        <button className="chat-send-btn" onClick={send} disabled={loading || !input.trim()}>↑</button>
      </div>
    </div>
  )
}