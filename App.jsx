import React, { useEffect, useMemo, useRef, useState } from 'react'

function formatWhen(d) {
  try {
    const dt = new Date(d)
    return dt.toLocaleString()
  } catch {
    return ''
  }
}

export default function App() {
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000'

  // --- chat state ---
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  // language (persisted)
  const [lang, setLang] = useState(() => localStorage.getItem('lang') || 'bn')
  useEffect(() => { localStorage.setItem('lang', lang) }, [lang])

  // conversation id (persisted)
  const [cid, setCid] = useState(() => {
    try {
      const saved = localStorage.getItem('cid')
      if (saved) return saved
      const id = (crypto?.randomUUID?.() || Math.random().toString(36).slice(2))
      localStorage.setItem('cid', id)
      return id
    } catch {
      return 'default'
    }
  })

  // conversations list for sidebar
  const [convos, setConvos] = useState([])
  const [convosLoading, setConvosLoading] = useState(false)

  const bottomRef = useRef(null)

  // --- helpers ---
  const isMobile = useMemo(() => {
    if (typeof window === 'undefined') return false
    return window.innerWidth < 900
  }, [])

  async function refreshConvos() {
    try {
      setConvosLoading(true)
      const r = await fetch(`${API_BASE}/api/conversations`)
      const json = r.headers.get('content-type')?.includes('application/json')
      const data = json ? await r.json() : []
      setConvos(Array.isArray(data) ? data : [])
    } catch {
      // ignore
    } finally {
      setConvosLoading(false)
    }
  }

  // load messages for current conversation
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/messages?cid=${encodeURIComponent(cid)}`)
        const json = r.headers.get('content-type')?.includes('application/json')
        const data = json ? await r.json() : []
        if (!cancelled) setMessages(Array.isArray(data) ? data : [])
      } catch {
        if (!cancelled) setMessages([])
      }
    })()
    return () => { cancelled = true }
  }, [API_BASE, cid])

  // load conversations initially and whenever cid changes
  useEffect(() => { refreshConvos() }, [API_BASE])
  useEffect(() => { refreshConvos() }, [cid])

  // auto-scroll to last message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text) return

    setLoading(true)
    setInput('')
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, conversationId: cid, lang })
      })
      const isJson = res.headers.get('content-type')?.includes('application/json')
      const data = isJson ? await res.json() : { error: await res.text() }
      if (!res.ok) throw new Error(data?.error || 'Request failed')
      if (data?.messages) setMessages((prev) => [...prev, ...data.messages])
    } catch (err) {
      console.error(err)
      alert(`Failed to send message: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete() {
    if (!confirm('Delete this conversation? This cannot be undone.')) return
    try {
      await fetch(`${API_BASE}/api/conversations/${cid}`, { method: 'DELETE' })
      const id = (crypto?.randomUUID?.() || Math.random().toString(36).slice(2))
      localStorage.setItem('cid', id)
      setCid(id)
      setMessages([])
    } catch (e) {
      alert('Failed to delete conversation')
    }
  }

  function handleNewChat() {
    const id = (crypto?.randomUUID?.() || Math.random().toString(36).slice(2))
    localStorage.setItem('cid', id)
    setCid(id)
    setMessages([])
  }

  function handlePickConversation(newCid) {
    const id = newCid || 'default'
    if (id === cid) return
    localStorage.setItem('cid', id)
    setCid(id)
    setMessages([])
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'stretch',
      background: '#0b1220',
      color: '#e8eefc',
      fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif'
    }}>
      {/* Top bar */}
      <div style={{
        padding: '14px 16px',
        borderBottom: '1px solid #192446',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
            <h1 style={{ margin: 0, fontSize: 22 }}>যেখানেই দরদাম, সেখানেই : Dalal</h1>
          </div>
          {!isMobile && (
            <div style={{ opacity: 0.7, fontSize: 12 }}>
              Conversation: {cid.slice(0, 8)}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <select
            aria-label="Language"
            value={lang}
            onChange={(e) => setLang(e.target.value)}
            style={{
              padding: '8px 10px',
              borderRadius: 10,
              border: '1px solid #243056',
              background: '#0f172a',
              color: '#e8eefc'
            }}
          >
            <option value="bn">বাংলা</option>
            <option value="en">English</option>
          </select>

          <button
            onClick={handleNewChat}
            style={{
              padding: '8px 10px',
              borderRadius: 10,
              border: '1px solid #243056',
              background: '#1e2a5a',
              color: '#e8eefc',
              cursor: 'pointer'
            }}
          >
            New Chat
          </button>
        </div>
      </div>

      {/* Body: sidebar + chat */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: isMobile ? '1fr' : '260px 1fr',
        gap: 0,
        minHeight: 0,
        flex: 1
      }}>
        {/* Sidebar or dropdown */}
        {isMobile ? (
          <div style={{ padding: 12, borderBottom: '1px solid #192446' }}>
            <label htmlFor="convo-dd" style={{ fontSize: 12, opacity: 0.7, marginRight: 8 }}>Conversations</label>
            <select
              id="convo-dd"
              value={cid}
              onChange={(e) => handlePickConversation(e.target.value)}
              style={{
                padding: '8px 10px',
                borderRadius: 10,
                border: '1px solid #243056',
                background: '#0f172a',
                color: '#e8eefc',
                width: '100%',
                marginTop: 6
              }}
            >
              {convos.map(c => {
                const id = c?.conversationId || 'default'
                return (
                  <option key={id} value={id}>
                    {id.slice(0, 8)} — {(c.lastText || '…').slice(0, 24)}
                  </option>
                )
              })}
            </select>
          </div>
        ) : (
          <aside style={{
            borderRight: '1px solid #192446',
            minHeight: 'calc(100vh - 56px)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8
          }}>
            <div style={{ padding: '12px 12px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: 12, opacity: 0.7 }}>Conversations</div>
              <button
                onClick={refreshConvos}
                title="Refresh"
                style={{ fontSize: 12, padding: '4px 8px', borderRadius: 8, border: '1px solid #243056', background: '#0f172a', color: '#e8eefc' }}
              >
                {convosLoading ? '...' : '↻'}
              </button>
            </div>

            <div style={{ padding: 8, overflowY: 'auto' }}>
              {convos.length === 0 && (
                <div style={{ opacity: 0.6, fontSize: 12, padding: '8px 12px' }}>
                  No conversations yet.
                </div>
              )}
              {convos.map(c => {
                const id = c?.conversationId || 'default'
                const active = id === cid
                return (
                  <button
                    key={id}
                    onClick={() => handlePickConversation(id)}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '10px 12px',
                      marginBottom: 6,
                      borderRadius: 10,
                      border: active ? '1px solid #3b82f6' : '1px solid #243056',
                      background: active ? '#102041' : '#0f172a',
                      color: '#e8eefc',
                      cursor: 'pointer'
                    }}
                    title={c.lastText || ''}
                  >
                    <div style={{ fontSize: 12, opacity: 0.8 }}>
                      {id.slice(0, 8)} • {formatWhen(c.lastAt)}
                    </div>
                    <div style={{
                      marginTop: 4,
                      fontSize: 13,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      opacity: 0.9
                    }}>
                      {c.lastText || '…'}
                    </div>
                  </button>
                )
              })}
            </div>

            <div style={{ padding: 12, marginTop: 'auto' }}>
              <button
                onClick={handleDelete}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  borderRadius: 10,
                  border: '1px solid #3a2a2a',
                  background: '#3a1e1e',
                  color: '#ffdada',
                  cursor: 'pointer'
                }}
              >
                Delete Current
              </button>
            </div>
          </aside>
        )}

        {/* Chat column */}
        <main style={{
          width: '100%',
          maxWidth: 900,
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          padding: 16,
          margin: '0 auto'
        }}>
          <div style={{
            flex: 1,
            overflowY: 'auto',
            background: '#0f172a',
            border: '1px solid #243056',
            borderRadius: 12,
            padding: 12
          }}>
            {messages.map((m) => (
              <div key={m._id} style={{
                margin: '8px 0',
                display: 'flex',
                justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start'
              }}>
                <div style={{
                  maxWidth: '80%',
                  padding: '10px 12px',
                  borderRadius: 10,
                  background: m.role === 'user' ? '#2b3a67' : '#16213e',
                  border: '1px solid #223059'
                }}>
                  <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>
                    {m.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <form onSubmit={sendMessage} style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              placeholder={
                loading
                  ? (lang === 'bn' ? 'ভাবছি…' : 'Thinking…')
                  : (lang === 'bn' ? 'প্রশ্ন লিখুন…' : 'Type your question…')
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              style={{
                flex: 1,
                padding: '12px 14px',
                borderRadius: 12,
                border: '1px solid #243056',
                background: '#0f172a',
                color: '#e8eefc',
                outline: 'none'
              }}
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '12px 16px',
                borderRadius: 12,
                border: '1px solid #243056',
                background: loading ? '#1b2a4d' : '#1e2a5a',
                color: '#e8eefc',
                cursor: 'pointer'
              }}
            >
              {loading ? (lang === 'bn' ? 'পাঠানো হচ্ছে…' : 'Sending…') : (lang === 'bn' ? 'পাঠান' : 'Send')}
            </button>
          </form>

          <div style={{ padding: 6, opacity: 0.6, fontSize: 12, textAlign: 'center' }}>
            No login • Single page • DB-backed chat • LLM-ready • বাংলা/English
          </div>
        </main>
      </div>
    </div>
  )
}