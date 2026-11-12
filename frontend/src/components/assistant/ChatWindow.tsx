import { useEffect, useMemo, useRef, useState } from 'react'
import { useAssistant } from './AssistantProvider'
import { Markdown } from './Markdown'

const SUGGESTIONS = [
  'How does VoxSentinel AI work?',
  'How can I analyze an audio file?',
  'How do I detect an AI-generated voice?',
  'What is speaker verification?',
  'How does scam risk detection work?',
  'How can I understand my analysis report?',
  'How do I export my analysis history?',
]

const AVATAR = '/assistant/avatar-192.png'

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  )
}

function MinimiseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M5 12h14" />
    </svg>
  )
}

function ResetIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 11a8 8 0 1 0-2.6 6.4" />
      <path d="M20 4v7h-7" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 12l16-8-6 16-3.2-6.4L4 12z" />
    </svg>
  )
}

function Avatar({ size = 36 }: { size?: number }) {
  return (
    <img
      src={AVATAR}
      alt=""
      width={size}
      height={size}
      className="assistant-avatar-img"
      style={{ width: size, height: size }}
    />
  )
}

export function ChatWindow() {
  const { messages, pending, error, dismissError, send, reset, context, status, minimise, closeChat, maxChars } =
    useAssistant()
  const [draft, setDraft] = useState('')
  const [sendingFromSuggestion, setSendingFromSuggestion] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const online = status ? status.available : true
  const busy = pending || sendingFromSuggestion

  const suggested = useMemo(() => {
    const list = [context.suggestion, ...SUGGESTIONS]
    return [...new Set(list)].slice(0, 7)
  }, [context.suggestion])

  // Keep the newest message in view.
  useEffect(() => {
    const node = scrollRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [messages.length, pending, error])

  // Focus the composer when the window opens, and restore a failed question.
  useEffect(() => {
    inputRef.current?.focus()
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ text?: string }>).detail
      if (detail?.text) setDraft(detail.text)
    }
    window.addEventListener('voxsentinel:assistant-failed', handler)
    return () => window.removeEventListener('voxsentinel:assistant-failed', handler)
  }, [])

  // Esc closes; keep Tab inside the dialog while it is open.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        minimise()
        return
      }
      if (event.key !== 'Tab') return
      const focusables = panelRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), textarea, [href], input, [tabindex]:not([tabindex="-1"])',
      )
      if (!focusables?.length) return
      const first = focusables[0]
      const last = focusables[focusables.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [minimise])

  const submit = async (text: string) => {
    const value = text.trim()
    if (!value || busy) return
    if (value.length > maxChars) {
      setDraft(value.slice(0, maxChars))
      return
    }
    setDraft('')
    await send(value)
    inputRef.current?.focus()
  }

  const onSuggestion = async (question: string) => {
    setSendingFromSuggestion(true)
    try {
      await send(question)
    } finally {
      setSendingFromSuggestion(false)
    }
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submit(draft)
    }
  }

  return (
    <div
      ref={panelRef}
      className="assistant-panel"
      role="dialog"
      aria-modal="false"
      aria-label="VoxSentinel Assistant"
    >
      <header className="assistant-panel-header">
        <div className="assistant-brand">
          <span className="assistant-avatar-ring">
            <Avatar size={38} />
            <span className={`assistant-presence ${online ? 'is-online' : 'is-offline'}`} aria-hidden="true" />
          </span>
          <span className="assistant-brand-text">
            <span className="assistant-title">VoxSentinel Assistant</span>
            <span className="assistant-status">
              {online ? 'Online · powered by Gemini' : 'Assistant unavailable'}
              {status?.model ? <span className="assistant-model">{status.model}</span> : null}
            </span>
          </span>
        </div>
        <div className="assistant-controls">
          <button
            type="button"
            className="assistant-icon-button"
            onClick={reset}
            title="Clear conversation"
            aria-label="Clear conversation"
            disabled={busy || messages.length === 0}
          >
            <ResetIcon />
          </button>
          <button
            type="button"
            className="assistant-icon-button"
            onClick={minimise}
            title="Minimise"
            aria-label="Minimise assistant"
          >
            <MinimiseIcon />
          </button>
          <button
            type="button"
            className="assistant-icon-button"
            onClick={closeChat}
            title="Close"
            aria-label="Close assistant"
          >
            <CloseIcon />
          </button>
        </div>
      </header>

      <div className="assistant-context-strip">
        <span className="assistant-context-dot" aria-hidden="true" />
        Viewing: <strong>{context.page}</strong>
      </div>

      <div className="assistant-body" ref={scrollRef} aria-live="polite">
        {messages.length === 0 ? (
          <div className="assistant-welcome">
            <Avatar size={56} />
            <p className="assistant-welcome-title">
              Hi! I&apos;m VoxSentinel Assistant. I&apos;m here to help you understand and use VoxSentinel AI. What
              would you like to explore?
            </p>
            <p className="assistant-welcome-hint">{context.greeting}</p>
            <div className="assistant-suggestions">
              {suggested.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="assistant-suggestion"
                  onClick={() => void onSuggestion(question)}
                  disabled={busy}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`assistant-row ${message.role === 'user' ? 'is-user' : 'is-assistant'}`}
            >
              {message.role === 'assistant' && (
                <span className="assistant-row-avatar">
                  <Avatar size={28} />
                </span>
              )}
              <div className={`assistant-bubble ${message.role === 'user' ? 'is-user' : 'is-assistant'}`}>
                {message.role === 'assistant' ? <Markdown text={message.content} /> : message.content}
              </div>
            </div>
          ))
        )}

        {busy && (
          <div className="assistant-row is-assistant">
            <span className="assistant-row-avatar">
              <Avatar size={28} />
            </span>
            <div className="assistant-bubble is-assistant assistant-typing" aria-label="Assistant is typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}

        {error && (
          <div className="assistant-error" role="alert">
            <p>{error}</p>
            <button type="button" className="assistant-error-dismiss" onClick={dismissError}>
              Dismiss
            </button>
          </div>
        )}
      </div>

      <form
        className="assistant-composer"
        onSubmit={(event) => {
          event.preventDefault()
          void submit(draft)
        }}
      >
        <textarea
          ref={inputRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value.slice(0, maxChars))}
          onKeyDown={onKeyDown}
          rows={1}
          placeholder="Ask about any VoxSentinel feature…"
          aria-label="Message VoxSentinel Assistant"
          maxLength={maxChars}
        />
        <button
          type="submit"
          className="assistant-send"
          disabled={busy || draft.trim().length === 0}
          aria-label="Send message"
          title="Send (Enter)"
        >
          <SendIcon />
        </button>
      </form>
      <p className="assistant-footnote">
        Enter to send · Shift+Enter for a new line · Answers come from Gemini and may be imperfect
      </p>
    </div>
  )
}
