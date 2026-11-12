import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useLocation } from 'react-router-dom'
import {
  ASSISTANT_MAX_MESSAGE_CHARS,
  fetchAssistantStatus,
  sendAssistantMessage,
  type AssistantMessage,
  type AssistantStatus,
} from '../../lib/assistant'

/**
 * Holds one conversation for the whole browser session.
 *
 * The provider is mounted above the router's pages (in `main.tsx`), so opening
 * another route does not unmount the chat: the transcript, the open state and
 * any in-flight request survive navigation. Messages are mirrored into
 * sessionStorage so a reload inside the same tab keeps the conversation.
 */

const SESSION_KEY = 'voxsentinel.assistant.session.v1'
const SEEN_KEY = 'voxsentinel.assistant.seen'

export interface PageContext {
  route: string
  page: string
  module: string
  greeting: string
  suggestion: string
}

const PAGE_CONTEXTS: Array<Omit<PageContext, 'route'> & { match: string[] }> = [
  {
    match: ['/analyze'],
    page: 'Analyze Audio',
    module: 'Analyze Audio',
    greeting: 'You are on Analyze Audio. I can walk you through uploading a file and reading the result.',
    suggestion: 'How do I analyze an audio file?',
  },
  {
    match: ['/record'],
    page: 'Live Recording',
    module: 'Live Recorder',
    greeting: 'You are on Live Recording. Ask me how the provisional risk reading works.',
    suggestion: 'How does the live recorder work?',
  },
  {
    match: ['/verify'],
    page: 'Speaker Verification',
    module: 'Speaker Verification',
    greeting: 'I can explain how to compare two voice samples, and what the similarity score does not prove.',
    suggestion: 'What is speaker verification?',
  },
  {
    match: ['/history'],
    page: 'Analysis History',
    module: 'History',
    greeting: 'You are on Analysis History. I can help you find a past analysis or export the list.',
    suggestion: 'How do I export my analysis history?',
  },
  {
    match: ['/protection'],
    page: 'Family Emergency & Scam Protection',
    module: 'Risk Analysis',
    greeting: 'You are on Scam Protection. Ask me how the transcript, indicators and risk level fit together.',
    suggestion: 'How does scam risk detection work?',
  },
  {
    match: ['/evaluation'],
    page: 'Evaluation',
    module: 'Evaluation',
    greeting: 'You are on Evaluation. I can explain precision, recall, F1-score and ROC-AUC.',
    suggestion: 'What do precision and recall mean here?',
  },
  {
    match: ['/dashboard'],
    page: 'Dashboard',
    module: 'Overview',
    greeting: 'You are on the Dashboard. Ask me about the activity summary or where to start.',
    suggestion: 'How does VoxSentinel AI work?',
  },
  {
    match: ['/'],
    page: 'Landing page',
    module: 'Overview',
    greeting: 'Welcome. I can give you a quick tour of what VoxSentinel AI does.',
    suggestion: 'How does VoxSentinel AI work?',
  },
]

function contextForRoute(pathname: string): PageContext {
  const route = pathname.replace(/\/+$/, '') || '/'
  const found =
    PAGE_CONTEXTS.find((entry) => entry.match.includes(route)) ??
    PAGE_CONTEXTS.find((entry) => entry.match.some((m) => m !== '/' && route.startsWith(m))) ??
    PAGE_CONTEXTS[PAGE_CONTEXTS.length - 1]
  return { route, page: found.page, module: found.module, greeting: found.greeting, suggestion: found.suggestion }
}

interface AssistantContextValue {
  open: boolean
  openChat: () => void
  closeChat: () => void
  toggle: () => void
  minimise: () => void
  messages: AssistantMessage[]
  pending: boolean
  error: string | null
  dismissError: () => void
  send: (text: string) => Promise<void>
  reset: () => void
  context: PageContext
  status: AssistantStatus | null
  hasUnread: boolean
  markSeen: () => void
  maxChars: number
}

const AssistantContext = createContext<AssistantContextValue | null>(null)

let idCounter = 0
function nextId(): string {
  idCounter += 1
  return `m${Date.now().toString(36)}-${idCounter}`
}

function readSession(): { messages: AssistantMessage[]; open: boolean } {
  try {
    const raw = window.sessionStorage.getItem(SESSION_KEY)
    if (!raw) return { messages: [], open: false }
    const parsed = JSON.parse(raw) as { messages?: AssistantMessage[]; open?: boolean }
    const messages = Array.isArray(parsed.messages)
      ? parsed.messages.filter(
          (m): m is AssistantMessage =>
            Boolean(m) && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string',
        )
      : []
    return { messages, open: Boolean(parsed.open) }
  } catch {
    return { messages: [], open: false }
  }
}

export function AssistantProvider({ children }: { children: ReactNode }) {
  const location = useLocation()
  const context = useMemo(() => contextForRoute(location.pathname), [location.pathname])

  const initial = useRef<{ messages: AssistantMessage[]; open: boolean }>()
  if (!initial.current) initial.current = readSession()

  const [open, setOpen] = useState<boolean>(initial.current.open)
  const [messages, setMessages] = useState<AssistantMessage[]>(initial.current.messages)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<AssistantStatus | null>(null)
  const [hasUnread, setHasUnread] = useState(false)
  const pendingRef = useRef(false)

  // One status probe per session: tells the UI whether Gemini is wired up.
  useEffect(() => {
    let cancelled = false
    void fetchAssistantStatus().then((value) => {
      if (!cancelled) setStatus(value)
    })
    return () => {
      cancelled = true
    }
  }, [])

  // First-time visitors see a small notification dot on the mascot.
  useEffect(() => {
    try {
      if (!window.localStorage.getItem(SEEN_KEY)) setHasUnread(true)
    } catch {
      /* storage blocked: no dot */
    }
  }, [])

  useEffect(() => {
    try {
      window.sessionStorage.setItem(SESSION_KEY, JSON.stringify({ messages, open }))
    } catch {
      /* private mode: the transcript stays in memory */
    }
  }, [messages, open])

  const markSeen = useCallback(() => {
    setHasUnread(false)
    try {
      window.localStorage.setItem(SEEN_KEY, '1')
    } catch {
      /* ignore */
    }
  }, [])

  const openChat = useCallback(() => {
    setOpen(true)
    markSeen()
  }, [markSeen])

  const closeChat = useCallback(() => setOpen(false), [])
  const minimise = useCallback(() => setOpen(false), [])
  const toggle = useCallback(() => {
    setOpen((current) => {
      if (!current) markSeen()
      return !current
    })
  }, [markSeen])

  const dismissError = useCallback(() => setError(null), [])

  const reset = useCallback(() => {
    setMessages([])
    setError(null)
    try {
      window.sessionStorage.removeItem(SESSION_KEY)
    } catch {
      /* ignore */
    }
  }, [])

  const send = useCallback(
    async (text: string) => {
      const content = text.trim()
      if (!content || pendingRef.current) return
      if (content.length > ASSISTANT_MAX_MESSAGE_CHARS) {
        setError(`Please keep questions under ${ASSISTANT_MAX_MESSAGE_CHARS} characters.`)
        return
      }

      const userMessage: AssistantMessage = {
        id: nextId(),
        role: 'user',
        content,
        createdAt: Date.now(),
      }
      const history = [...messages, userMessage]
      setMessages(history)
      setError(null)
      setPending(true)
      pendingRef.current = true

      try {
        const reply = await sendAssistantMessage(
          history.map((m) => ({ role: m.role, content: m.content })),
          context.route,
        )
        setMessages((current) => [
          ...current,
          { id: nextId(), role: 'assistant', content: reply.reply, createdAt: Date.now() },
        ])
      } catch (caught) {
        const message =
          caught instanceof Error && caught.message
            ? caught.message
            : 'The assistant is unavailable right now. Please try again.'
        setError(message)
        // A failed turn is removed so the transcript stays a clean alternation
        // of questions and real answers; the question stays in the input box.
        setMessages((current) => current.filter((m) => m.id !== userMessage.id))
        window.dispatchEvent(
          new CustomEvent('voxsentinel:assistant-failed', { detail: { text: content } }),
        )
      } finally {
        setPending(false)
        pendingRef.current = false
      }
    },
    [context.route, messages],
  )

  const value = useMemo<AssistantContextValue>(
    () => ({
      open,
      openChat,
      closeChat,
      toggle,
      minimise,
      messages,
      pending,
      error,
      dismissError,
      send,
      reset,
      context,
      status,
      hasUnread,
      markSeen,
      maxChars: status?.maxMessageChars ?? ASSISTANT_MAX_MESSAGE_CHARS,
    }),
    [
      open,
      openChat,
      closeChat,
      toggle,
      minimise,
      messages,
      pending,
      error,
      dismissError,
      send,
      reset,
      context,
      status,
      hasUnread,
      markSeen,
    ],
  )

  return <AssistantContext.Provider value={value}>{children}</AssistantContext.Provider>
}

export function useAssistant(): AssistantContextValue {
  const ctx = useContext(AssistantContext)
  if (!ctx) throw new Error('useAssistant must be used inside AssistantProvider')
  return ctx
}
