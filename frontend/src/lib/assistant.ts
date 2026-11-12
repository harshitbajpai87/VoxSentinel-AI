import { ApiError, ApiUnavailableError, getConfiguredBaseUrl } from './api'

/**
 * Client for the server-side VoxSentinel Assistant.
 *
 * The Gemini key never reaches the browser: this module only posts the message
 * transcript and the current route, and the backend adds the system prompt and
 * calls Gemini. An assistant answer always comes from the model; when the call
 * fails we surface the failure and never invent a reply.
 */

/**
 * Candidates in priority order: the same configured base the rest of the app
 * uses (Settings override, then VITE_API_BASE_URL, then the page origin), with
 * the page's own origin kept as a fallback.
 */
function candidates(): string[] {
  const configured = getConfiguredBaseUrl()
  const sameOrigin = typeof window !== 'undefined' ? window.location.origin : ''
  return [...new Set([configured, sameOrigin].filter(Boolean))]
}

export type AssistantRole = 'user' | 'assistant'

export interface AssistantMessage {
  id: string
  role: AssistantRole
  content: string
  createdAt: number
}

export interface AssistantStatus {
  available: boolean
  provider: string | null
  providerLabel: string | null
  model: string | null
  maxMessages: number
  maxMessageChars: number
}

export interface AssistantReply {
  reply: string
  model: string | null
  latencyMs: number | null
  page: string | null
}

const CHAT_TIMEOUT_MS = 45000
const STATUS_TIMEOUT_MS = 12000

export const ASSISTANT_MAX_MESSAGE_CHARS = 2000

async function readError(response: Response): Promise<{ code: string | null; message: string | null }> {
  try {
    const body = (await response.json()) as { error?: { code?: string; message?: string } }
    return { code: body?.error?.code ?? null, message: body?.error?.message ?? null }
  } catch {
    return { code: null, message: null }
  }
}

/** Best-effort status probe: a failure means "not available", never a crash. */
export async function fetchAssistantStatus(): Promise<AssistantStatus | null> {
  for (const base of candidates()) {
    const controller = new AbortController()
    const timer = window.setTimeout(() => controller.abort(), STATUS_TIMEOUT_MS)
    try {
      const response = await fetch(`${base}/api/v1/assistant/status`, { signal: controller.signal })
      if (!response.ok) continue
      const body = (await response.json()) as {
        available?: boolean
        provider?: string | null
        provider_label?: string | null
        model?: string | null
        limits?: { max_messages?: number; max_message_chars?: number }
      }
      return {
        available: Boolean(body.available),
        provider: body.provider ?? null,
        providerLabel: body.provider_label ?? null,
        model: body.model ?? null,
        maxMessages: body.limits?.max_messages ?? 12,
        maxMessageChars: Math.min(body.limits?.max_message_chars ?? 4000, ASSISTANT_MAX_MESSAGE_CHARS),
      }
    } catch {
      /* try the next candidate */
    } finally {
      window.clearTimeout(timer)
    }
  }
  return null
}

/**
 * One conversational turn. Throws `ApiError` with the server's friendly message
 * (or a timeout message) so the UI can show it verbatim.
 */
export async function sendAssistantMessage(
  messages: Array<{ role: AssistantRole; content: string }>,
  route: string,
): Promise<AssistantReply> {
  const payload = JSON.stringify({
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    route,
  })

  let lastError: unknown = null
  for (const base of candidates()) {
    const controller = new AbortController()
    const timer = window.setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS)
    try {
      const response = await fetch(`${base}/api/v1/assistant/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        signal: controller.signal,
      })
      if (!response.ok) {
        const { code, message } = await readError(response)
        lastError = new ApiError(
          message ?? `The assistant could not answer (HTTP ${response.status}).`,
          response.status,
          code,
        )
        continue
      }
      const body = (await response.json()) as {
        reply?: string
        model?: string | null
        latency_ms?: number | null
        page?: string | null
      }
      const reply = (body.reply ?? '').trim()
      if (!reply) {
        lastError = new ApiError('The assistant returned an empty answer. Please try again.', 0, 'empty_reply')
        continue
      }
      return { reply, model: body.model ?? null, latencyMs: body.latency_ms ?? null, page: body.page ?? null }
    } catch (error) {
      if (error instanceof DOMException) {
        lastError = new ApiError('The assistant took too long to answer. Please try again.', 0, 'timeout')
      } else {
        lastError = new ApiUnavailableError('Cannot reach the VoxSentinel backend')
      }
    } finally {
      window.clearTimeout(timer)
    }
  }

  if (lastError instanceof Error) throw lastError
  throw new ApiUnavailableError('Cannot reach the VoxSentinel backend')
}
