import type {
  AnalysisListResponse,
  AnalysisRecord,
  AuditListResponse,
  FamilyVoiceProfile,
  ProtectionAnalysis,
  ProtectionAnalysisSummary,
  CleanupRunResult,
  EvaluationJob,
  EvaluationMetrics,
  EvaluationSummary,
  HealthResponse,
  ModelInfo,
  RetentionPolicy,
  SpeakerVerifyResponse,
} from './types'

const DEFAULT_BASE_URL = 'http://localhost:8000'
const BASE_URL_STORAGE_KEY = 'voxsentinel.apiBaseUrl'
export const REQUEST_TIMEOUT_MS = 6000

/** Uploads can legitimately take longer; analysis gets its own timeout. */
export const ANALYZE_TIMEOUT_MS = 120000

/**
 * Family Emergency & Scam Protection runs transcription + Gemini contextual
 * analysis + detector inference in one request, so it needs a wider budget
 * than the core analysis workflow (which stays at 120s).
 */
export const PROTECTION_TIMEOUT_MS = 240000

/** Raised when the backend cannot be reached at all (network, CORS, down). */
export class ApiUnavailableError extends Error {
  constructor(message = 'Cannot reach the VoxSentinel backend') {
    super(message)
    this.name = 'ApiUnavailableError'
  }
}

/** Raised when the backend answers with a non-2xx status. */
export class ApiError extends Error {
  status: number
  code: string | null

  constructor(message: string, status: number, code: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

/**
 * Resolves the backend base URL. Priority: manual override stored by the
 * Settings page, then the VITE_API_BASE_URL build-time variable, then the
 * local development default. Trailing slashes are stripped.
 */
export function getConfiguredBaseUrl(): string {
  let stored: string | null = null
  try {
    stored = window.localStorage.getItem(BASE_URL_STORAGE_KEY)
  } catch {
    stored = null
  }
  const fromEnv = (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env
    ?.VITE_API_BASE_URL
  return (stored || fromEnv || DEFAULT_BASE_URL).replace(/\/+$/, '')
}

export function setConfiguredBaseUrl(url: string | null): void {
  if (url === null || url.trim() === '') {
    window.localStorage.removeItem(BASE_URL_STORAGE_KEY)
  } else {
    window.localStorage.setItem(BASE_URL_STORAGE_KEY, url.trim().replace(/\/+$/, ''))
  }
}

export function getBaseUrlOrigin(): string {
  try {
    return new URL(getConfiguredBaseUrl()).origin
  } catch {
    return getConfiguredBaseUrl()
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    let response: Response
    try {
      response = await fetch(`${getConfiguredBaseUrl()}${path}`, {
        ...init,
        signal: controller.signal,
      })
    } catch {
      // Fetch rejects on network failure, CORS block, or our abort timeout.
      throw new ApiUnavailableError()
    }
    if (!response.ok) {
      let code: string | null = null
      try {
        const body = (await response.json()) as { error?: { code?: string } }
        code = body?.error?.code ?? null
      } catch {
        // Non-JSON error body; keep the safe status text only.
      }
      throw new ApiError(`Backend returned ${response.status} ${response.statusText}`, response.status, code)
    }
    return (await response.json()) as T
  } finally {
    window.clearTimeout(timeout)
  }
}

export const api = {
  getHealth: () => request<HealthResponse>('/health'),
  listModels: () => request<{ models: ModelInfo[] }>('/api/v1/models'),
  listAnalyses: (params: { category?: string; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams()
    if (params.category) q.set('category', params.category)
    q.set('limit', String(params.limit ?? 50))
    q.set('offset', String(params.offset ?? 0))
    return request<AnalysisListResponse>(`/api/v1/analyses?${q.toString()}`)
  },
  getAnalysis: (id: string) =>
    request<AnalysisRecord>(`/api/v1/analyses/${encodeURIComponent(id)}`),
  getEvaluation: () => request<EvaluationMetrics>('/api/v1/evaluation/metrics'),
  getEvaluationSummary: () => request<EvaluationSummary>('/api/v1/evaluation'),
  startEvaluationRun: () => request<EvaluationJob>('/api/v1/evaluation/run', { method: 'POST' }),
  getEvaluationRun: (jobId: string) =>
    request<EvaluationJob>(`/api/v1/evaluation/run/${encodeURIComponent(jobId)}`),
  listAuditEvents: (params: { event_type?: string; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams()
    if (params.event_type) q.set('event_type', params.event_type)
    q.set('limit', String(params.limit ?? 50))
    q.set('offset', String(params.offset ?? 0))
    return request<AuditListResponse>(`/api/v1/audit?${q.toString()}`)
  },
  getRetentionPolicy: () => request<RetentionPolicy>('/api/v1/admin/retention'),
  runRetentionCleanup: (dryRun: boolean) =>
    request<CleanupRunResult>(`/api/v1/admin/retention/run?dry_run=${dryRun ? 'true' : 'false'}`, { method: 'POST' }),
}

/* ------------------------------------------------------------------ */
/* POST /api/v1/speaker-verify (Module 3)                             */
/* ------------------------------------------------------------------ */

export async function verifySpeakers(
  reference: File,
  test: File,
  opts: { consent: boolean; referenceAuthorized: boolean },
): Promise<SpeakerVerifyResponse> {
  const form = new FormData()
  form.append('reference', reference, reference.name)
  form.append('test', test, test.name)
  form.append('consent', String(opts.consent))
  form.append('reference_authorized', String(opts.referenceAuthorized))
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS)
  try {
    let response: Response
    try {
      response = await fetch(`${getConfiguredBaseUrl()}/api/v1/speaker-verify`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
    } catch {
      throw new ApiUnavailableError('Could not reach the backend for the comparison.')
    }
    if (!response.ok) {
      let body: { error?: { code?: string; message?: string } } = {}
      try {
        body = (await response.json()) as { error?: { code?: string } }
      } catch {
        /* non-JSON body */
      }
      const message = body?.error?.message
      throw new ApiError(message ?? `Backend returned ${response.status}`, response.status, body?.error?.code ?? null)
    }
    return (await response.json()) as SpeakerVerifyResponse
  } finally {
    window.clearTimeout(timeout)
  }
}

/* ------------------------------------------------------------------ */
/* GET /api/v1/analyses/export.csv (Module 4)                         */
/* ------------------------------------------------------------------ */

/** Downloads the authorized history as a CSV Blob (DB-backed, escaped). */
export async function exportAnalysesCsv(): Promise<Blob> {
  let response: Response
  try {
    response = await fetch(`${getConfiguredBaseUrl()}/api/v1/analyses/export.csv`)
  } catch {
    throw new ApiUnavailableError('Could not reach the backend for the export.')
  }
  if (!response.ok) {
    throw new ApiError(`Backend returned ${response.status} ${response.statusText}`, response.status, null)
  }
  return response.blob()
}

/* ------------------------------------------------------------------ */
/* POST /api/v1/analyze                                                */
/* ------------------------------------------------------------------ */

export interface AnalyzeProgress {
  /** Bytes uploaded so far. */
  loaded: number
  /** Total bytes to upload. */
  total: number
}

export interface AnalyzeOptions {
  onProgress?: (progress: AnalyzeProgress) => void
  timeoutMs?: number
  /** false = compute but do NOT store (provisional live-recording windows). */
  persist?: boolean
}

/**
 * Uploads the audio for analysis as multipart/form-data per the API contract:
 * fields `file` and `consent=true`. Uses XMLHttpRequest because fetch cannot
 * report upload progress.
 *
 * The audio is only sent when the user explicitly calls this — selection and
 * preview alone never transmit anything.
 */
export function analyzeAudio(file: File, options: AnalyzeOptions = {}): Promise<AnalysisRecord> {
  const timeoutMs = options.timeoutMs ?? ANALYZE_TIMEOUT_MS

  return new Promise<AnalysisRecord>((resolve, reject) => {
    const form = new FormData()
    form.append('file', file, file.name)
    form.append('consent', 'true')
    if (options.persist === false) form.append('persist', 'false')

    const xhr = new XMLHttpRequest()
    let settled = false
    const timeout = window.setTimeout(() => {
      if (settled) return
      settled = true
      xhr.abort()
      reject(new ApiError('The analysis request timed out.', 0, 'timeout'))
    }, timeoutMs)

    const settle = (fn: () => void) => {
      if (settled) return
      settled = true
      window.clearTimeout(timeout)
      fn()
    }

    xhr.open('POST', `${getConfiguredBaseUrl()}/api/v1/analyze`)
    xhr.responseType = 'json'

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && options.onProgress) {
        options.onProgress({ loaded: event.loaded, total: event.total })
      }
    }
    xhr.onerror = () =>
      settle(() =>
        reject(
          new ApiUnavailableError(
            'Could not reach the backend (network error or CORS). The file was not analyzed.',
          ),
        ),
      )
    xhr.onabort = () =>
      settle(() => reject(new ApiError('The request was aborted.', 0, 'aborted')))
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        settle(() => resolve(xhr.response as AnalysisRecord))
      } else {
        const body = (xhr.response ?? {}) as { error?: { code?: string } }
        settle(() =>
          reject(
            new ApiError(
              `Backend returned ${xhr.status} ${xhr.statusText}`,
              xhr.status,
              body?.error?.code ?? null,
            ),
          ),
        )
      }
    }

    xhr.send(form)
  })
}

/**
 * Maps an analysis failure to a safe, friendly message. Exported for tests.
 * 4xx codes mean the request/file was the problem; 5xx means the backend.
 */
export function mapAnalyzeError(status: number, code: string | null): string {
  if (status === 413 || code === 'file_too_large')
    return 'The backend rejected the file as too large.'
  if (status === 415 || code === 'unsupported_media_type')
    return 'The backend does not accept this audio format.'
  if (status === 422 || code === 'invalid_audio')
    return 'The backend could not process this audio (for example: silent, corrupted or too short).'
  if (status === 429) return 'Too many analysis requests right now — please wait a moment and retry.'
  if (status === 503 || code === 'model_unavailable')
    return 'The detection model is not available on the backend right now.'
  if (status === 400) return 'The backend rejected the request as malformed.'
  if (status === 0) return 'The request timed out or was aborted.'
  return 'The analysis request failed with an unexpected backend error.'
}

/* ---------- Family Emergency & Scam Protection (additive module) ---------- */

export async function analyzeProtection(
  file: File,
  opts: { consent: boolean; profileId?: string | null },
): Promise<ProtectionAnalysis> {
  const form = new FormData()
  form.append('file', file, file.name)
  form.append('consent', String(opts.consent))
  if (opts.profileId) form.append('profile_id', opts.profileId)
  return uploadJson<ProtectionAnalysis>('/api/v1/protection/analyze', form, PROTECTION_TIMEOUT_MS)
}

async function uploadJson<T>(
  path: string,
  form: FormData,
  timeoutMs: number = ANALYZE_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    let response: Response
    try {
      response = await fetch(`${getConfiguredBaseUrl()}${path}`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
    } catch {
      throw new ApiUnavailableError('Could not reach the backend for the protection analysis.')
    }
    if (!response.ok) {
      let code: string | null = null
      let message = `Backend returned ${response.status}`
      try {
        const body = (await response.json()) as { error?: { code?: string; message?: string } }
        code = body?.error?.code ?? null
        message = body?.error?.message ?? message
      } catch {
        // keep the safe default message
      }
      throw new ApiError(message, response.status, code)
    }
    return (await response.json()) as T
  } finally {
    window.clearTimeout(timeout)
  }
}

export const protectionApi = {
  analyze: analyzeProtection,
  listAnalyses: (limit = 50) =>
    request<{ analyses: ProtectionAnalysisSummary[]; count: number }>(`/api/v1/protection/analyses?limit=${limit}`),
  getAnalysis: (id: string) =>
    request<ProtectionAnalysis>(`/api/v1/protection/analyses/${encodeURIComponent(id)}`),
  deleteAnalysis: (id: string) =>
    request<{ deleted: string }>(`/api/v1/protection/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  reportUrl: (id: string) => `${getConfiguredBaseUrl()}/api/v1/protection/analyses/${encodeURIComponent(id)}/report`,
  listProfiles: () => request<{ profiles: FamilyVoiceProfile[]; count: number }>('/api/v1/protection/family-voices'),
  createProfile: async (name: string, relationship: string, reference: File): Promise<FamilyVoiceProfile> => {
    const form = new FormData()
    form.append('name', name)
    form.append('relationship', relationship)
    form.append('consent', 'true')
    form.append('reference', reference, reference.name)
    return uploadJson<FamilyVoiceProfile>('/api/v1/protection/family-voices', form)
  },
  deleteProfile: (id: string) =>
    request<{ deleted: string }>(`/api/v1/protection/family-voices/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  capabilities: () =>
    request<{ transcription: Record<string, unknown>; speaker_verifier_ready: boolean; detector_ready: boolean }>(
      '/api/v1/protection/capabilities',
    ),
}
