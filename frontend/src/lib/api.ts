import { extensionOf, isKnownAudioExtension, toModelWav } from './audio/transcode'
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
import type { PipelineCapabilities } from './unified'

/**
 * Local development talks to uvicorn on :8000. Anywhere else (a static or
 * serverless deployment on the same domain as the API) the sane default is the
 * page's own origin, so a deployed build reaches /api without configuration.
 */
const DEFAULT_BASE_URL = ((): string => {
  const host = typeof window !== 'undefined' ? window.location?.hostname ?? '' : ''
  const isLocal = host === '' || host === 'localhost' || host === '127.0.0.1' || host === '[::1]'
  return isLocal ? 'http://localhost:8000' : window.location.origin
})()
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

/**
 * Bases worth trying, in order. The configured base wins, but the page's own
 * origin is always a fallback: a stale override saved in Settings (for example
 * a laptop's http://localhost:8000) otherwise makes the whole app look offline
 * even though the deployed API is healthy. Without this, the UI shows
 * "could not reach the backend" while the same request works from curl.
 */
function baseCandidates(): string[] {
  const configured = getConfiguredBaseUrl()
  const sameOrigin = typeof window !== 'undefined' ? window.location.origin : ''
  const list = [preferredBase ?? configured, configured]
  if (sameOrigin) list.push(sameOrigin)
  return [...new Set(list.filter(Boolean))]
}

let preferredBase: string | null = null

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    let response: Response | null = null
    let lastError: unknown = null
    for (const base of baseCandidates()) {
      try {
        response = await fetch(`${base}${path}`, { ...init, signal: controller.signal })
        preferredBase = base
        break
      } catch (err) {
        // Fetch rejects on network failure, CORS block, or our abort timeout.
        lastError = err
      }
    }
    if (!response) {
      throw lastError instanceof DOMException
        ? lastError
        : new ApiUnavailableError()
    }
    if (!response.ok) {
      let code: string | null = null
      let message = `Backend returned ${response.status} ${response.statusText}`
      try {
        const body = (await response.json()) as { error?: { code?: string; message?: string } }
        code = body?.error?.code ?? null
        if (body?.error?.message) message = body.error.message
      } catch {
        // Non-JSON error body (a proxy/platform page); keep the status text.
      }
      throw new ApiError(message, response.status, code)
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
  const policy = await getUploadPolicy()
  // Each recording is prepared and size-checked independently, so the error
  // names the offending file instead of "the request is too large".
  const [refFile, testFile] = await Promise.all([
    prepareUpload(reference, policy).catch((err: unknown) => {
      if (err instanceof ApiError) throw new ApiError(`Reference recording: ${err.message}`, err.status, err.code)
      throw err
    }),
    prepareUpload(test, policy).catch((err: unknown) => {
      if (err instanceof ApiError) throw new ApiError(`Comparison recording: ${err.message}`, err.status, err.code)
      throw err
    }),
  ])
  const form = new FormData()
  form.append('reference', refFile, refFile.name)
  form.append('test', testFile, testFile.name)
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
/* Upload preparation (shared by every audio workflow)                 */
/* ------------------------------------------------------------------ */

/** What this deployment can decode, and how large an upload it accepts. */
export interface UploadPolicy {
  /** Effective ceiling in bytes (already min(configured, hosting limit)). */
  maxBytes: number
  /** Extensions the SERVER decodes itself (ffmpeg-aware). */
  decodableExtensions: string[]
  ffmpeg: boolean
}

const FALLBACK_DECODABLE = ['wav', 'mp3', 'flac', 'ogg', 'oga']
let cachedPolicy: UploadPolicy | null = null

/** Cached capability lookup; falls back to libsndfile-only coverage on failure. */
export async function getUploadPolicy(force = false): Promise<UploadPolicy> {
  if (cachedPolicy && !force) return cachedPolicy
  try {
    const health = await request<{
      effective_max_upload_bytes?: number
      max_upload_bytes?: number
      audio?: { decodable_extensions?: string[]; decoders?: { ffmpeg?: boolean } }
    }>('/health')
    const extensions = health?.audio?.decodable_extensions
    cachedPolicy = {
      maxBytes:
        health?.effective_max_upload_bytes ||
        health?.max_upload_bytes ||
        DEFAULT_MAX_UPLOAD_BYTES,
      decodableExtensions:
        Array.isArray(extensions) && extensions.length ? extensions : FALLBACK_DECODABLE,
      ffmpeg: Boolean(health?.audio?.decoders?.ffmpeg),
    }
  } catch {
    cachedPolicy = {
      maxBytes: DEFAULT_MAX_UPLOAD_BYTES,
      decodableExtensions: FALLBACK_DECODABLE,
      ffmpeg: false,
    }
  }
  return cachedPolicy
}

/** The one size constant the UI talks about (mirrors the backend default). */
export const DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

function describeBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024)
  return `${mb >= 10 ? mb.toFixed(0) : mb.toFixed(1)} MB`
}

/**
 * Decide what actually goes over the wire for one recording.
 *
 * * the server decodes this container -> send the ORIGINAL bytes untouched
 *   (smallest body, no re-encoding, no size explosion)
 * * the server cannot decode it (no ffmpeg) -> convert to 16 kHz mono WAV in
 *   the browser and re-check the size BEFORE uploading
 *
 * A converted file that no longer fits the deployment limit fails here with a
 * message that names both sizes, instead of a generic "too large" from the API.
 */
export async function prepareUpload(file: File, policy?: UploadPolicy): Promise<File> {
  const effective = policy ?? (await getUploadPolicy())
  const ext = extensionOf(file.name)
  const serverDecodes = effective.decodableExtensions.includes(ext)

  if (serverDecodes) {
    if (file.size > effective.maxBytes) {
      throw new ApiError(
        `\u201c${file.name}\u201d is ${describeBytes(file.size)}, above this deployment's ${describeBytes(
          effective.maxBytes,
        )} upload limit.`,
        413,
        'file_too_large',
      )
    }
    return file
  }

  if (!isKnownAudioExtension(file.name)) return file // let the server reject it with its own message

  const converted = await toModelWav(file)
  if (converted.size > effective.maxBytes) {
    throw new ApiError(
      `\u201c${file.name}\u201d is ${describeBytes(file.size)} in ${ext.toUpperCase()} and converts to ` +
        `${describeBytes(converted.size)} of WAV, above this deployment's ${describeBytes(
          effective.maxBytes,
        )} upload limit. Deploy the container backend (which decodes ${ext.toUpperCase()} directly) ` +
        'or upload a shorter clip.',
      413,
      'file_too_large',
    )
  }
  return converted
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
export async function analyzeAudio(
  file: File,
  options: AnalyzeOptions = {},
): Promise<AnalysisRecord> {
  const bases = baseCandidates()
  const attempt = (index: number, upload: File): Promise<AnalysisRecord> =>
    analyzeWithBase(bases[index], upload, options).catch((err: unknown) => {
      // A transport failure (not an HTTP error) may just mean this base is
      // wrong: try the next candidate before telling the user it is offline.
      if (err instanceof ApiUnavailableError && index + 1 < bases.length) {
        return attempt(index + 1, upload)
      }
      throw err
    })

  // Send the original when the server can decode it; convert only when it
  // cannot. Size is re-validated after any conversion, so a big compressed clip
  // can never silently become an oversized WAV upload.
  const upload = await prepareUpload(file)
  try {
    return await attempt(0, upload)
  } catch (err) {
    // The deployment lost/gained ffmpeg between the cached capability probe and
    // this request (or the probe failed): convert once and retry, instead of
    // showing "this format is not supported" for a file we can convert.
    if (err instanceof ApiError && err.code === 'decoder_unavailable' && upload === file) {
      const converted = await toModelWav(file)
      return attempt(0, converted)
    }
    throw err
  }
}

function analyzeWithBase(
  base: string,
  file: File,
  options: AnalyzeOptions = {},
): Promise<AnalysisRecord> {
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

    xhr.open('POST', `${base}/api/v1/analyze`)
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
        preferredBase = base
        settle(() => resolve(xhr.response as AnalysisRecord))
      } else {
        const body = (xhr.response ?? {}) as {
          error?: { code?: string; message?: string }
        }
        settle(() =>
          reject(
            new ApiError(
              body?.error?.message ?? `Backend returned ${xhr.status} ${xhr.statusText}`,
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
/** True when a server message is specific enough to show instead of a generic one. */
function isSpecificMessage(message: string | null | undefined): boolean {
  if (!message) return false
  return !/^Backend returned \d+/i.test(message) && message.trim().length > 8
}

export function mapAnalyzeError(
  status: number,
  code: string | null,
  serverMessage?: string | null,
): string {
  // A specific explanation from the API (real limit, real format, real reason)
  // always beats a client-side generic phrase.
  if (isSpecificMessage(serverMessage)) return serverMessage as string
  if (status === 413 || code === 'file_too_large')
    return 'The backend rejected the file as too large.'
  if (code === 'decoder_unavailable')
    return 'This server cannot decode that audio format (its ffmpeg decoder is missing). Try WAV, MP3, FLAC or OGG.'
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
  opts: { consent: boolean; profileId?: string | null; language?: string | null },
): Promise<ProtectionAnalysis> {
  // Same preparation as Analyze Audio: the original is uploaded when the server
  // can decode it, otherwise it is converted and re-size-checked first.
  const uploadable = await prepareUpload(file)
  const form = new FormData()
  form.append('file', uploadable, uploadable.name)
  form.append('consent', String(opts.consent))
  if (opts.profileId) form.append('profile_id', opts.profileId)
  if (opts.language) form.append('language', opts.language)
  // Unified pipeline: transcription, speaker verification, acoustic forensics,
  // scam risk, LLM reasoning, badges and the structured explanation in one call.
  return uploadJson<ProtectionAnalysis>('/api/analyze', form, PROTECTION_TIMEOUT_MS)
}

/** Analyze pasted text with no audio (same content, LLM and risk layers). */
export async function analyzeProtectionText(
  text: string,
  opts: { consent: boolean; language?: string | null; profileId?: string | null },
): Promise<ProtectionAnalysis> {
  return request<ProtectionAnalysis>('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      consent: opts.consent,
      language: opts.language ?? undefined,
      profile_id: opts.profileId ?? undefined,
    }),
  })
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
  analyzeText: analyzeProtectionText,
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
  /** Which model families are actually live on the backend right now. */
  pipelineCapabilities: () => request<PipelineCapabilities>('/api/capabilities'),
}
