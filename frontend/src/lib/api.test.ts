import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  analyzeAudio,
  analyzeProtectionText,
  api,
  ApiError,
  ApiUnavailableError,
  mapAnalyzeError,
  protectionApi,
} from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

const HEALTH = {
  status: 'ok',
  backend_ready: true,
  model_ready: false,
  speaker_model_ready: false,
  version: '0.1.0',
  mode: 'baseline',
}

function jsonRes(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status })
}

describe('api.getHealth', () => {
  it('parses a healthy response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonRes(HEALTH)))
    const h = await api.getHealth()
    expect(h.status).toBe('ok')
    expect(h.model_ready).toBe(false)
  })

  it('maps network failures to ApiUnavailableError', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch') }))
    await expect(api.getHealth()).rejects.toBeInstanceOf(ApiUnavailableError)
  })

  it('maps error statuses to ApiError with the backend error code', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonRes({ error: { code: 'file_too_large', message: 'no' } }, 413)),
    )
    const err = await api.getHealth().catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(413)
    expect((err as ApiError).code).toBe('file_too_large')
  })
})

describe('mapAnalyzeError', () => {
  it.each([
    [413, null, /too large/],
    [415, null, /format/],
    [422, null, /could not process/],
    [429, null, /too many/i],
    [503, null, /model is not available/],
    [400, null, /malformed/],
    [0, 'timeout', /timed out/],
    [500, null, /unexpected/],
  ])('maps %s (%s) to a friendly message', (status, code, re) => {
    expect(mapAnalyzeError(status, code)).toMatch(re)
  })
})

/* --------------------------------------------------------------- */
/* POST /api/v1/analyze via XMLHttpRequest                         */
/* --------------------------------------------------------------- */

class FakeXHR {
  static last: FakeXHR | null = null

  responseType = ''
  status = 0
  statusText = ''
  response: unknown = null
  sent: FormData | null = null
  upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = {
    onprogress: null,
  }
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  onabort: (() => void) | null = null

  open() {}
  send(form: FormData) {
    this.sent = form
    FakeXHR.last = this
  }
  abort() {
    this.onabort?.()
  }
  // Test helpers.
  succeed(status: number, body: unknown) {
    this.status = status
    this.response = body
    this.onload?.()
  }
  fail() {
    this.onerror?.()
  }
}

describe('analyzeAudio', () => {
  it('sends multipart form data with consent and resolves with the record', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR)
    FakeXHR.last = null

    const file = new File([new Uint8Array([1, 2, 3])], 'clip.wav', { type: 'audio/wav' })
    const record = {
      id: 'a1',
      created_at: '2026-10-22T10:00:00Z',
      input: { display_name: 'clip.wav', duration_s: 3, sample_rate_hz: 16000, channels: 1, quality_warnings: [] },
      model: { name: 'baseline', version: '0.1', mode: 'baseline' },
      scores: { raw_score: 0.4, raw_score_definition: 'x', calibrated_probability: null, uncertainty: null, uncertainty_method: null },
      decision: { category: 'inconclusive', label: 'Inconclusive', reasons: [], recommendations: [] },
      latency_ms: 12,
    }

    const promise = analyzeAudio(file)
    await vi.waitFor(() => expect(FakeXHR.last).not.toBeNull())
    const xhr = FakeXHR.last!

    expect((xhr.sent as FormData).get('file')).toBeInstanceOf(File)
    expect((xhr.sent as FormData).get('consent')).toBe('true')

    xhr.succeed(200, record)
    const result = await promise
    expect(result.id).toBe('a1')
    expect(result.decision.category).toBe('inconclusive')
  })

  it('reports upload progress when measurable', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR)
    FakeXHR.last = null

    const onProgress = vi.fn()
    const file = new File([new Uint8Array(10)], 'clip.wav', { type: 'audio/wav' })
    const promise = analyzeAudio(file, { onProgress })
    await vi.waitFor(() => expect(FakeXHR.last).not.toBeNull())

    FakeXHR.last!.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 })
    expect(onProgress).toHaveBeenCalledWith({ loaded: 5, total: 10 })

    FakeXHR.last!.succeed(200, {
      id: 'a2',
      created_at: '2026-10-22T10:00:00Z',
      input: { display_name: 'clip.wav', duration_s: 1, sample_rate_hz: 16000, channels: 1, quality_warnings: [] },
      model: { name: 'baseline', version: '0.1', mode: 'baseline' },
      scores: { raw_score: 0, raw_score_definition: '', calibrated_probability: null, uncertainty: null, uncertainty_method: null },
      decision: { category: 'low_observed_risk', label: 'low', reasons: [], recommendations: [] },
      latency_ms: 5,
    })
    await promise
  })

  it('rejects with ApiError carrying the backend error code on 503', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR)
    FakeXHR.last = null

    const file = new File([new Uint8Array(10)], 'clip.wav', { type: 'audio/wav' })
    const promise = analyzeAudio(file)
    await vi.waitFor(() => expect(FakeXHR.last).not.toBeNull())
    FakeXHR.last!.succeed(503, { error: { code: 'model_unavailable' } })

    const err = await promise.catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(503)
    expect((err as ApiError).code).toBe('model_unavailable')
    expect(mapAnalyzeError((err as ApiError).status, (err as ApiError).code)).toMatch(/not available/)
  })

  it('rejects with ApiUnavailableError after every candidate base fails', async () => {
    // The client now retries each candidate base (configured + same origin)
    // before declaring the backend unreachable, so every attempt must fail.
    vi.stubGlobal('XMLHttpRequest', FakeXHR)
    FakeXHR.last = null

    const file = new File([new Uint8Array(10)], 'clip.wav', { type: 'audio/wav' })
    const promise = analyzeAudio(file)
    const err = await (async () => {
      while (true) {
        await vi.waitFor(() => expect(FakeXHR.last).not.toBeNull())
        FakeXHR.last!.fail()
        FakeXHR.last = null
        const settled = await Promise.race([
          promise.then(() => 'resolved').catch((e) => e),
          new Promise((r) => setTimeout(() => r('pending'), 25)),
        ])
        if (settled !== 'pending') return settled
      }
    })()
    expect(err).toBeInstanceOf(ApiUnavailableError)
  })
})

describe('unified protection endpoints', () => {
  it('posts pasted text to /api/analyze as JSON with consent', async () => {
    const calls: { url: string; init?: RequestInit }[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({ url: String(url), init })
        return jsonRes({ id: 'a1', risk_badges: [], explanation: { verdict: 'HIGH' } })
      }),
    )
    const result = await analyzeProtectionText('send me 1 lakh rupees now', { consent: true })
    expect(result.id).toBe('a1')
    expect(calls[0].url).toContain('/api/analyze')
    expect(calls[0].init?.method).toBe('POST')
    expect(JSON.parse(String(calls[0].init?.body))).toMatchObject({
      text: 'send me 1 lakh rupees now',
      consent: true,
    })
  })

  it('reads the live model families from /api/capabilities', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonRes({
          schema_version: '1.0',
          endpoint: 'POST /api/analyze',
          input_kinds: ['audio', 'text'],
          detector_ready: true,
          models: { speaker_verification: { ready: true, model: 'ecapa', embedding_dim: 192 } },
        }),
      ),
    )
    const caps = await protectionApi.pipelineCapabilities()
    expect(caps.models.speaker_verification?.embedding_dim).toBe(192)
    expect(caps.input_kinds).toEqual(['audio', 'text'])
  })
})
