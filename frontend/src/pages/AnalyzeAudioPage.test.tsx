// Page-level integration tests for the Analyze Audio workflow.
//
// jsdom has no Web Audio decoder and no real media playback, so decoding,
// the audio element and the backend are MOCKED here — for tests only. The
// "analysis results" below are canned mock payloads used to exercise the
// UI's response handling; they are never presented as real predictions.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { AnalysisRecord, HealthResponse } from '../lib/types'
import type { DecodedAudioLike } from '../lib/audio/validation'

/* ---------------- mocks ---------------- */

const decodeMock = vi.hoisted(() => vi.fn())
vi.mock('../lib/audio/decode', () => ({
  decodeAudioFile: decodeMock,
  decodeSupportHint: () => ({ level: 'yes', message: null }),
}))

const analyzeMock = vi.hoisted(() => vi.fn())
vi.mock('../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../lib/api')>('../lib/api')
  return { ...actual, analyzeAudio: analyzeMock }
})

const statusMock = vi.hoisted(() => ({
  value: {
    status: 'online' as 'online' | 'offline' | 'checking',
    health: null as HealthResponse | null,
    lastCheckedAt: null as Date | null,
    refresh: async () => {},
  },
}))
vi.mock('../lib/apiStatus', () => ({
  useApiStatus: () => statusMock.value,
}))

import AnalyzeAudioPage from './AnalyzeAudioPage'
import { ApiError, ApiUnavailableError } from '../lib/api'

/* ---------------- helpers ---------------- */

const HEALTH_READY: HealthResponse = {
  status: 'ok',
  backend_ready: true,
  model_ready: true,
  speaker_model_ready: false,
  version: '0.1.0',
  mode: 'baseline',
}

function goodBuffer(overrides: Partial<DecodedAudioLike> = {}): DecodedAudioLike {
  const data = new Float32Array(8000).fill(0.3)
  return {
    duration: 5,
    sampleRate: 16000,
    numberOfChannels: 1,
    getChannelData: () => data,
    ...overrides,
  }
}

const MOCK_RECORD: AnalysisRecord = {
  id: 'test-analysis-1',
  created_at: '2026-10-22T10:00:00Z',
  input: { display_name: 'clip.wav', duration_s: 5, sample_rate_hz: 16000, channels: 1, quality_warnings: [] },
  model: { name: 'test-model', version: '0.0.1', mode: 'baseline' },
  scores: { raw_score: 0.42, raw_score_definition: 'TEST DEFINITION', calibrated_probability: null, uncertainty: null, uncertainty_method: null },
  decision: { category: 'inconclusive', label: 'Inconclusive TEST', reasons: ['r1'], recommendations: ['rec1'] },
  latency_ms: 7,
}

function wavFile(name = 'clip.wav') {
  return new File([new Uint8Array([1, 2, 3, 4])], name, { type: 'audio/wav', lastModified: 1 })
}

function renderPage() {
  return render(
    <MemoryRouter>
      <AnalyzeAudioPage />
    </MemoryRouter>,
  )
}

function dropFile(file: File) {
  fireEvent.drop(screen.getByTestId('audio-dropzone'), { dataTransfer: { files: [file] } })
}

/* Fake audio element so the player hook can be exercised. */
class FakeAudio {
  static instances: FakeAudio[] = []
  paused = true
  currentTime = 0
  duration = 5
  src = ''
  preload = ''
  listeners = new Map<string, Set<() => void>>()
  removed = 0
  loadCalled = false
  constructor() {
    FakeAudio.instances.push(this)
  }
  addEventListener(t: string, fn: () => void) {
    if (!this.listeners.has(t)) this.listeners.set(t, new Set())
    this.listeners.get(t)!.add(fn)
  }
  removeEventListener(t: string, fn: () => void) {
    this.listeners.get(t)?.delete(fn)
    this.removed++
  }
  emit(t: string) {
    this.listeners.get(t)?.forEach((fn) => fn())
  }
  play() {
    this.paused = false
    this.emit('play')
    return Promise.resolve()
  }
  pause() {
    this.paused = true
    this.emit('pause')
  }
  removeAttribute() {
    this.src = ''
  }
  load() {
    this.loadCalled = true
  }
}

let createUrl: ReturnType<typeof vi.fn>
let revokeUrl: ReturnType<typeof vi.fn>

beforeEach(() => {
  decodeMock.mockReset()
  analyzeMock.mockReset()
  FakeAudio.instances = []
  vi.stubGlobal('Audio', FakeAudio)
  createUrl = vi.fn(() => `blob:test-${Math.random()}`)
  revokeUrl = vi.fn()
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: createUrl, revokeObjectURL: revokeUrl }))
  statusMock.value = {
    status: 'online',
    health: HEALTH_READY,
    lastCheckedAt: null,
    refresh: async () => {},
  }
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

/* ---------------- tests ---------------- */

describe('AnalyzeAudioPage: selection & validation', () => {
  it('starts empty with the dropzone and no analyze button enabled', () => {
    renderPage()
    expect(screen.getByTestId('audio-dropzone')).toBeTruthy()
    const start = screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement
    expect(start.disabled).toBe(true)
  })

  it('decodes a valid file and shows real metadata', async () => {
    decodeMock.mockResolvedValue(goodBuffer({ duration: 5, sampleRate: 44100, numberOfChannels: 2 }))
    renderPage()
    dropFile(wavFile())
    await screen.findByText(/decoded & validated/i)
    expect(screen.getByText('44,100 Hz')).toBeTruthy()
    expect(screen.getByText('5.0 s')).toBeTruthy()
    expect(screen.getByText('clip.wav')).toBeTruthy()
    expect(screen.getByRole('slider', { name: /audio timeline/i })).toBeTruthy()
  })

  it('rejects unsupported extensions without attempting to decode', async () => {
    renderPage()
    dropFile(new File([new Uint8Array([1])], 'notes.txt', { type: 'text/plain' }))
    await screen.findByText(/file not accepted/i)
    expect(decodeMock).not.toHaveBeenCalled()
  })

  it('rejects empty files', async () => {
    renderPage()
    dropFile(new File([], 'empty.wav', { type: 'audio/wav' }))
    await screen.findByText(/file not accepted/i)
    expect(screen.getByText(/is empty/i)).toBeTruthy()
    expect(decodeMock).not.toHaveBeenCalled()
  })

  it('rejects oversized files', async () => {
    renderPage()
    const big = new File([new Uint8Array([1])], 'big.wav', { type: 'audio/wav' })
    Object.defineProperty(big, 'size', { value: 30 * 1024 * 1024 })
    dropFile(big)
    await screen.findByText(/file not accepted/i)
    expect(screen.getByText(/larger than/i)).toBeTruthy()
    expect(decodeMock).not.toHaveBeenCalled()
  })

  it('shows a decode error for corrupted audio and disables analysis', async () => {
    decodeMock.mockRejectedValue(new Error('The browser could not decode this file as audio.'))
    renderPage()
    dropFile(wavFile('corrupt.wav'))
    await screen.findByText(/could not decode audio/i)
    expect((screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('flags silent audio as unable to analyze and blocks analysis', async () => {
    decodeMock.mockResolvedValue(
      goodBuffer({ getChannelData: () => new Float32Array(8000).fill(0) }),
    )
    renderPage()
    dropFile(wavFile('silent.wav'))
    await screen.findByText(/unable to analyze this recording/i)
    fireEvent.click(screen.getByRole('checkbox'))
    expect((screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('warns (but still allows analysis) for short audio', async () => {
    decodeMock.mockResolvedValue(goodBuffer({ duration: 1 }))
    renderPage()
    dropFile(wavFile('short.wav'))
    await screen.findByText(/very short/i)
    fireEvent.click(screen.getByRole('checkbox'))
    expect((screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement).disabled).toBe(false)
  })
})

describe('AnalyzeAudioPage: file lifecycle & cleanup', () => {
  it('removes the file, revokes its object URL and returns to the empty state', async () => {
    decodeMock.mockResolvedValue(goodBuffer())
    renderPage()
    dropFile(wavFile())
    await screen.findByText(/decoded & validated/i)
    const urlsBefore = createUrl.mock.results.map((r) => r.value)

    fireEvent.click(screen.getByRole('button', { name: /remove file/i }))
    expect(revokeUrl).toHaveBeenCalledWith(urlsBefore[0])
    expect(screen.getByTestId('audio-dropzone')).toBeTruthy()
  })

  it('revokes the object URL and releases the audio element on unmount', async () => {
    decodeMock.mockResolvedValue(goodBuffer())
    const { unmount } = renderPage()
    dropFile(wavFile())
    await screen.findByText(/decoded & validated/i)
    const url = createUrl.mock.results[0].value
    const audio = FakeAudio.instances[FakeAudio.instances.length - 1]

    unmount()
    expect(revokeUrl).toHaveBeenCalledWith(url)
    expect(audio.paused).toBe(true)
    expect(audio.loadCalled).toBe(true)
    expect(audio.listeners.get('timeupdate')?.size ?? 0).toBe(0)
  })

  it('revokes the previous URL when a new file replaces it', async () => {
    decodeMock.mockResolvedValue(goodBuffer())
    renderPage()
    dropFile(wavFile('first.wav'))
    await screen.findByText(/decoded & validated/i)
    const first = createUrl.mock.results[0].value

    // Remove then pick another (dropzone only shows when empty).
    fireEvent.click(screen.getByRole('button', { name: /remove file/i }))
    dropFile(wavFile('second.wav'))
    await screen.findByText('second.wav')
    expect(revokeUrl).toHaveBeenCalledWith(first)
  })
})

describe('AnalyzeAudioPage: playback controls', () => {
  it('toggles play and pause and shows the current time', async () => {
    decodeMock.mockResolvedValue(goodBuffer())
    renderPage()
    dropFile(wavFile())
    await screen.findByText(/decoded & validated/i)
    const audio = FakeAudio.instances[FakeAudio.instances.length - 1]

    fireEvent.click(screen.getByRole('button', { name: 'Play' }))
    await screen.findByRole('button', { name: 'Pause' })
    expect(audio.paused).toBe(false)

    audio.currentTime = 2
    audio.emit('timeupdate')
    await screen.findByText(/0:02 \/ 0:05/)

    fireEvent.click(screen.getByRole('button', { name: 'Pause' }))
    await screen.findByRole('button', { name: 'Play' })
    expect(audio.paused).toBe(true)
  })

  it('seeks the audio when the timeline is clicked', async () => {
    decodeMock.mockResolvedValue(goodBuffer())
    renderPage()
    dropFile(wavFile())
    await screen.findByText(/decoded & validated/i)
    const audio = FakeAudio.instances[FakeAudio.instances.length - 1]

    // jsdom rect is zero-width; component falls back to 300px: x=150 -> 0.5 -> 2.5 s
    fireEvent.click(screen.getByRole('slider', { name: /audio timeline/i }), { clientX: 150 })
    await waitFor(() => expect(audio.currentTime).toBeGreaterThan(2.3))
    expect(audio.currentTime).toBeLessThan(2.7)
  })
})

describe('AnalyzeAudioPage: backend integration states', () => {
  async function validated() {
    decodeMock.mockResolvedValue(goodBuffer())
    renderPage()
    dropFile(wavFile())
    await screen.findByText(/decoded & validated/i)
  }

  it('keeps preview working but disables analysis when the backend is offline', async () => {
    statusMock.value = { ...statusMock.value, status: 'offline', health: null }
    await validated()
    expect(screen.getByText(/backend is offline/i)).toBeTruthy()
    fireEvent.click(screen.getByRole('checkbox'))
    expect((screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement).disabled).toBe(true)
    expect(analyzeMock).not.toHaveBeenCalled()
  })

  it('disables analysis when health reports the model is not ready', async () => {
    statusMock.value = { ...statusMock.value, health: { ...HEALTH_READY, model_ready: false } }
    await validated()
    expect(screen.getByText(/detection model\s+as not ready/i)).toBeTruthy()
    fireEvent.click(screen.getByRole('checkbox'))
    expect((screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement).disabled).toBe(true)
    expect(analyzeMock).not.toHaveBeenCalled()
  })

  it('requires consent before analysis and never uploads automatically', async () => {
    await validated()
    const start = screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement
    expect(start.disabled).toBe(true)
    expect(analyzeMock).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('checkbox'))
    expect(start.disabled).toBe(false)
    expect(analyzeMock).not.toHaveBeenCalled() // still nothing sent until clicked
  })

  it('displays the backend response (mock payload) and blocks duplicate analysis', async () => {
    analyzeMock.mockResolvedValue(MOCK_RECORD)
    await validated()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: /start analysis/i }))

    await screen.findByText('Inconclusive TEST')
    // UI spec: score descriptions and the model-name pill are no longer shown,
    // and the raw score is formatted to five decimals.
    expect(screen.queryByText('TEST DEFINITION')).toBeNull()
    expect(screen.queryByText(/test-model v0\.0\.1/)).toBeNull()
    expect(screen.getByText('0.42000')).toBeTruthy()
    expect(analyzeMock).toHaveBeenCalledTimes(1)

    // Same file again: duplicate processing is prevented.
    expect((screen.getByRole('button', { name: /start analysis/i }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText(/already analyzed/i)).toBeTruthy()
  })

  it('shows a model-unavailable message for a 503 response', async () => {
    analyzeMock.mockRejectedValue(new ApiError('x', 503, 'model_unavailable'))
    await validated()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: /start analysis/i }))
    await screen.findByText(/detection model is not available/i)
    expect(screen.queryByText(/analysis result/i)).toBeNull()
  })

  it('shows the validation error for a 422 response and no result', async () => {
    analyzeMock.mockRejectedValue(new ApiError('x', 422, 'invalid_audio'))
    await validated()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: /start analysis/i }))
    await screen.findByText(/could not process this audio/i)
  })

  it('shows a connection failure when the backend cannot be reached mid-request', async () => {
    analyzeMock.mockRejectedValue(new ApiUnavailableError('Could not reach the backend.'))
    await validated()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: /start analysis/i }))
    await screen.findByText(/could not reach the backend/i)
    expect(screen.getByText(/analysis request failed/i)).toBeTruthy()
  })
})
