import { describe, expect, it, vi, afterEach } from 'vitest'
import { detectRecorderSupport, meanScore } from './recorderSupport'

afterEach(() => vi.restoreAllMocks())

describe('detectRecorderSupport', () => {
  it('reports unsupported when MediaRecorder is missing', () => {
    vi.stubGlobal('window', { MediaRecorder: undefined })
    expect(detectRecorderSupport().supported).toBe(false)
    vi.unstubAllGlobals()
  })

  it('picks the first supported container', () => {
    vi.stubGlobal('MediaRecorder', {
      isTypeSupported: (m: string) => m === 'audio/webm;codecs=opus',
    })
    const support = detectRecorderSupport()
    expect(support).toEqual({
      supported: true,
      mimeType: 'audio/webm;codecs=opus',
      extension: 'webm',
    })
    vi.unstubAllGlobals()
  })

  it('falls back to mp4 when webm is unsupported', () => {
    vi.stubGlobal('MediaRecorder', {
      isTypeSupported: (m: string) => m === 'audio/mp4',
    })
    const support = detectRecorderSupport()
    expect(support.supported).toBe(true)
    expect(support.extension).toBe('m4a')
    vi.unstubAllGlobals()
  })

  it('reports unsupported when no candidate container works', () => {
    vi.stubGlobal('MediaRecorder', { isTypeSupported: () => false })
    expect(detectRecorderSupport().supported).toBe(false)
    vi.unstubAllGlobals()
  })
})

describe('meanScore', () => {
  it('returns null for no windows', () => expect(meanScore([])).toBeNull())
  it('averages collected window scores', () => {
    expect(meanScore([0.2, 0.4, 0.6])).toBeCloseTo(0.4, 10)
  })
})
