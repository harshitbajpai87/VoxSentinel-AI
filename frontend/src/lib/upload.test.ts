import { afterEach, describe, expect, it, vi } from 'vitest'
import { vi as vitest } from 'vitest'

// Mock only the browser transcoder: its job is covered by real-browser use, and
// jsdom has no AudioContext to decode with.
vitest.mock('./audio/transcode', async () => {
  const actual = await vitest.importActual<typeof import('./audio/transcode')>(
    './audio/transcode',
  )
  return { ...actual, toModelWav: vitest.fn() }
})

import { toModelWav } from './audio/transcode'
import {
  DEFAULT_MAX_UPLOAD_BYTES,
  getUploadPolicy,
  mapAnalyzeError,
  prepareUpload,
  type UploadPolicy,
} from './api'

const toModelWavMock = toModelWav as unknown as ReturnType<typeof vi.fn>

afterEach(() => {
  vi.unstubAllGlobals()
  toModelWavMock.mockReset()
})

function jsonRes(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status })
}

function sizedFile(name: string, bytes: number): File {
  const f = new File([new Uint8Array(8)], name, { type: 'application/octet-stream' })
  Object.defineProperty(f, 'size', { value: bytes })
  return f
}

const DECODES_EVERYTHING: UploadPolicy = {
  maxBytes: 25 * 1024 * 1024,
  decodableExtensions: ['wav', 'mp3', 'flac', 'ogg', 'oga', 'm4a', 'aac', 'opus', 'webm'],
  ffmpeg: true,
}

const LIBSNDFILE_ONLY: UploadPolicy = {
  maxBytes: 25 * 1024 * 1024,
  decodableExtensions: ['wav', 'mp3', 'flac', 'ogg', 'oga'],
  ffmpeg: false,
}

describe('getUploadPolicy', () => {
  it('reads the real ceiling and decoder list from /health', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonRes({
          status: 'ok',
          backend_ready: true,
          max_upload_bytes: 25 * 1024 * 1024,
          effective_max_upload_bytes: 4_500_000,
          audio: {
            decodable_extensions: ['wav', 'm4a', 'webm'],
            decoders: { ffmpeg: true },
          },
        }),
      ),
    )
    const policy = await getUploadPolicy(true)
    expect(policy.maxBytes).toBe(4_500_000)
    expect(policy.decodableExtensions).toContain('m4a')
    expect(policy.ffmpeg).toBe(true)
  })

  it('falls back to libsndfile coverage when /health cannot be reached', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new Error('offline'))))
    const policy = await getUploadPolicy(true)
    expect(policy.maxBytes).toBe(DEFAULT_MAX_UPLOAD_BYTES)
    expect(policy.decodableExtensions).toEqual(['wav', 'mp3', 'flac', 'ogg', 'oga'])
    expect(policy.ffmpeg).toBe(false)
  })
})

describe('prepareUpload', () => {
  it('sends the original untouched when the server decodes the container', async () => {
    const original = sizedFile('call.m4a', 5_000_000)
    const result = await prepareUpload(original, DECODES_EVERYTHING)
    expect(result).toBe(original)
    expect(toModelWavMock).not.toHaveBeenCalled()
  })

  it('rejects an oversized original with its real size and the real limit', async () => {
    const original = sizedFile('long.wav', 26 * 1024 * 1024)
    await expect(prepareUpload(original, DECODES_EVERYTHING)).rejects.toMatchObject({
      status: 413,
      code: 'file_too_large',
    })
    await expect(prepareUpload(original, DECODES_EVERYTHING)).rejects.toThrow(/26 MB/)
    await expect(prepareUpload(original, DECODES_EVERYTHING)).rejects.toThrow(/25 MB/)
  })

  it('converts in the browser only when the server cannot decode the format', async () => {
    const original = sizedFile('call.m4a', 3_000_000)
    const converted = sizedFile('call.wav', 4_000_000)
    toModelWavMock.mockResolvedValue(converted)

    const result = await prepareUpload(original, LIBSNDFILE_ONLY)
    expect(toModelWavMock).toHaveBeenCalledWith(original)
    expect(result).toBe(converted)
  })

  it('re-checks the size after conversion instead of letting the API reject it', async () => {
    const original = sizedFile('call.m4a', 5_500_000)
    const huge = sizedFile('call.wav', 40 * 1024 * 1024)
    toModelWavMock.mockResolvedValue(huge)

    await expect(prepareUpload(original, LIBSNDFILE_ONLY)).rejects.toMatchObject({
      status: 413,
      code: 'file_too_large',
    })
    await expect(prepareUpload(original, LIBSNDFILE_ONLY)).rejects.toThrow(/converts to/)
    // The message names both the source format and the way out.
    await expect(prepareUpload(original, LIBSNDFILE_ONLY)).rejects.toThrow(/M4A/)
  })

  it('leaves an unknown extension for the server to reject by name', async () => {
    const original = sizedFile('notes.txt', 400)
    const result = await prepareUpload(original, LIBSNDFILE_ONLY)
    expect(result).toBe(original)
    expect(toModelWavMock).not.toHaveBeenCalled()
  })

  it('treats a file at exactly the limit as acceptable', async () => {
    const original = sizedFile('exact.m4a', 25 * 1024 * 1024)
    await expect(prepareUpload(original, DECODES_EVERYTHING)).resolves.toBe(original)
  })

  it('keeps a converted file that still fits', async () => {
    const original = sizedFile('call.webm', 2_000_000)
    const converted = sizedFile('call.wav', 6_000_000)
    toModelWavMock.mockResolvedValue(converted)
    const result = await prepareUpload(original, LIBSNDFILE_ONLY)
    expect(result.size).toBe(6_000_000)
  })
})

describe('mapAnalyzeError', () => {
  it('prefers a specific message from the API over a generic phrase', () => {
    const message = mapAnalyzeError(
      413,
      'file_too_large',
      'The file is 30 MB but this deployment accepts up to 25 MB.',
    )
    expect(message).toContain('accepts up to 25 MB')
  })

  it('ignores a transport-style message that adds nothing', () => {
    const message = mapAnalyzeError(413, 'file_too_large', 'Backend returned 413 Payload Too Large')
    expect(message).toBe('The backend rejected the file as too large.')
  })

  it('explains a server without ffmpeg', () => {
    expect(mapAnalyzeError(503, 'decoder_unavailable', null)).toMatch(/ffmpeg/)
  })

  it('still maps the common statuses', () => {
    expect(mapAnalyzeError(415, 'unsupported_media_type', null)).toMatch(/format/i)
    expect(mapAnalyzeError(422, 'invalid_audio', null)).toMatch(/audio/i)
    expect(mapAnalyzeError(503, null, null)).toMatch(/model/i)
  })
})
