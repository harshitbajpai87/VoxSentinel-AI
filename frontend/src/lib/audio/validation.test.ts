import { describe, expect, it } from 'vitest'
import {
  analyzeDecodedAudio,
  checkSelectedFile,
  computePeaks,
  DEFAULT_AUDIO_LIMITS,
  fileKeyOf,
  formatBytes,
  formatTime,
  type AudioLimits,
  type DecodedAudioLike,
} from './validation'

function mkFile(name: string, size = 10, type = 'audio/wav'): File {
  return new File([new Uint8Array(size)], name, { type, lastModified: 12345 })
}

const smallLimits: AudioLimits = { ...DEFAULT_AUDIO_LIMITS, maxFileSizeBytes: 100 }

function fakeBuffer(
  data: number[],
  opts: { duration?: number; sampleRate?: number; channels?: number } = {},
): DecodedAudioLike {
  return {
    duration: opts.duration ?? data.length / 1000,
    sampleRate: opts.sampleRate ?? 16000,
    numberOfChannels: opts.channels ?? 1,
    getChannelData: () => Float32Array.from(data),
  }
}

describe('checkSelectedFile', () => {
  it('accepts a plausible wav file', () => {
    const result = checkSelectedFile(mkFile('clip.wav', 1000))
    expect(result.ok).toBe(true)
    expect(result.errorCode).toBeNull()
  })

  it('rejects empty files with a friendly message', () => {
    const result = checkSelectedFile(new File([], 'clip.wav', { type: 'audio/wav' }))
    expect(result.ok).toBe(false)
    expect(result.errorCode).toBe('empty_file')
    expect(result.errorMessage).toMatch(/empty/i)
  })

  it('rejects unsupported extensions and lists the accepted formats', () => {
    const result = checkSelectedFile(mkFile('notes.txt'))
    expect(result.ok).toBe(false)
    expect(result.errorCode).toBe('unsupported_extension')
    expect(result.errorMessage).toMatch(/WAV, MP3, FLAC, M4A/)
  })

  it('rejects files above the size limit', () => {
    const result = checkSelectedFile(mkFile('big.wav', 1000), smallLimits)
    expect(result.ok).toBe(false)
    expect(result.errorCode).toBe('too_large')
    expect(result.errorMessage).toMatch(/larger than/)
  })

  it('warns — but does not reject — a suspicious reported MIME type', () => {
    const result = checkSelectedFile(mkFile('clip.wav', 500, 'application/zip'))
    expect(result.ok).toBe(true)
    expect(result.warnings).toHaveLength(1)
    expect(result.warnings[0]).toMatch(/application\/zip/)
  })

  it('does not warn for video/mp4 (common label for M4A audio)', () => {
    const result = checkSelectedFile(mkFile('clip.m4a', 500, 'video/mp4'))
    expect(result.ok).toBe(true)
    expect(result.warnings).toHaveLength(0)
  })
})

describe('fileKeyOf (duplicate prevention)', () => {
  it('gives identical files the same key and different files different keys', () => {
    const a = mkFile('clip.wav', 100)
    const b = mkFile('clip.wav', 100)
    const c = mkFile('clip.wav', 200)
    expect(fileKeyOf(a)).toBe(fileKeyOf(b))
    expect(fileKeyOf(a)).not.toBe(fileKeyOf(c))
  })
})

describe('analyzeDecodedAudio', () => {
  it('extracts metadata and passes clean audio', () => {
    const report = analyzeDecodedAudio(fakeBuffer(Array(5000).fill(0.3), { duration: 5, sampleRate: 44100 }))
    expect(report.durationS).toBe(5)
    expect(report.sampleRateHz).toBe(44100)
    expect(report.channels).toBe(1)
    expect(report.rms).toBeCloseTo(0.3, 2)
    expect(report.peak).toBeCloseTo(0.3, 2)
    expect(report.warnings).toHaveLength(0)
    expect(report.blockers).toHaveLength(0)
  })

  it('blocks silent audio as unusable', () => {
    const report = analyzeDecodedAudio(fakeBuffer(Array(5000).fill(0), { duration: 5 }))
    expect(report.blockers.join(' ')).toMatch(/silent/i)
  })

  it('warns about extremely quiet audio', () => {
    const report = analyzeDecodedAudio(fakeBuffer(Array(5000).fill(0.007), { duration: 5 }))
    expect(report.blockers).toHaveLength(0)
    expect(report.warnings.join(' ')).toMatch(/extremely low/i)
  })

  it('warns about very short recordings', () => {
    const report = analyzeDecodedAudio(fakeBuffer(Array(1000).fill(0.3), { duration: 1 }))
    expect(report.warnings.join(' ')).toMatch(/very short/i)
  })

  it('blocks recordings longer than the maximum duration', () => {
    const report = analyzeDecodedAudio(fakeBuffer(Array(1000).fill(0.3), { duration: 700 }))
    expect(report.blockers.join(' ')).toMatch(/longer than/)
  })

  it('warns about possible clipping', () => {
    const report = analyzeDecodedAudio(
      fakeBuffer([...Array(4999).fill(0.3), 1.0], { duration: 5 }),
    )
    expect(report.warnings.join(' ')).toMatch(/clipped/i)
  })
})

describe('computePeaks', () => {
  it('produces a correct min/max envelope per bucket (always includes the zero baseline)', () => {
    const data = [0, -1, 0.5, 0.25, -0.25, 0.75]
    const { mins, maxs } = computePeaks(Float32Array.from(data), 3)
    expect(mins).toHaveLength(3)
    expect(maxs).toHaveLength(3)
    expect(mins[0]).toBe(-1) // bucket 0: [0, -1]
    expect(maxs[0]).toBe(0)
    expect(mins[1]).toBe(0) // bucket 1: [0.5, 0.25] -> all positive, baseline 0
    expect(maxs[1]).toBe(0.5)
    expect(mins[2]).toBe(-0.25) // bucket 2: [-0.25, 0.75]
    expect(maxs[2]).toBe(0.75)
  })

  it('never requests more buckets than samples', () => {
    const { mins, maxs } = computePeaks(Float32Array.from([0.5]), 400)
    expect(mins).toHaveLength(1)
    expect(maxs).toHaveLength(1)
  })
})

describe('formatters', () => {
  it('formats bytes', () => {
    expect(formatBytes(500)).toBe('500 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(25 * 1024 * 1024)).toBe('25.0 MB')
  })
  it('formats playback time', () => {
    expect(formatTime(0)).toBe('0:00')
    expect(formatTime(65)).toBe('1:05')
    expect(formatTime(-3)).toBe('0:00')
  })
})
