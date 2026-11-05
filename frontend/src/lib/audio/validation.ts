// Client-side audio validation rules. PURE functions, no browser APIs at module
// level, so they are unit-testable in Node and usable during SSR.
//
// IMPORTANT: client-side validation is a UX convenience filter only. The
// Python backend must independently re-validate every uploaded file; the
// frontend is NOT a security boundary.
//
// Also important: passing validation says NOTHING about whether a recording
// is genuine or synthetic. It only says the file is decodable audio that
// meets the input requirements for analysis.

export interface AudioLimits {
  /** Hard cap on upload size (frontend convenience; backend enforces its own). */
  maxFileSizeBytes: number
  /** Below this duration we warn that detection may be unreliable. */
  minDurationS: number
  /** Above this duration we refuse analysis (too large to process). */
  maxDurationS: number
  /** RMS below this is treated as silent — nothing to analyze. */
  silenceRms: number
  /** RMS below this (but above silence) is flagged as extremely quiet. */
  quietRms: number
  /** Peak at or above this is flagged as possible clipping. */
  clippingPeak: number
}

export const DEFAULT_AUDIO_LIMITS: AudioLimits = {
  maxFileSizeBytes: 25 * 1024 * 1024, // 25 MB
  minDurationS: 2,
  maxDurationS: 600, // 10 minutes
  silenceRms: 0.005,
  quietRms: 0.02,
  clippingPeak: 0.995,
}

export const ACCEPTED_EXTENSIONS = ['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg', 'oga', 'opus', 'webm'] as const
export type AcceptedExtension = (typeof ACCEPTED_EXTENSIONS)[number]

export const ACCEPTED_EXTENSIONS_LABEL = 'WAV, MP3, FLAC, M4A'

export type SelectionErrorCode =
  | 'empty_file'
  | 'unsupported_extension'
  | 'too_large'

export interface SelectionCheck {
  ok: boolean
  errorCode: SelectionErrorCode | null
  errorMessage: string | null
  /** Non-fatal notices, e.g. a suspicious reported MIME type. */
  warnings: string[]
  /** Stable identity of the file, used to prevent duplicate processing. */
  fileKey: string
}

/** A stable key for "is this the exact same file I already analyzed?". */
export function fileKeyOf(file: File): string {
  return `${file.name}::${file.size}::${file.lastModified}`
}

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  if (dot <= 0 || dot === name.length - 1) return ''
  return name.slice(dot + 1).toLowerCase()
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * Fast, synchronous checks before we spend time decoding. File extensions and
 * browser MIME types are treated as hints: a wrong extension is rejected (the
 * backend would reject it too), but an odd MIME type is only a warning because
 * browsers frequently report audio containers incorrectly. The real proof is
 * decoding, which happens next.
 */
export function checkSelectedFile(
  file: File,
  limits: AudioLimits = DEFAULT_AUDIO_LIMITS,
): SelectionCheck {
  const warnings: string[] = []

  if (file.size === 0) {
    return {
      ok: false,
      errorCode: 'empty_file',
      errorMessage: 'This file is empty (0 bytes).',
      warnings,
      fileKey: fileKeyOf(file),
    }
  }

  const ext = extensionOf(file.name)
  if (!ACCEPTED_EXTENSIONS.includes(ext as AcceptedExtension)) {
    return {
      ok: false,
      errorCode: 'unsupported_extension',
      errorMessage: `“${file.name}” is not a supported audio format. Accepted formats: ${ACCEPTED_EXTENSIONS_LABEL}.`,
      warnings,
      fileKey: fileKeyOf(file),
    }
  }

  if (file.size > limits.maxFileSizeBytes) {
    return {
      ok: false,
      errorCode: 'too_large',
      errorMessage: `This file is ${formatBytes(file.size)}, larger than the ${formatBytes(
        limits.maxFileSizeBytes,
      )} limit.`,
      warnings,
      fileKey: fileKeyOf(file),
    }
  }

  // MIME is a hint only: real-world files are often mislabeled by the OS or
  // browser, and M4A is sometimes reported as video/mp4. Warn, don't reject.
  if (
    file.type &&
    !file.type.startsWith('audio/') &&
    file.type !== 'video/mp4' &&
    file.type !== 'application/octet-stream'
  ) {
    warnings.push(
      `The browser reports this file as “${file.type}”, not audio. It will still be decoded to check it really contains audio.`,
    )
  }

  return { ok: true, errorCode: null, errorMessage: null, warnings, fileKey: fileKeyOf(file) }
}

/** Minimal structural shape of an AudioBuffer (works in Node tests too). */
export interface DecodedAudioLike {
  duration: number
  sampleRate: number
  numberOfChannels: number
  getChannelData(channel: number): Float32Array
}

export interface AudioQualityReport {
  durationS: number
  sampleRateHz: number
  channels: number
  /** Maximum absolute sample value (0..1). */
  peak: number
  /** Root-mean-square level across all channels (0..1). */
  rms: number
  /** Quality problems that do not block analysis. */
  warnings: string[]
  /** Problems that make analysis impossible or meaningless. */
  blockers: string[]
}

/**
 * Measures the decoded audio and produces honest quality findings.
 * Returns warnings (analysis still possible) and blockers (unable to analyze).
 */
export function analyzeDecodedAudio(
  buffer: DecodedAudioLike,
  limits: AudioLimits = DEFAULT_AUDIO_LIMITS,
): AudioQualityReport {
  const warnings: string[] = []
  const blockers: string[] = []

  const durationS = buffer.duration
  const channels = buffer.numberOfChannels

  let peak = 0
  let sumSquares = 0
  let sampleCount = 0
  for (let ch = 0; ch < channels; ch++) {
    const data = buffer.getChannelData(ch)
    for (let i = 0; i < data.length; i++) {
      const v = data[i]
      const abs = Math.abs(v)
      if (abs > peak) peak = abs
      sumSquares += v * v
      sampleCount++
    }
  }
  const rms = sampleCount > 0 ? Math.sqrt(sumSquares / sampleCount) : 0

  if (rms < limits.silenceRms) {
    blockers.push(
      'The recording appears to be silent — there is no usable audio to analyze.',
    )
  } else if (rms < limits.quietRms) {
    warnings.push(
      'The audio level is extremely low, which can make detection unreliable.',
    )
  }

  if (durationS < limits.minDurationS) {
    warnings.push(
      `The recording is very short (${durationS.toFixed(1)} s). Detection results on short clips may be unreliable.`,
    )
  }
  if (durationS > limits.maxDurationS) {
    blockers.push(
      `The recording is ${durationS.toFixed(0)} s, longer than the ${limits.maxDurationS} s limit.`,
    )
  }

  if (peak >= limits.clippingPeak) {
    warnings.push(
      'Samples reach maximum level — the recording may be clipped. Clipping can distort detection features.',
    )
  }

  return {
    durationS,
    sampleRateHz: buffer.sampleRate,
    channels,
    peak,
    rms,
    warnings,
    blockers,
  }
}

export interface WaveformPeaks {
  mins: Float32Array
  maxs: Float32Array
}

/**
 * Downsamples a channel to a min/max peak envelope for efficient canvas
 * rendering. The waveform then shows real sample amplitudes.
 */
export function computePeaks(
  data: ArrayLike<number>,
  bucketCount: number,
): WaveformPeaks {
  const n = data.length
  const count = Math.max(1, Math.min(bucketCount, n))
  const mins = new Float32Array(count)
  const maxs = new Float32Array(count)
  const per = n / count
  for (let b = 0; b < count; b++) {
    const start = Math.floor(b * per)
    const end = Math.min(n, Math.max(start + 1, Math.floor((b + 1) * per)))
    let mn = 0
    let mx = 0
    for (let i = start; i < end; i++) {
      const v = data[i]
      if (v < mn) mn = v
      if (v > mx) mx = v
    }
    mins[b] = mn
    maxs[b] = mx
  }
  return { mins, maxs }
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
