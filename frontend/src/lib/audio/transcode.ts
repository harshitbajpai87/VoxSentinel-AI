// Server-side decoding is authoritative: a backend WITH ffmpeg decodes
// WAV/MP3/FLAC/OGG/M4A/AAC/WebM/Opus itself, so the original file is uploaded
// untouched (smallest possible body, no double encoding). Converting in the
// browser is a FALLBACK for hosts that have no ffmpeg (the slim serverless
// bundle) — see prepareUpload() in lib/api.ts.
//
// When the fallback does run, the WAV is normalised to 16 kHz mono, the rate the
// models use internally. This matters: a 5.5 MB M4A holds several minutes of
// audio, which expands to well over 25 MB as 44.1 kHz stereo PCM and used to be
// rejected as "too large" after passing the client-side size check.

import { decodeAudioFile } from './decode'

/** Extensions libsndfile covers on every deployment. */
export const BACKEND_NATIVE_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'oga'] as const

/** Containers only a backend WITH ffmpeg can decode. */
export const FFMPEG_ONLY_EXTENSIONS = [
  'm4a', 'm4b', 'mp4', 'aac', 'webm', 'opus', 'mkv', 'amr', '3gp', 'wma',
] as const

/** Internal model rate: both the detector and the speaker embedder use 16 kHz. */
export const MODEL_SAMPLE_RATE = 16000

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot + 1).toLowerCase()
}

/** True when the file must be converted in the browser before upload. */
export function needsTranscode(name: string): boolean {
  const ext = extensionOf(name)
  if (!ext) return true
  return !(BACKEND_NATIVE_EXTENSIONS as readonly string[]).includes(ext)
}

/** True when the extension is a legitimate audio container we accept at all. */
export function isKnownAudioExtension(name: string): boolean {
  const ext = extensionOf(name)
  if (!ext) return false
  return (
    (BACKEND_NATIVE_EXTENSIONS as readonly string[]).includes(ext) ||
    (FFMPEG_ONLY_EXTENSIONS as readonly string[]).includes(ext)
  )
}

/** Interleave Float32 channels into 16-bit PCM and wrap in a RIFF/WAVE header. */
export function encodeWavFromChannels(channels: Float32Array[], sampleRate: number): Blob {
  const channelCount = Math.max(1, channels.length)
  const frames = channels[0]?.length ?? 0
  const bytesPerSample = 2
  const blockAlign = channelCount * bytesPerSample
  const dataBytes = frames * blockAlign
  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }

  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // format = PCM
  view.setUint16(22, channelCount, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true) // bits per sample
  writeAscii(36, 'data')
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let frame = 0; frame < frames; frame += 1) {
    for (let ch = 0; ch < channelCount; ch += 1) {
      const sample = channels[ch]?.[frame] ?? 0
      const clamped = Math.max(-1, Math.min(1, sample))
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
      offset += bytesPerSample
    }
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

/** Mix every channel down to one channel (mono). */
function toMono(channels: Float32Array[]): Float32Array {
  if (channels.length <= 1) return channels[0] ?? new Float32Array(0)
  const frames = channels[0].length
  const mono = new Float32Array(frames)
  for (const channel of channels) {
    for (let i = 0; i < frames; i += 1) mono[i] += channel[i] ?? 0
  }
  for (let i = 0; i < frames; i += 1) mono[i] /= channels.length
  return mono
}

/**
 * Decode any container the browser understands and re-encode it as 16 kHz mono
 * PCM WAV — the internal model rate, and ~3x smaller than 48 kHz stereo PCM.
 */
export async function toModelWav(file: File): Promise<File> {
  let decoded: AudioBuffer
  try {
    decoded = await decodeAudioFile(file)
  } catch {
    // The browser could not decode this container (AMR/WMA and some DRM or
    // proprietary variants). Say which file failed and what does work instead
    // of surfacing a generic failure.
    throw new Error(
      `Your browser could not decode “${file.name}”. Try WAV, MP3, M4A, AAC, FLAC, OGG or OPUS.`,
    )
  }
  const channels: Float32Array[] = []
  for (let ch = 0; ch < decoded.numberOfChannels; ch += 1) {
    channels.push(decoded.getChannelData(ch))
  }
  const mono = toMono(channels)
  // 16 kHz mono: the backend resamples to this rate for inference anyway.
  const wav = encodeWavFromChannels([mono], MODEL_SAMPLE_RATE)
  const base = file.name.replace(/\.[^./\\]+$/, '') || 'recording'
  return new File([wav], `${base}.wav`, { type: 'audio/wav' })
}

/**
 * Returns a file the backend can always decode. Native formats pass through
 * untouched; everything else is decoded by the browser and re-encoded as WAV.
 */
export async function toBackendAudio(file: File): Promise<File> {
  if (!needsTranscode(file.name)) return file
  return toModelWav(file)
}
