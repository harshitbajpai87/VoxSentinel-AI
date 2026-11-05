// Browser-only audio decoding via the Web Audio API. Kept separate from
// validation.ts so the pure rules stay testable in Node.
//
// decodeAudioData is the authority here: a file only counts as valid audio if
// the browser can actually decode it. Some formats (notably FLAC) are not
// decodable in every browser — that is a real limitation we surface honestly
// instead of pretending all four formats always work.

export interface DecodeSupport {
  level: 'yes' | 'maybe' | 'no' | 'unknown'
  message: string | null
}

const MIME_BY_EXTENSION: Record<string, string> = {
  wav: 'audio/wav',
  mp3: 'audio/mpeg',
  flac: 'audio/flac',
  m4a: 'audio/mp4',
}

/**
 * Asks the browser whether it can *probably* play this format. This is only a
 * hint for messaging; decodeAudioData is still attempted either way.
 */
export function decodeSupportHint(extension: string): DecodeSupport {
  if (typeof document === 'undefined') return { level: 'unknown', message: null }
  const mime = MIME_BY_EXTENSION[extension]
  if (!mime) return { level: 'unknown', message: null }
  try {
    const audio = document.createElement('audio')
    const can = audio.canPlayType(mime)
    if (can === 'probably') return { level: 'yes', message: null }
    if (can === 'maybe') return { level: 'maybe', message: null }
    return {
      level: 'no',
      message: `This browser may not be able to decode ${extension.toUpperCase()} files for preview. Decoding will still be attempted.`,
    }
  } catch {
    return { level: 'unknown', message: null }
  }
}

let sharedContext: AudioContext | null = null

type AudioContextCtor = typeof AudioContext

function getAudioContext(): AudioContext {
  const w = window as unknown as {
    AudioContext?: AudioContextCtor
    webkitAudioContext?: AudioContextCtor
  }
  const Ctor = w.AudioContext ?? w.webkitAudioContext
  if (!Ctor) {
    throw new Error('The Web Audio API is not available in this browser.')
  }
  if (!sharedContext) sharedContext = new Ctor()
  return sharedContext
}

/**
 * Decodes the file's audio data with the Web Audio API. Rejects when the file
 * is corrupted or in a format this browser cannot decode.
 */
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer()
  const ctx = getAudioContext()
  try {
    return await ctx.decodeAudioData(arrayBuffer)
  } catch {
    throw new Error(
      'The browser could not decode this file as audio. It may be corrupted, or the format (for example FLAC) is not supported by this browser.',
    )
  }
}
