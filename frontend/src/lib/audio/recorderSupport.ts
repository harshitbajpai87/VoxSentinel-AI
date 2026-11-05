/**
 * Browser MediaRecorder capability helpers. Exported for tests: the UI must
 * degrade honestly (a clear message) when live recording is not supported.
 */

export interface RecorderSupport {
  supported: boolean
  mimeType: string
  extension: string
}

/** Picks the best supported audio recording container, or reports unsupported. */
export function detectRecorderSupport(): RecorderSupport {
  if (typeof window === 'undefined' || typeof MediaRecorder === 'undefined') {
    return { supported: false, mimeType: '', extension: '' }
  }
  const candidates = [
    { mimeType: 'audio/webm;codecs=opus', extension: 'webm' },
    { mimeType: 'audio/webm', extension: 'webm' },
    { mimeType: 'audio/mp4', extension: 'm4a' },
    { mimeType: 'audio/ogg;codecs=opus', extension: 'ogg' },
  ]
  for (const c of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(c.mimeType)) return { supported: true, ...c }
    } catch {
      /* isTypeSupported can throw on exotic browsers; keep probing */
    }
  }
  return { supported: false, mimeType: '', extension: '' }
}

/** Mean of the collected window scores; null when nothing was scored yet. */
export function meanScore(scores: number[]): number | null {
  if (!scores.length) return null
  return scores.reduce((a, b) => a + b, 0) / scores.length
}
