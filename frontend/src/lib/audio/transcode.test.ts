import { describe, expect, it } from 'vitest'
import {
  BACKEND_NATIVE_EXTENSIONS,
  encodeWavFromChannels,
  extensionOf,
  needsTranscode,
} from './transcode'

describe('needsTranscode', () => {
  it('passes through the formats the deployed API decodes', () => {
    for (const ext of BACKEND_NATIVE_EXTENSIONS) {
      expect(needsTranscode(`clip.${ext}`)).toBe(false)
    }
  })

  it('converts phone and browser recording formats', () => {
    for (const name of ['clip.m4a', 'clip.aac', 'clip.webm', 'clip.opus', 'clip.amr', 'clip.aiff']) {
      expect(needsTranscode(name)).toBe(true)
    }
  })

  it('treats a nameless file as needing conversion', () => {
    expect(needsTranscode('recording')).toBe(true)
  })

  it('reads the extension case-insensitively', () => {
    expect(extensionOf('CLIP.M4A')).toBe('m4a')
    expect(needsTranscode('CLIP.MP3')).toBe(false)
  })
})

async function bytesOf(blob: Blob): Promise<ArrayBuffer> {
  // jsdom's Blob has no arrayBuffer() in this environment; FileReader always works.
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(reader.error)
    reader.readAsArrayBuffer(blob)
  })
}

function readAscii(view: DataView, offset: number, length: number): string {
  let out = ''
  for (let i = 0; i < length; i += 1) out += String.fromCharCode(view.getUint8(offset + i))
  return out
}

describe('encodeWavFromChannels', () => {
  it('writes a valid mono PCM WAV header and samples', async () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1])
    const blob = encodeWavFromChannels([samples], 16000)
    const view = new DataView(await bytesOf(blob))

    expect(readAscii(view, 0, 4)).toBe('RIFF')
    expect(readAscii(view, 8, 4)).toBe('WAVE')
    expect(readAscii(view, 12, 4)).toBe('fmt ')
    expect(view.getUint16(20, true)).toBe(1) // PCM
    expect(view.getUint16(22, true)).toBe(1) // mono
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint16(34, true)).toBe(16) // bit depth
    expect(readAscii(view, 36, 4)).toBe('data')
    expect(view.getUint32(40, true)).toBe(samples.length * 2)
    expect(view.byteLength).toBe(44 + samples.length * 2)

    // Values are little-endian int16 and clamped into range.
    expect(view.getInt16(44, true)).toBe(0)
    // setInt16 truncates toward zero, so allow one LSB of slack.
    expect(Math.abs(view.getInt16(46, true) - 0.5 * 0x7fff)).toBeLessThanOrEqual(1)
    expect(Math.abs(view.getInt16(48, true) + 0.5 * 0x8000)).toBeLessThanOrEqual(1)
    expect(view.getInt16(52, true)).toBe(-0x8000) // -1 clamps, never wraps
  })

  it('interleaves stereo samples', async () => {
    const left = new Float32Array([1, 0])
    const right = new Float32Array([-1, 0])
    const blob = encodeWavFromChannels([left, right], 44100)
    const view = new DataView(await bytesOf(blob))

    expect(view.getUint16(22, true)).toBe(2)
    expect(view.getUint32(24, true)).toBe(44100)
    expect(view.getUint16(32, true)).toBe(4) // block align
    expect(view.getInt16(44, true)).toBe(0x7fff) // L frame 0
    expect(view.getInt16(46, true)).toBe(-0x8000) // R frame 0
  })
})
