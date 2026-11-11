import { useEffect, useRef, useState, type MouseEvent } from 'react'
import type { WaveformPeaks } from '../../lib/audio/validation'

interface WaveformCanvasProps {
  /** Min/max peak envelope computed from the real decoded samples. */
  peaks: WaveformPeaks
  /** Playback position as a 0..1 fraction of the duration. */
  progress: number
  onSeek?: (fraction: number) => void
  height?: number
  className?: string
}

/**
 * Canvas waveform of the actual audio. Renders the downsampled peak envelope
 * of real sample amplitudes — never random or decorative bars. Played region
 * is purple, unplayed is muted lavender; a navy playhead marks the position. Clicking
 * seeks. Redraws on resize via ResizeObserver.
 */
export function WaveformCanvas({
  peaks,
  progress,
  onSeek,
  height = 110,
  className = '',
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [width, setWidth] = useState(0)

  // Responsive width.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width)
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  // Draw.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    // Assigning width/height resets the transform, so scale is safe per draw.
    canvas.width = Math.max(1, Math.round(width * dpr))
    canvas.height = Math.round(height * dpr)
    ctx.scale(dpr, dpr)

    const w = width
    const h = height
    const mid = h / 2
    const scale = (h / 2) * 0.92
    ctx.clearRect(0, 0, w, h)

    const n = peaks.mins.length
    const step = w / n
    const barWidth = Math.max(1, step * 0.7)
    const playedX = Math.min(1, Math.max(0, progress)) * w

    for (let i = 0; i < n; i++) {
      const x = i * step
      const yMax = mid - peaks.maxs[i] * scale
      const yMin = mid - peaks.mins[i] * scale
      ctx.fillStyle = x <= playedX ? 'rgba(134, 82, 245, 0.85)' : 'rgba(123, 132, 153, 0.30)'
      ctx.fillRect(x, yMax, barWidth, Math.max(1, yMin - yMax))
    }

    // Playhead.
    if (progress > 0 && progress < 1) {
      ctx.fillStyle = '#713F60'
      ctx.fillRect(playedX, 0, 1.5, h)
    }
  }, [peaks, progress, width, height])

  const handleClick = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const w = rect.width || 300
    const fraction = Math.min(1, Math.max(0, (e.clientX - rect.left) / w))
    onSeek(fraction)
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{ height, width: '100%' }}
      className={`block cursor-pointer ${className}`}
      role="slider"
      aria-label="Audio timeline. Click to seek."
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(Math.min(1, Math.max(0, progress)) * 100)}
    />
  )
}
