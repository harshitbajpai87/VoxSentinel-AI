import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { computePeaks } from '../../lib/audio/validation'
import { WaveformCanvas } from './WaveformCanvas'

const data = Float32Array.from([0, -0.5, 0.5, 0.25, -0.25, 0.75, 0.1, -0.1, 0.3, 0.2])
const peaks = computePeaks(data, 10)

afterEach(() => cleanup())

describe('WaveformCanvas', () => {
  it('renders an accessible slider timeline', () => {
    render(<WaveformCanvas peaks={peaks} progress={0.4} />)
    const slider = screen.getByRole('slider', { name: /audio timeline/i })
    expect(slider).toBeTruthy()
    expect(slider.getAttribute('aria-valuenow')).toBe('40')
  })

  it('reports seek fractions from click position', () => {
    const onSeek = vi.fn()
    render(<WaveformCanvas peaks={peaks} progress={0} onSeek={onSeek} />)
    const canvas = screen.getByRole('slider')

    // jsdom reports a zero-width rect; the component falls back to 300px,
    // so clientX=150 seeks to fraction 0.5.
    fireEvent.click(canvas, { clientX: 150 })
    expect(onSeek).toHaveBeenCalledTimes(1)
    const fraction = onSeek.mock.calls[0][0] as number
    expect(fraction).toBeGreaterThanOrEqual(0)
    expect(fraction).toBeLessThanOrEqual(1)
    expect(Math.abs(fraction - 0.5)).toBeLessThan(0.02)
  })

  it('does not call onSeek when it is not provided', () => {
    render(<WaveformCanvas peaks={peaks} progress={0.5} />)
    const canvas = screen.getByRole('slider')
    expect(() => fireEvent.click(canvas, { clientX: 100 })).not.toThrow()
  })
})
