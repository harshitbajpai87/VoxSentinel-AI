import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import WindowScoresChart from './WindowScoresChart'
import type { WindowScore } from '../../lib/types'

afterEach(() => cleanup())

const windows: WindowScore[] = [
  { start_s: 0, end_s: 4, score: 0.12 },
  { start_s: 4, end_s: 8, score: 0.87 },
  { start_s: 8, end_s: 12, score: 0.91 },
]

describe('WindowScoresChart', () => {
  it('renders one labeled segment per window', () => {
    render(<WindowScoresChart windows={windows} />)
    const imgs = screen.getAllByRole('img')
    expect(imgs.length).toBe(windows.length)
    expect(screen.getByText(/3 segments/)).toBeTruthy()
  })

  it('renders nothing without windows', () => {
    const { container } = render(<WindowScoresChart windows={[]} />)
    expect(container.firstChild).toBeNull()
  })
})
