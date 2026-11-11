import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import LandingPage from './LandingPage'

vi.mock('../components/motion/Reveal', () => ({ Reveal: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))

function setup() { return render(<MemoryRouter><LandingPage /></MemoryRouter>) }
afterEach(cleanup)

describe('LandingPage', () => {
  it('offers the existing analysis and recording routes without claiming a detection result', () => {
    setup()
    expect(screen.getByRole('link', { name: 'Analyze a clip' }).getAttribute('href')).toBe('/analyze')
    expect(screen.getByRole('link', { name: /Or record your own/ }).getAttribute('href')).toBe('/record')
    expect(screen.getByText('Research Prototype v0.1 • For evaluation only')).toBeTruthy()
  })

  it('switches tool descriptions and destination links', () => {
    setup()
    fireEvent.click(screen.getByRole('tab', { name: /Speaker comparison/ }))
    expect(screen.getByRole('tab', { name: /Speaker comparison/ }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('link', { name: /Open speaker comparison/ }).getAttribute('href')).toBe('/verify')
    expect(screen.getByText('Compare two speaker recordings')).toBeTruthy()
  })

  it('supports keyboard navigation through the toolkit tabs', () => {
    setup()
    const first = screen.getByRole('tab', { name: /Audio analysis/ })
    first.focus()
    fireEvent.keyDown(first, { key: 'ArrowRight' })
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: /Speaker comparison/ }))
    fireEvent.keyDown(document.activeElement!, { key: 'End' })
    expect(screen.getByRole('tab', { name: /Live recording/ }).getAttribute('aria-selected')).toBe('true')
  })

  it('lets the visitor pause decorative motion', () => {
    const { container } = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Pause decorative motion' }))
    expect(container.querySelector('.landing')?.classList.contains('motion-paused')).toBe(true)
    expect(screen.getByRole('button', { name: 'Resume decorative motion' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('no longer links to the removed static pages', () => {
    setup()
    for (const href of ['/docs', '/settings', '/security', '/explainability']) {
      expect(document.querySelector(`a[href="${href}"]`)).toBeNull()
    }
  })

  it('provides a mobile menu with real project destinations', () => {
    setup()
    fireEvent.click(screen.getByRole('button', { name: 'Open menu' }))
    expect(screen.getByRole('navigation', { name: 'Mobile site navigation' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Close menu' }).getAttribute('aria-expanded')).toBe('true')
  })
})
