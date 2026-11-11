// Navigation contract: the workspace exposes exactly the six active pages,
// and the four static pages (explainability, security, docs, settings) are gone.
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

vi.mock('../../lib/apiStatus', () => ({
  useApiStatus: () => ({
    status: 'online' as const,
    health: null,
    lastCheckedAt: null,
    refresh: async () => {},
  }),
}))

import { NAV_ITEMS, Sidebar } from './Sidebar'

afterEach(cleanup)

describe('Sidebar navigation', () => {
  it('exposes only the active workspace pages', () => {
    expect(NAV_ITEMS.map((i) => i.to)).toEqual([
      '/dashboard',
      '/analyze',
      '/record',
      '/verify',
      '/history',
      '/protection',
    ])
  })

  it('removed static pages are not registered as navigation items', () => {
    const routes = NAV_ITEMS.map((i) => i.to)
    for (const gone of ['/explainability', '/security', '/docs', '/settings']) {
      expect(routes).not.toContain(gone)
    }
  })

  it('renders the evaluation-only footer note', () => {
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>,
    )
    expect(screen.getByText('Research Prototype • For Evaluation Only')).toBeTruthy()
    expect(screen.queryByText(/Results never constitute proof/)).toBeNull()
  })
})
