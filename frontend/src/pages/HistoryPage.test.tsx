// History page tests: server-side filter wiring and the DB-backed CSV export.
// The backend is MOCKED in jsdom; these verify UI behavior only.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { AnalysisRecord, HealthResponse } from '../lib/types'

const apiMock = vi.hoisted(() => ({
  listAnalyses: vi.fn(),
}))
const exportMock = vi.hoisted(() => vi.fn())
vi.mock('../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../lib/api')>('../lib/api')
  return { ...actual, api: apiMock, exportAnalysesCsv: exportMock }
})

const statusMock = vi.hoisted(() => ({
  value: {
    status: 'online' as const,
    health: null as HealthResponse | null,
    lastCheckedAt: null as Date | null,
    refresh: async () => {},
  },
}))
vi.mock('../lib/apiStatus', () => ({ useApiStatus: () => statusMock.value }))

import HistoryPage from './HistoryPage'

const RECORD: AnalysisRecord = {
  id: 'abcd1234-0000-0000-0000-000000000000',
  created_at: '2026-10-23T00:00:00Z',
  input: { display_name: 'clip.wav', mime_type: 'audio/wav', size_bytes: 1000, duration_s: 4.2 },
  model: { name: 'vox-mfcc', version: '0.2.0', mode: 'production' },
  scores: { raw_score: 0.9 },
  decision: { category: 'high_risk', label: 'Synthetic', confidence: 0.9, reasons: [], recommendations: [] },
  latency_ms: 42,
} as unknown as AnalysisRecord

afterEach(() => cleanup())

describe('HistoryPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listAnalyses.mockResolvedValue({ analyses: [RECORD], total: 1 })
  })

  it('loads records and passes the server-side category filter', async () => {
    render(
      <MemoryRouter>
        <HistoryPage />
      </MemoryRouter>,
    )
    await screen.findByText('clip.wav')
    expect(apiMock.listAnalyses).toHaveBeenCalledWith(undefined)

    fireEvent.change(screen.getByLabelText(/Filter by risk category/i), {
      target: { value: 'high_risk' },
    })
    await waitFor(() => {
      const calls = apiMock.listAnalyses.mock.calls
      const last = calls.length ? calls[calls.length - 1][0] : undefined
      expect(last).toEqual({ category: 'high_risk' })
    })
  })

  it('exports the backend CSV when records exist', async () => {
    exportMock.mockResolvedValue(new Blob(['id,created_at\n'], { type: 'text/csv' }))
    render(
      <MemoryRouter>
        <HistoryPage />
      </MemoryRouter>,
    )
    const btn = await screen.findByRole('button', { name: /Export CSV/i })
    expect(btn.getAttribute('disabled')).toBeNull()
    fireEvent.click(btn)
    await waitFor(() => expect(exportMock).toHaveBeenCalledTimes(1))
  })

  it('disables export while offline or with no records', async () => {
    apiMock.listAnalyses.mockResolvedValue({ analyses: [], total: 0 })
    render(
      <MemoryRouter>
        <HistoryPage />
      </MemoryRouter>,
    )
    const btn = await screen.findByRole('button', { name: /Export CSV/i })
    expect(btn.getAttribute('disabled')).not.toBeNull()
  })
})
