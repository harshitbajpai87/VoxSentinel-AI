// Page tests for the evaluation workflow. The backend job is MOCKED here (jsdom
// cannot run real inference); these tests verify the UI's run/poll/render logic,
// not model performance. Canned metrics are never presented as measured results.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { HealthResponse } from '../lib/types'

const apiMock = vi.hoisted(() => ({
  getEvaluationSummary: vi.fn(),
  startEvaluationRun: vi.fn(),
  getEvaluationRun: vi.fn(),
}))
vi.mock('../lib/api', async () => {
  const actual = await vi.importActual<typeof import('../lib/api')>('../lib/api')
  return { ...actual, api: apiMock }
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

import EvaluationPage from './EvaluationPage'

afterEach(() => cleanup())

const METRICS = {
  n: 20,
  accuracy: 0.95,
  f1_synthetic: 0.94,
  precision_synthetic: 0.92,
  recall_synthetic: 0.96,
  roc_auc: 0.97,
  eer: 0.08,
  false_positive_rate: 0.1,
  false_negative_rate: 0.04,
  threshold: 0.5,
  positive_class: 'synthetic',
  confusion_matrix: { tn: 9, fp: 1, fn: 1, tp: 9 },
  roc_curve: [
    { fpr: 0, tpr: 0 },
    { fpr: 0.5, tpr: 0.9 },
    { fpr: 1, tpr: 1 },
  ],
}

describe('EvaluationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getEvaluationSummary.mockResolvedValue({ evaluated: false, message: 'No evaluation has been run yet.' })
  })

  it('shows the honest empty state when no run has been recorded', async () => {
    render(
      <MemoryRouter>
        <EvaluationPage />
      </MemoryRouter>,
    )
    expect(await screen.findByText('No evaluation results yet')).toBeTruthy()
    expect(screen.getByText('No evaluation has been run yet.')).toBeTruthy()
  })

  it('runs a job, polls it, and renders the real metric tiles on completion', async () => {
    apiMock.startEvaluationRun.mockResolvedValue({ job_id: 'job-1', status: 'running', progress: { done: 0, total: 20 } })
    apiMock.getEvaluationRun
      .mockResolvedValueOnce({ job_id: 'job-1', status: 'running', progress: { done: 10, total: 20 } })
      .mockResolvedValueOnce({ job_id: 'job-1', status: 'completed', progress: { done: 20, total: 20 }, metrics: METRICS })
    // first summary read: nothing run yet; after completion the summary reflects it
    apiMock.getEvaluationSummary
      .mockResolvedValueOnce({ evaluated: false, message: 'No evaluation has been run yet.' })
      .mockResolvedValue({ evaluated: true, evaluated_at: '2026-10-23T00:00:00Z', metrics: METRICS })

    render(
      <MemoryRouter>
        <EvaluationPage />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByRole('button', { name: /Run evaluation/i }))

    await waitFor(
      () => {
        expect(screen.getByText('Latest measured metrics')).toBeTruthy()
      },
      { timeout: 4000 },
    )
    expect(apiMock.startEvaluationRun).toHaveBeenCalledTimes(1)
    expect(apiMock.getEvaluationRun.mock.calls.length).toBeGreaterThanOrEqual(2)
    // tiles for the mocked metrics
    expect(screen.getByText('0.9700')).toBeTruthy() // ROC-AUC
    expect(screen.getByText('20')).toBeTruthy() // n samples
    // confusion matrix renders both true negatives and true positives
    expect(screen.getByText('Confusion matrix')).toBeTruthy()
    const cells = screen.getAllByText('9')
    expect(cells.length).toBe(2)
  })

  it('surfaces a failed job honestly', async () => {
    apiMock.startEvaluationRun.mockResolvedValue({ job_id: 'job-2', status: 'running', progress: { done: 0, total: 5 } })
    apiMock.getEvaluationRun.mockResolvedValue({ job_id: 'job-2', status: 'failed', progress: { done: 1, total: 5 }, error: 'eval manifest missing' })

    render(
      <MemoryRouter>
        <EvaluationPage />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByRole('button', { name: /Run evaluation/i }))

    await waitFor(
      () => {
        expect(screen.getByText(/eval manifest missing/i)).toBeTruthy()
      },
      { timeout: 4000 },
    )
  })
})
