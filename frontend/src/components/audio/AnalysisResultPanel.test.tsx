// Component tests for AnalysisResultPanel's risk-percentage display.
//
// The "analysis records" below are canned mock payloads used to exercise the
// UI's rendering; they are never presented as real predictions.
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { AnalysisResultPanel } from './AnalysisResultPanel'
import type { AnalysisRecord } from '../../lib/types'

afterEach(() => cleanup())

function mockRecord(scores: Partial<AnalysisRecord['scores']>): AnalysisRecord {
  return {
    id: 'test-risk-1',
    created_at: '2026-10-25T10:00:00Z',
    input: { display_name: 'clip.wav', duration_s: 5, sample_rate_hz: 16000, channels: 1, quality_warnings: [] },
    model: { name: 'test-model', version: '0.0.1', mode: 'baseline' },
    scores: {
      raw_score: 0.42,
      raw_score_definition: 'TEST DEFINITION',
      calibrated_probability: null,
      uncertainty: null,
      uncertainty_method: null,
      ...scores,
    },
    decision: { category: 'inconclusive', label: 'Inconclusive TEST', reasons: ['r1'], recommendations: ['rec1'] },
    latency_ms: 7,
  }
}

describe('AnalysisResultPanel: risk percentage', () => {
  it('shows the calibrated probability as a prominent percentage', () => {
    render(<AnalysisResultPanel record={mockRecord({ calibrated_probability: 0.824, raw_score: 1.9 })} />)
    expect(screen.getByText('Risk percentage')).toBeTruthy()
    expect(screen.getAllByText('82.4%').length).toBeGreaterThanOrEqual(1)
  })

  it('honestly shows "not provided" when the backend sent no calibrated probability', () => {
    render(<AnalysisResultPanel record={mockRecord({ calibrated_probability: null })} />)
    expect(screen.getAllByText(/not provided/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders 0% as a percentage, not as a missing value', () => {
    render(<AnalysisResultPanel record={mockRecord({ calibrated_probability: 0 })} />)
    expect(screen.getAllByText('0.0%').length).toBeGreaterThanOrEqual(1)
  })
})
