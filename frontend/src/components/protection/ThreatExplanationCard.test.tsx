import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen, fireEvent } from '@testing-library/react'

afterEach(cleanup)
import { ThreatExplanationCard } from './ThreatExplanationCard'
import type { ThreatExplanation } from '../../lib/unified'

const EXPLANATION: ThreatExplanation = {
  verdict: 'HIGH',
  threat_type: 'FAMILY_EMERGENCY_IMPERSONATION',
  severity: 'HIGH',
  confidence: 0.75,
  summary: 'Caller claims an accident and demands an urgent transfer.',
  evidence: ['urgency: "within 10 minutes"'],
  transcript_excerpt: 'Papa, I had an accident',
  manipulation_tactics: ['isolation'],
  requested_actions: ['transfer money'],
  speaker: { outcome: 'likely_same_speaker', similarity: 0.9, threshold: 0.62 },
  authenticity: { fused_score: 0.44, verdict: 'inconclusive', components_used: ['detector'] },
  why_not_conclusive: ['Acoustic authenticity was inconclusive.'],
  recommended_actions: ['Call the family member directly.'],
  model_trail: { llm_reasoning: 'mistral-small-latest' },
}

describe('ThreatExplanationCard', () => {
  it('shows the structured verdict and the recommended actions', () => {
    render(<ThreatExplanationCard explanation={EXPLANATION} />)
    expect(screen.getByText('HIGH')).toBeTruthy()
    expect(screen.getByText(/family emergency impersonation/i)).toBeTruthy()
    expect(screen.getByText('Caller claims an accident and demands an urgent transfer.')).toBeTruthy()
    expect(screen.getByText(/within 10 minutes/)).toBeTruthy()
    expect(screen.getByText('Call the family member directly.')).toBeTruthy()
    // The "what this does not prove" sub-list was removed from the card on
    // request; the JSON view still exposes why_not_conclusive for integrators.
    expect(screen.queryByText('What this does not prove')).toBeNull()
  })

  it('reveals the exact JSON payload on demand', () => {
    render(<ThreatExplanationCard explanation={EXPLANATION} />)
    expect(screen.queryByTestId('explanation-json')).toBeNull()
    fireEvent.click(screen.getByTestId('toggle-explanation-json'))
    const json = screen.getByTestId('explanation-json')
    expect(json.textContent).toContain('"verdict": "HIGH"')
    expect(json.textContent).toContain('"why_not_conclusive"')
  })
})
