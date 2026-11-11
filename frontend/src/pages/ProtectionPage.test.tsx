import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react'
import ProtectionPage from './ProtectionPage'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const TEXT_ANALYSIS = {
  id: 'text-1',
  created_at: '2026-11-11T09:00:00Z',
  status: 'completed',
  display_name: 'pasted transcript',
  size_bytes: 120,
  duration_s: 0,
  profile_id: null,
  profile_name: null,
  input_kind: 'text',
  schema_version: '1.0',
  stages: [],
  transcript: {
    status: 'completed',
    text: 'Papa, I had an accident. Send 1 lakh rupees now.',
    language: 'en',
    language_probability: null,
    model: 'client-supplied transcript',
    segments: [],
    error: null,
    limitations: [],
  },
  voice: {
    status: 'skipped',
    detail: 'Text-only request',
    score: null,
    calibrated_probability: null,
    suspected_synthetic: null,
    verdict: 'NOT_ASSESSED',
    model: null,
  },
  speaker_comparison: null,
  speaker_verification: {
    status: 'skipped',
    outcome: 'skipped',
    match: null,
    similarity: null,
    threshold: null,
    profile_id: null,
    profile_name: null,
    backend: 'ecapa',
    model: null,
    embedding_dim: null,
    sample_rate_hz: null,
    detail: 'No family voice profile was supplied, so no same/different-speaker verdict exists.',
    warnings: [],
  },
  content: {
    summary: 'Family emergency impersonation with an urgent payment demand.',
    primary_category: 'family_emergency_impersonation',
    signal_strength: 9,
    discussion_context: false,
    flags: { family_claim: true, payment_demand: true },
    categories: [],
    evidence_quotes: ['send 1 lakh rupees'],
    limitations: [],
    trained_classifier: { status: 'completed', label: 'scam', confidence: 0.91 },
    trained_classifier_raised_signal: false,
    transformer_ensemble: { status: 'completed', label: 'scam', score: 0.93 },
  },
  ai: {
    provider: 'router',
    provider_label: 'Local voxisentinel-mistral',
    role: 'primary_contextual_classifier',
    llm: { status: 'completed', result: null, detail: null },
    gemini: { status: 'completed', result: null },
  },
  risk: {
    category: 'HIGH',
    reasoning: ['Family emergency claim plus an urgent payment demand.'],
    recommended_actions: ['Call the family member directly.'],
    limitations: [],
  },
  risk_badges: [
    { id: 'risk', label: 'High risk', tone: 'critical', value: 'HIGH' },
    { id: 'voice_authenticity', label: 'No audio to screen', tone: 'info' },
    { id: 'speaker_match', label: 'Speaker not verified', tone: 'info' },
    { id: 'scam_intent', label: 'Scam language detected', tone: 'critical', value: 9 },
    { id: 'llm_reasoning', label: 'Local voxisentinel-mistral', tone: 'ok' },
  ],
  explanation: {
    verdict: 'HIGH',
    threat_type: 'family_emergency_impersonation',
    severity: 'HIGH',
    confidence: 0.75,
    summary: 'Caller impersonates a family member and demands an urgent transfer.',
    evidence: ['Rule match: "send 1 lakh rupees"'],
    transcript_excerpt: 'Papa, I had an accident.',
    manipulation_tactics: ['urgency'],
    requested_actions: ['transfer money'],
    speaker: { outcome: 'skipped', similarity: null, threshold: null },
    authenticity: { fused_score: null, verdict: 'not_assessed', components_used: [] },
    why_not_conclusive: ['Text-only analysis: voice authenticity cannot be assessed.'],
    recommended_actions: ['Call the family member directly.'],
    model_trail: {},
  },
  warnings: [],
  processing_ms: 42,
  pipeline: {},
}

function stubBackend() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const path = String(url)
      if (path.includes('/api/analyze')) {
        return new Response(JSON.stringify(TEXT_ANALYSIS), { status: 200 })
      }
      if (path.includes('/analyses')) {
        return new Response(JSON.stringify({ analyses: [], count: 0 }), { status: 200 })
      }
      if (path.includes('/family-voices')) {
        return new Response(JSON.stringify({ profiles: [], count: 0 }), { status: 200 })
      }
      return new Response(JSON.stringify({}), { status: 200 })
    }),
  )
}

describe('ProtectionPage unified pipeline', () => {
  it('analyzes a pasted transcript and renders badges plus the JSON explanation', async () => {
    stubBackend()
    render(<ProtectionPage />)

    fireEvent.click(screen.getByTestId('input-mode-text'))
    fireEvent.change(screen.getByTestId('protection-transcript'), {
      target: { value: 'Papa, I had an accident. Send 1 lakh rupees now.' },
    })
    fireEvent.click(screen.getByLabelText(/authorized to analyze this recording/i))

    const button = screen.getByRole('button', { name: /analyze transcript/i })
    await waitFor(() => expect(button.hasAttribute('disabled')).toBe(false))
    fireEvent.click(button)

    await waitFor(() => expect(screen.getByTestId('risk-badges')).toBeTruthy())
    expect(screen.getByText('Scam language detected')).toBeTruthy()
    expect(screen.getByText('Local voxisentinel-mistral')).toBeTruthy()

    // Speaker match status is shown as its own tile.
    expect(screen.getByText('Speaker match')).toBeTruthy()

    // The structured explanation is rendered and its raw JSON is available.
    expect(screen.getByTestId('threat-explanation')).toBeTruthy()
    fireEvent.click(screen.getByTestId('toggle-explanation-json'))
    expect(screen.getByTestId('explanation-json').textContent).toContain('"verdict": "HIGH"')
  })
})
