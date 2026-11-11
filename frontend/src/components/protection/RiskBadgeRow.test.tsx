import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

afterEach(cleanup)
import { RiskBadgeRow } from './RiskBadgeRow'
import type { RiskBadge } from '../../lib/unified'

const BADGES: RiskBadge[] = [
  { id: 'risk', label: 'High risk', tone: 'critical', value: 'HIGH' },
  { id: 'voice_authenticity', label: 'Synthetic voice suspected', tone: 'critical', value: 0.87 },
  { id: 'speaker_match', label: 'Voice matches profile', tone: 'ok', value: 0.81 },
  { id: 'llm_reasoning', label: 'Local voxisentinel-mistral', tone: 'ok' },
]

describe('RiskBadgeRow', () => {
  it('renders every badge the backend sent', () => {
    render(<RiskBadgeRow badges={BADGES} />)
    expect(screen.getByTestId('risk-badges')).toBeTruthy()
    expect(screen.getByText('High risk')).toBeTruthy()
    expect(screen.getByText('Synthetic voice suspected')).toBeTruthy()
    expect(screen.getByText('Voice matches profile')).toBeTruthy()
    expect(screen.getByText('Local voxisentinel-mistral')).toBeTruthy()
  })

  it('shows a numeric score only where one exists', () => {
    render(<RiskBadgeRow badges={BADGES} />)
    expect(screen.getByTestId('badge-voice_authenticity').textContent).toContain('0.87')
    expect(screen.getByTestId('badge-llm_reasoning').textContent).not.toContain('0.')
  })

  it('renders nothing when the backend sent no badges', () => {
    const { container } = render(<RiskBadgeRow badges={[]} />)
    expect(container.firstChild).toBeNull()
  })
})
