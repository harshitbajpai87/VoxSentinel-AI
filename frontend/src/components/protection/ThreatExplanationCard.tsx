import { useState } from 'react'
import type { ThreatExplanation } from '../../lib/unified'
import { Card, Button, Pill } from '../ui'

type PillTone = 'ok' | 'warn' | 'err' | 'info' | 'neutral'

function toneFor(severity: string | null): PillTone {
  switch ((severity ?? '').toUpperCase()) {
    case 'CRITICAL':
    case 'HIGH':
      return 'err'
    case 'MEDIUM':
      return 'warn'
    case 'LOW':
      return 'ok'
    default:
      return 'neutral'
  }
}

/**
 * The structured threat explanation, plus the raw JSON the API returned.
 *
 * The JSON tab is the exact payload an integrator can diff against, so nothing
 * shown here is re-worded on the client.
 */
export function ThreatExplanationCard({ explanation }: { explanation: ThreatExplanation }) {
  const [showJson, setShowJson] = useState(false)
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(explanation, null, 2))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setShowJson(true)
    }
  }

  return (
    <Card
      title="Threat explanation"
      subtitle="Structured verdict from the reasoning model and the evidence behind it"
      actions={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={copy} data-testid="copy-explanation">
            {copied ? 'Copied' : 'Copy JSON'}
          </Button>
          <Button variant="secondary" onClick={() => setShowJson((v) => !v)} data-testid="toggle-explanation-json">
            {showJson ? 'Hide JSON' : 'Show JSON'}
          </Button>
        </div>
      }
    >
      <div className="space-y-4" data-testid="threat-explanation">
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone={toneFor(explanation.verdict)}>{explanation.verdict ?? 'inconclusive'}</Pill>
          {explanation.threat_type ? <Pill tone="neutral">{explanation.threat_type.replace(/_/g, ' ')}</Pill> : null}
          {explanation.confidence != null ? (
            <span className="text-xs text-ink-faint">model confidence {(explanation.confidence * 100).toFixed(0)}%</span>
          ) : null}
        </div>

        {explanation.summary ? <p className="text-sm text-ink">{explanation.summary}</p> : null}

        {explanation.evidence.length ? (
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">Evidence</p>
            <ul className="mt-1 space-y-1 text-sm text-ink-dim">
              {explanation.evidence.map((item, index) => (
                <li key={index} className="border-l-2 border-line pl-2">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {explanation.recommended_actions.length ? (
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">What to do</p>
            <ol className="mt-1 list-decimal space-y-1 pl-5 text-sm text-ink">
              {explanation.recommended_actions.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ol>
          </div>
        ) : null}

        {showJson ? (
          <pre
            className="max-h-80 overflow-auto rounded-lg border border-line bg-surface p-3 text-xs text-ink-dim"
            data-testid="explanation-json"
          >
            {JSON.stringify(explanation, null, 2)}
          </pre>
        ) : null}
      </div>
    </Card>
  )
}

export default ThreatExplanationCard
