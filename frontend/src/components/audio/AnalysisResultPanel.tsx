import type { AnalysisRecord } from '../../lib/types'
import {
  Button,
  Callout,
  Card,
  Pill,
} from '../ui'
import { DownloadIcon } from '../icons'
import WindowScoresChart from './WindowScoresChart'

const CATEGORY_TONE: Record<string, 'ok' | 'warn' | 'err' | 'info' | 'neutral'> = {
  high_risk: 'err',
  elevated_risk: 'warn',
  low_observed_risk: 'ok',
  inconclusive: 'info',
  unable_to_analyze: 'neutral',
}

function downloadJsonReport(record: AnalysisRecord) {
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `voxsentinel-analysis-${record.id}.json`
  a.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/**
 * Displays a REAL analysis result returned by the backend. Every field shown
 * comes from the response; nothing is simulated. Values the backend did not
 * provide render as "not provided" instead of a fabricated number.
 */
export function AnalysisResultPanel({ record }: { record: AnalysisRecord }) {
  const tone = CATEGORY_TONE[record.decision.category] ?? 'neutral'
  const scores = record.scores
  const evidence = record.evidence

  return (
    <Card
      title="Analysis result"
      subtitle={`${new Date(record.created_at).toLocaleString()} · ${record.latency_ms} ms`}
      actions={
        <Button variant="secondary" onClick={() => downloadJsonReport(record)}>
          <DownloadIcon width={14} height={14} /> Download JSON report
        </Button>
      }
    >
      <div className="space-y-5">
        {/* Outcome */}
        <div className="flex flex-wrap items-center gap-3">
          <Pill tone={tone}>{record.decision.category.replace(/_/g, ' ')}</Pill>
          <p className="text-sm font-medium text-ink">{record.decision.label}</p>
        </div>

        {/* Risk percentage — the calibrated probability, shown as a percentage.
            If the backend did not provide it, say so; never invent a number. */}
        <div className="rounded-lg border border-line bg-surface-2 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-dim">
            Risk percentage
          </p>
          {scores.calibrated_probability !== null && scores.calibrated_probability !== undefined ? (
            <>
              <p className="mt-1 text-4xl font-semibold text-ink">
                {(scores.calibrated_probability * 100).toFixed(1)}%
              </p>
            </>
          ) : (
            <>
              <p className="mt-1 text-3xl font-semibold text-ink-faint">not provided</p>
              <p className="mt-1 text-xs text-ink-faint">
                The backend did not return a calibrated probability for this clip.
              </p>
            </>
          )}
        </div>

        {/* Scores — kept separate, never merged. */}
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-dim">
            Scores
          </h3>
          <dl className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-md border border-line bg-surface-2/50 p-3">
              <dt className="text-xs text-ink-dim">Raw model score</dt>
              <dd className="mt-1 font-mono text-lg text-ink">
                {typeof scores.raw_score === 'number'
                  ? scores.raw_score.toFixed(5)
                  : (scores.raw_score ?? '—')}
              </dd>
            </div>
            <div className="rounded-md border border-line bg-surface-2/50 p-3">
              <dt className="text-xs text-ink-dim">Calibrated probability</dt>
              <dd className="mt-1 font-mono text-lg text-ink">
                {scores.calibrated_probability !== null && scores.calibrated_probability !== undefined
                  ? `${(scores.calibrated_probability * 100).toFixed(1)}%`
                  : 'not provided'}
              </dd>
            </div>
            <div className="rounded-md border border-line bg-surface-2/50 p-3">
              <dt className="text-xs text-ink-dim">Uncertainty estimate</dt>
              <dd className="mt-1 font-mono text-lg text-ink">
                {scores.uncertainty ?? 'not provided'}
              </dd>
            </div>
          </dl>
        </div>

        {/* Model provenance */}
        <div className="flex flex-wrap gap-2 text-xs text-ink-dim">
          <Pill tone="neutral">
            Input: {record.input.duration_s.toFixed(1)} s · {record.input.sample_rate_hz} Hz ·{' '}
            {record.input.channels} ch
          </Pill>
        </div>

        {/* Measured features (facts about the audio). */}
        {evidence && evidence.measured_features.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-dim">
              Measured features
            </h3>
            <table className="w-full text-sm">
              <tbody className="divide-y divide-line/60">
                {evidence.measured_features.map((f) => (
                  <tr key={f.name}>
                    <td className="py-2 text-ink-dim">{f.name}</td>
                    <td className="py-2 text-right font-mono text-ink">
                      {f.value} {f.unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Model-derived explanation. */}
        {evidence?.model_explanation && (
          <Callout tone="info" title="Model-derived explanation">
            {evidence.model_explanation}
          </Callout>
        )}

        {/* Segment-level evidence from the same model (explainability). */}
        {(evidence?.time_window_scores?.length ?? 0) > 0 && (
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-dim">
              Segment-level scores · evidence, not the verdict
            </h3>
            <WindowScoresChart windows={evidence?.time_window_scores ?? []} />
          </div>
        )}

        {/* Policy decision. */}
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-dim">
              Decision reasons
            </h3>
            <ul className="space-y-1.5 text-sm text-ink-dim">
              {record.decision.reasons.map((r) => (
                <li key={r}>· {r}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-dim">
              Recommended next steps
            </h3>
            <ol className="space-y-1.5 text-sm text-accent">
              {record.decision.recommendations.map((r, i) => (
                <li key={r}>
                  {i + 1}. {r}
                </li>
              ))}
            </ol>
          </div>
        </div>

        {/* Limitations. */}
        {(evidence?.limitations.length ?? 0) > 0 && (
          <Callout tone="warning" title="Limitations">
            <ul className="space-y-1">
              {evidence?.limitations.map((l) => (
                <li key={l}>· {l}</li>
              ))}
            </ul>
          </Callout>
        )}

        <Callout tone="ethics" title="How to read this">
          This result is decision support, not proof. “No strong synthetic-speech evidence
          detected” does not mean the recording is genuine. For high-impact decisions, follow
          the recommendations above and verify independently.
        </Callout>
      </div>
    </Card>
  )
}
