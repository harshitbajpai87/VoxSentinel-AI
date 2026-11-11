import type { AnalysisRecord } from '../../lib/types'
import { Button, Card, Pill } from '../ui'
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

/** Telemetry values are shown at a fixed 5 dp so two results can be compared by eye. */
function fixed5(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(5) : 'not provided'
}

function MetricCell({
  label,
  value,
  hint,
  emphasis,
}: {
  label: string
  value: string
  hint?: string
  emphasis?: boolean
}) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-dim">{label}</p>
      <p
        className={`mt-1 font-mono text-ink ${emphasis ? 'text-3xl font-semibold' : 'text-lg'}`}
        title={value}
      >
        {value}
      </p>
      {hint ? <p className="mt-0.5 text-[11px] leading-snug text-ink-faint">{hint}</p> : null}
    </div>
  )
}

/**
 * Displays a REAL analysis result returned by the backend. Every field shown
 * comes from the response; nothing is simulated. Values the backend did not
 * provide render as "not provided" instead of a fabricated number.
 *
 * All telemetry (scores, probabilities, uncertainty, measured features and the
 * segment breakdown) sits inside one bordered container so the numbers read as
 * a single instrument panel rather than a stack of separate boxes.
 */
export function AnalysisResultPanel({
  record,
  showDecisionAdvice = true,
}: {
  record: AnalysisRecord
  /** Live recording hides the decision reasons and next steps; Analyze keeps them. */
  showDecisionAdvice?: boolean
}) {
  const tone = CATEGORY_TONE[record.decision.category] ?? 'neutral'
  const scores = record.scores
  const evidence = record.evidence
  const hasCalibrated =
    scores.calibrated_probability !== null && scores.calibrated_probability !== undefined
  const hasRaw = typeof scores.raw_score === 'number'

  // The headline number: the calibrated probability when it exists, otherwise
  // the raw heuristic score, clearly labelled. Never invent a number.
  const headline = hasCalibrated
    ? `${((scores.calibrated_probability as number) * 100).toFixed(1)}%`
    : hasRaw
      ? `${(Math.max(0, Math.min(1, scores.raw_score as number)) * 100).toFixed(1)}%`
      : 'not provided'
  const headlineHint = hasCalibrated
    ? 'Calibrated probability.'
    : hasRaw
      ? 'Detector score (heuristic screen, not a calibrated probability).'
      : 'The backend did not return a score for this clip.'

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

        {/* Single unified telemetry container. */}
        <div
          className="rounded-xl border border-line bg-surface-2/40 p-4"
          data-testid="analysis-telemetry"
        >
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCell
              label="Risk percentage"
              value={headline}
              hint={headlineHint}
              emphasis
            />
            <MetricCell
              label="Raw model score"
              value={hasRaw ? (scores.raw_score as number).toFixed(5) : fixed5(null)}
            />
            <MetricCell
              label="Calibrated probability"
              value={
                hasCalibrated
                  ? `${((scores.calibrated_probability as number) * 100).toFixed(1)}%`
                  : 'not provided'
              }
            />
            <MetricCell
              label="Uncertainty estimate"
              value={fixed5(scores.uncertainty)}
              hint={scores.uncertainty_method ?? undefined}
            />
          </div>

          {/* Facts about the audio. */}
          <div className="mt-4 border-t border-line/70 pt-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-dim">
              Input
            </p>
            <p className="mt-1 font-mono text-sm text-ink">
              {record.input.duration_s.toFixed(1)} s · {record.input.sample_rate_hz} Hz ·{' '}
              {record.input.channels} ch
            </p>
          </div>

          {evidence && evidence.measured_features.length > 0 && (
            <div className="mt-4 border-t border-line/70 pt-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-dim">
                Measured features
              </p>
              <dl className="mt-2 grid gap-x-6 gap-y-2 sm:grid-cols-2">
                {evidence.measured_features.map((f) => (
                  <div
                    key={f.name}
                    className="flex min-w-0 items-baseline justify-between gap-3 border-b border-line/50 pb-1.5"
                  >
                    <dt className="truncate text-sm text-ink-dim" title={f.name}>
                      {f.name}
                    </dt>
                    <dd className="shrink-0 font-mono text-sm text-ink">
                      {f.value} {f.unit}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {(evidence?.time_window_scores?.length ?? 0) > 0 && (
            <div className="mt-4 border-t border-line/70 pt-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-dim">
                Segment-level scores · evidence, not the verdict
              </p>
              <div className="mt-2">
                <WindowScoresChart windows={evidence?.time_window_scores ?? []} />
              </div>
            </div>
          )}
        </div>

        {/* Policy decision. Kept outside the telemetry container: these are
            actions, not measurements. */}
        {showDecisionAdvice && (
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
        )}
      </div>
    </Card>
  )
}
