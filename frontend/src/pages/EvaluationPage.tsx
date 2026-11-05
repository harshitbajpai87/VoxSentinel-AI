import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { EvaluationJob, EvaluationSummary } from '../lib/types'
import {
  Button,
  Callout,
  Card,
  EmptyState,
  PageHeader,
  Pill,
  Tag,
} from '../components/ui'
import { ChartIcon, SparkIcon } from '../components/icons'

/** Metrics keys the backend's real evaluation report serves. */
interface JobMetrics {
  n?: number
  accuracy?: number
  f1_synthetic?: number
  precision_synthetic?: number
  recall_synthetic?: number
  roc_auc?: number
  eer?: number
  average_precision?: number
  false_positive_rate?: number
  false_negative_rate?: number
  specificity_real?: number
  threshold?: number
  positive_class?: string
  score_semantics?: string
  confusion_matrix?: Record<string, number>
  class_distribution?: Record<string, number>
  roc_curve?: { fpr: number; tpr: number }[]
  pr_curve?: { recall: number; precision: number }[]
}

const fmt = (v: number | undefined, digits = 4) =>
  v === undefined || v === null ? '—' : v.toFixed(digits)

/** Renders a real ROC curve as an SVG polyline (rate points from the report). */
function RocCurve({ points }: { points: { fpr: number; tpr: number }[] }) {
  if (points.length < 2) return null
  const path = points.map((p) => `${p.fpr * 100},${100 - p.tpr * 100}`).join(' ')
  return (
    <svg viewBox="0 0 100 100" className="h-56 w-56" role="img" aria-label="ROC curve">
      <rect x="0" y="0" width="100" height="100" fill="none" stroke="currentColor" strokeOpacity="0.2" />
      <line x1="0" y1="100" x2="100" y2="0" stroke="currentColor" strokeOpacity="0.25" strokeDasharray="3,3" />
      <polyline points={path} fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

function PrCurve({ points }: { points: { recall: number; precision: number }[] }) {
  if (points.length < 2) return null
  const path = points.map((p) => `${p.recall * 100},${100 - p.precision * 100}`).join(' ')
  return (
    <svg viewBox="0 0 100 100" className="h-56 w-56" role="img" aria-label="Precision-recall curve">
      <rect x="0" y="0" width="100" height="100" fill="none" stroke="currentColor" strokeOpacity="0.2" />
      <polyline points={path} fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

export default function EvaluationPage() {
  const { status, refresh } = useApiStatus()
  const [summary, setSummary] = useState<EvaluationSummary | null>(null)
  const [job, setJob] = useState<EvaluationJob | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const loadSummary = useCallback(async () => {
    try {
      setSummary(await api.getEvaluationSummary())
    } catch {
      setSummary(null)
    }
  }, [])

  useEffect(() => {
    if (status === 'online') void loadSummary()
    return stopPolling
  }, [status, loadSummary])

  /** Starts a REAL inference run over the configured eval partition. */
  const runEvaluation = async () => {
    setError(null)
    setRunning(true)
    try {
      const started = await api.startEvaluationRun()
      setJob(started)
      stopPolling()
      pollRef.current = window.setInterval(async () => {
        try {
          const state = await api.getEvaluationRun(started.job_id)
          setJob(state)
          if (state.status === 'completed' || state.status === 'failed') {
            stopPolling()
            setRunning(false)
            if (state.status === 'failed') setError(state.error ?? 'The evaluation run failed.')
            else void loadSummary()
            void refresh()
          }
        } catch {
          stopPolling()
          setRunning(false)
          setError('Lost contact with the evaluation job.')
        }
      }, 1000)
    } catch (e) {
      setRunning(false)
      setError(e instanceof Error ? e.message : 'Could not start the evaluation.')
    }
  }

  const online = status === 'online'
  const metrics = (job?.status === 'completed' ? (job.metrics as JobMetrics | null) : (summary?.metrics as JobMetrics | undefined)) ?? null
  const source = job?.status === 'completed' ? job.job_id : 'persisted report'

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Model Evaluation"
        subtitle="Measured performance of the detection model over the held-out evaluation partition. Metrics come from real inference runs — never from mock data."
        actions={
          running ? (
            <Pill tone="info">Running… {job?.progress.done ?? 0}/{job?.progress.total ?? '?'}</Pill>
          ) : metrics ? (
            <Pill tone="ok">Evaluated</Pill>
          ) : (
            <Tag>No run yet</Tag>
          )
        }
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card
            title="Run an evaluation"
            subtitle="Scores every clip in the configured eval partition with the loaded model, then computes the metrics below."
          >
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={() => void runEvaluation()} disabled={!online || running}>
                <ChartIcon width={14} height={14} /> {running ? 'Evaluating…' : 'Run evaluation'}
              </Button>
              <span className="text-xs text-ink-faint">
                {online
                  ? 'Runs server-side inference; progress polls the job endpoint.'
                  : 'Backend offline — start it to run an evaluation.'}
              </span>
            </div>
            {running && job && (
              <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full bg-accent transition-all"
                  style={{
                    width: job.progress.total > 0 ? `${(job.progress.done / job.progress.total) * 100}%` : '0%',
                  }}
                />
              </div>
            )}
            {error && (
              <div className="mt-4">
                <Callout tone="warning" title="Evaluation problem">{error}</Callout>
              </div>
            )}
          </Card>

          {!metrics && !running && (
            <EmptyState
              icon={<ChartIcon width={26} height={26} />}
              title="No evaluation results yet"
              description={
                summary && summary.evaluated === false
                  ? summary.message
                  : 'Run an evaluation to measure accuracy, ROC-AUC, EER and more over the held-out partition.'
              }
            />
          )}

          {metrics && (
            <Card
              title="Latest measured metrics"
              subtitle={`Source: ${source}`}
              actions={<Tag>{metrics.positive_class ? `positive class: ${metrics.positive_class}` : ''}</Tag>}
            >
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  { label: 'Accuracy', value: fmt(metrics.accuracy) },
                  { label: 'ROC-AUC', value: fmt(metrics.roc_auc) },
                  { label: 'EER', value: fmt(metrics.eer) },
                  { label: 'F1 (synthetic)', value: fmt(metrics.f1_synthetic) },
                  { label: 'Precision', value: fmt(metrics.precision_synthetic) },
                  { label: 'Recall', value: fmt(metrics.recall_synthetic) },
                  { label: 'FPR', value: fmt(metrics.false_positive_rate) },
                  { label: 'FNR', value: fmt(metrics.false_negative_rate) },
                  { label: 'Samples (n)', value: metrics.n ?? '—' },
                  { label: 'Threshold', value: fmt(metrics.threshold, 2) },
                  { label: 'Avg. precision', value: fmt(metrics.average_precision) },
                  { label: 'Specificity (real)', value: fmt(metrics.specificity_real) },
                ].map((m) => (
                  <div key={m.label} className="rounded-md border border-line bg-surface-2 p-3">
                    <p className="text-[11px] uppercase tracking-wider text-ink-dim">{m.label}</p>
                    <p className="mt-1 text-lg font-semibold text-ink">{m.value}</p>
                  </div>
                ))}
              </div>

              {metrics.confusion_matrix && (
                <div className="mt-6">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-dim">
                    Confusion matrix
                  </p>
                  <table className="text-sm">
                    <tbody>
                      {(['tn', 'fp'] as const).map((k) => (
                        <tr key={k}>
                          <td className="px-3 py-1 text-xs uppercase text-ink-faint">{k}</td>
                          <td className="px-3 py-1 font-mono text-ink">{metrics.confusion_matrix?.[k] ?? '—'}</td>
                        </tr>
                      ))}
                      {(['fn', 'tp'] as const).map((k) => (
                        <tr key={k}>
                          <td className="px-3 py-1 text-xs uppercase text-ink-faint">{k}</td>
                          <td className="px-3 py-1 font-mono text-ink">{metrics.confusion_matrix?.[k] ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {(metrics.roc_curve || metrics.pr_curve) && (
                <div className="mt-6 flex flex-wrap gap-8">
                  {metrics.roc_curve && (
                    <div>
                      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-dim">ROC curve</p>
                      <RocCurve points={metrics.roc_curve} />
                    </div>
                  )}
                  {metrics.pr_curve && (
                    <div>
                      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-dim">
                        Precision–recall
                      </p>
                      <PrCurve points={metrics.pr_curve} />
                    </div>
                  )}
                </div>
              )}

              {metrics.score_semantics && (
                <p className="mt-6 text-xs leading-relaxed text-ink-dim">{metrics.score_semantics}</p>
              )}
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Callout tone="ethics" title="How to read these numbers">
            <p className="space-y-2 leading-relaxed">
              The metrics describe this model on <em>its own held-out partition</em>: real
              LibriSpeech speech vs. Piper TTS synthesis, evaluated with speakers and a voice
              unseen in training. They are NOT a claim about other synthesis engines or
              real-world attack traffic.
            </p>
          </Callout>
          {summary?.evaluated && summary.evaluated_at && (
            <Card title="Persisted report">
              <p className="text-xs text-ink-dim">
                Last recorded run: {new Date(summary.evaluated_at).toLocaleString()}
              </p>
            </Card>
          )}
          <Card title="Scope reminder">
            <p className="flex items-start gap-2 text-xs leading-relaxed text-ink-dim">
              <SparkIcon width={14} height={14} className="mt-0.5 shrink-0 text-accent" />
              Scores rise with evidence of Piper-TTS synthesis. A high score is a signal for
              human review, not proof of forgery; a low score is not proof of authenticity.
            </p>
          </Card>
        </div>
      </div>
    </div>
  )
}
