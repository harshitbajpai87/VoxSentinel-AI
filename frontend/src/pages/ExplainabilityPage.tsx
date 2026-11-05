import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { AnalysisRecord } from '../lib/types'
import WindowScoresChart from '../components/audio/WindowScoresChart'
import {
  Button,
  Callout,
  Card,
  EmptyState,
  PageHeader,
  Tag,
} from '../components/ui'
import { SparkIcon } from '../components/icons'

const LAYERS = [
  {
    title: '1 · Measured features',
    text: 'Objective, reproducible numbers computed directly from the audio: duration, sample rate, channel count, signal level, quality warnings. These are facts about the input, not model opinions.',
    tone: 'border-accent/25 bg-accent/5',
  },
  {
    title: '2 · Model-derived explanation',
    text: 'What the model reacted to, using a technique technically supported by that model: feature importance for a classical model, or saliency-style attribution for a neural one. Labelled with the method used.',
    tone: 'border-magenta/25 bg-magenta/5',
  },
  {
    title: '3 · Policy decision',
    text: 'How the scores were turned into a risk category by documented thresholds, and what the recommended next action is. Auditable and configurable, not a model output.',
    tone: 'border-warning/25 bg-warning/5',
  },
]

export default function ExplainabilityPage() {
  const { status, refresh } = useApiStatus()
  const [latest, setLatest] = useState<AnalysisRecord | null>(null)
  const [failed, setFailed] = useState(false)
  const online = status === 'online'

  // Dynamic evidence: the most recent REAL analysis from the backend.
  useEffect(() => {
    if (!online) return
    api
      .listAnalyses({ limit: 1 })
      .then((res) => setLatest(res.analyses[0] ?? null))
      .catch(() => setFailed(true))
  }, [online])

  const windows = latest?.evidence?.time_window_scores ?? []

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Explainability & Evidence"
        subtitle="Every analysis result separates what was measured, what the model inferred, and what the policy decided. Nothing is presented as causal proof."
      />

      <div className="mb-6">
        {!online && (
          <EmptyState
            icon={<SparkIcon width={26} height={26} />}
            title="Backend offline"
            description="Explanations come from real analyses. Start the backend and analyze a clip to see its evidence here."
            action={<Button variant="secondary" onClick={() => void refresh()}>Retry connection</Button>}
          />
        )}
        {online && !latest && !failed && (
          <EmptyState
            icon={<SparkIcon width={26} height={26} />}
            title="No analyses yet"
            description="Analyze or record a clip first — its evidence layers (measured features, model explanation, window scores) will appear here, pulled live from the backend."
          />
        )}
        {failed && (
          <EmptyState
            icon={<SparkIcon width={26} height={26} />}
            title="Could not load the latest analysis"
            description="The backend is online but the history request failed. Retry in a moment."
          />
        )}
        {latest && (
          <Card
            title="Latest analysis evidence"
            subtitle={`${latest.input.display_name} · ${latest.model.name} v${latest.model.version}`}
            actions={<Tag>{latest.decision.category.replace(/_/g, ' ')}</Tag>}
          >
            {latest.evidence?.measured_features?.length ? (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {latest.evidence.measured_features.map((f) => (
                  <div key={f.name} className="rounded-md border border-line bg-surface-2 p-3">
                    <p className="text-[11px] uppercase tracking-wider text-ink-dim">{f.name}</p>
                    <p className="mt-1 text-sm font-semibold text-ink">
                      {f.value} {f.unit}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-ink-faint">No measured features recorded for this analysis.</p>
            )}
            {latest.evidence?.model_explanation && (
              <div className="mt-4">
                <Callout tone="info" title="Model-derived explanation">
                  {latest.evidence.model_explanation}
                </Callout>
              </div>
            )}
            {windows.length > 0 && (
              <div className="mt-5">
                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-dim">
                  Segment-level scores (evidence, not verdict)
                </p>
                <WindowScoresChart windows={windows} />
              </div>
            )}
            {latest.evidence?.limitations?.length ? (
              <div className="mt-4">
                <Callout tone="warning" title="Limitations">
                  {latest.evidence.limitations.join(' ')}
                </Callout>
              </div>
            ) : null}
          </Card>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {LAYERS.map((layer) => (
          <div key={layer.title} className={`rounded-lg border p-5 ${layer.tone}`}>
            <h2 className="text-sm font-semibold text-ink">{layer.title}</h2>
            <p className="mt-2 text-xs leading-relaxed text-ink-dim">{layer.text}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="What is shown per analysis">
          <ul className="space-y-2.5 text-xs leading-relaxed text-ink-dim">
            <li>Why the model raised (or did not raise) a concern.</li>
            <li>Input-quality indicators that could affect reliability.</li>
            <li>The acoustic evidence available for this clip.</li>
            <li>Model name, version and threshold values used.</li>
            <li>Uncertainty and its method, plus limitations.</li>
            <li>The recommended next action for the risk category.</li>
          </ul>
        </Card>

        <div className="space-y-4">
          <Callout tone="warning" title="What explanations are not">
            A waveform picture is not proof of cloning. An explanation describes which inputs
            influenced a model’s score — it is supporting evidence for a human decision, not a
            causal claim about how the audio was produced.
          </Callout>
          <Callout tone="ethics" title="Model-specific techniques only">
            The app only claims explanation techniques the selected model actually supports.
            Before the model is chosen (Phase 5), this page deliberately does not promise
            saliency maps or feature attributions.
          </Callout>
        </div>
      </div>
    </div>
  )
}
