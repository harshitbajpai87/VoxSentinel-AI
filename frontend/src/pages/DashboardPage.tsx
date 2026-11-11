import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { AnalysisRecord, RiskCategory } from '../lib/types'
import { Waveform } from '../components/Waveform'
import {
  Card,
  EmptyState,
  ErrorState,
  LinkButton,
  PageHeader,
  Pill,
  StatTile,
  StatusDot,
  Tag,
} from '../components/ui'
import { PlusIcon, WaveformIcon } from '../components/icons'
import { StaggerGroup, StaggerItem } from '../components/motion/Reveal'

const CATEGORY_META: Record<RiskCategory, { label: string; tone: 'ok' | 'warn' | 'err' | 'info' }> = {
  low_observed_risk: { label: 'Low observed risk', tone: 'ok' },
  elevated_risk: { label: 'Elevated risk', tone: 'warn' },
  high_risk: { label: 'High risk', tone: 'err' },
  inconclusive: { label: 'Inconclusive', tone: 'info' },
}

export default function DashboardPage() {
  const { status, health, refresh } = useApiStatus()
  const [analyses, setAnalyses] = useState<AnalysisRecord[] | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)

  const load = useCallback(async () => {
    setLoadFailed(false)
    try {
      const res = await api.listAnalyses()
      setAnalyses(res.analyses)
    } catch {
      setLoadFailed(true)
      setAnalyses([])
    }
  }, [])

  useEffect(() => {
    if (status === 'online') void load()
    else setAnalyses(null)
  }, [status, load])

  const count = (cat: RiskCategory) =>
    analyses ? analyses.filter((a) => a.decision.category === cat).length : null

  const offline = status === 'offline'
  const hasData = analyses !== null && analyses.length > 0
  const latest = analyses && analyses.length ? analyses[0] : null
  const latestRisk = latest?.scores.calibrated_probability ?? null

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Overview"
        subtitle="System telemetry and analysis logs."
        actions={
          <LinkButton href="/analyze">
            <PlusIcon width={15} height={15} /> Analyze audio
          </LinkButton>
        }
      />

      {/* Stat tiles */}
      <StaggerGroup className="grid grid-cols-2 gap-4 md:grid-cols-6">
        <StaggerItem>
        <StatTile
          label="Total analyses"
          value={analyses === null ? '—' : analyses.length}
          hint={analyses === null ? (offline ? 'Backend offline' : 'Connecting…') : undefined}
        />
        </StaggerItem>
        <StaggerItem>
        <StatTile
          label="High risk"
          value={count('high_risk') ?? '—'}
          tone="err"
          hint={count('high_risk') === 0 ? 'None recorded' : undefined}
        />
        </StaggerItem>
        <StaggerItem>
        <StatTile label="Elevated risk" value={count('elevated_risk') ?? '—'} tone="warn" />
        </StaggerItem>
        <StaggerItem>
        <StatTile label="Low observed risk" value={count('low_observed_risk') ?? '—'} tone="ok" />
        </StaggerItem>
        <StaggerItem>
        <StatTile label="Inconclusive" value={count('inconclusive') ?? '—'} tone="info" />
        </StaggerItem>
        <StaggerItem>
        <StatTile
          label="Latest risk"
          value={latestRisk !== null ? `${(latestRisk * 100).toFixed(1)}%` : '—'}
          hint={latest ? 'Calibrated probability, most recent analysis' : 'No analyses yet'}
          tone={latestRisk === null ? 'neutral' : latestRisk >= 0.66 ? 'err' : latestRisk >= 0.33 ? 'warn' : 'ok'}
        />
        </StaggerItem>
      </StaggerGroup>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        {/* Left: recent analyses */}
        <div className="space-y-6 lg:col-span-2">
          <Card
            title="Recent analyses"
            subtitle="Chronological history of processed audio clips."
          >
            {offline && (
              <ErrorState
                description="Start the Python backend, then re-check the connection."
                onRetry={() => void refresh()}
              />
            )}
            {!offline && loadFailed && (
              <ErrorState
                title="Could not load analyses"
                description="The backend is reachable but the analyses request failed."
                onRetry={() => void load()}
              />
            )}
            {!offline && !loadFailed && analyses === null && (
              <div className="py-10 text-center text-sm text-ink-dim">Loading analyses…</div>
            )}
            {!offline && !loadFailed && analyses?.length === 0 && (
              <EmptyState
                icon={<WaveformIcon width={26} height={26} />}
                title="No analyses yet"
                description="Analyze an audio clip to start a history of recordings and their reported risk signals."
                action={
                  <LinkButton href="/analyze">
                    <PlusIcon width={15} height={15} /> Analyze your first clip
                  </LinkButton>
                }
              />
            )}
            {hasData && (
              <ul className="divide-y divide-line/60">
                {analyses.slice(0, 6).map((a) => {
                  const meta =
                    a.decision.category in CATEGORY_META
                      ? CATEGORY_META[a.decision.category as RiskCategory]
                      : null
                  return (
                    <li key={a.id} className="flex items-center gap-3 py-2.5 text-sm">
                      <span className="flex-1 truncate text-ink">{a.input.display_name}</span>
                      <span className="hidden shrink-0 font-mono text-xs text-ink-faint sm:block">
                        {new Date(a.created_at).toLocaleString()}
                      </span>
                      {a.scores.calibrated_probability !== null &&
                        a.scores.calibrated_probability !== undefined && (
                          <span className="shrink-0 font-mono text-xs text-ink-dim">
                            {(a.scores.calibrated_probability * 100).toFixed(1)}%
                          </span>
                        )}
                      {meta ? (
                        <Pill tone={meta.tone}>{meta.label}</Pill>
                      ) : (
                        <Pill tone="neutral">{a.decision.category}</Pill>
                      )}
                    </li>
                  )
                })}
              </ul>
            )}
          </Card>

          <Card
            title="Risk distribution"
            subtitle="A chart appears here once real analyses exist."
            actions={<Tag>Recorded analyses</Tag>}
          >
            <Waveform active={false} className="justify-center opacity-30" />
            <p className="mt-3 text-center text-xs text-ink-faint">
              No chart is available yet. Review individual results in Analysis History.
            </p>
          </Card>
        </div>

        {/* Right: system status */}
        <div className="space-y-6">
          <Card title="Pipeline status">
            <ul className="space-y-3 text-sm">
              <li className="flex items-center justify-between">
                <span className="text-ink-dim">Backend</span>
                {status === 'online' ? (
                  <Pill tone="ok">
                    <StatusDot tone="ok" /> Connected
                  </Pill>
                ) : status === 'checking' ? (
                  <Pill tone="info">
                    <StatusDot tone="info" pulse /> Checking
                  </Pill>
                ) : (
                  <Pill tone="err">
                    <StatusDot tone="err" /> Offline
                  </Pill>
                )}
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-dim">Detection model</span>
                <Pill tone={health?.model_ready ? 'ok' : 'neutral'}>
                  {health ? (health.model_ready ? 'Ready' : 'Not ready') : 'Unknown'}
                </Pill>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-dim">Speaker model</span>
                <Pill tone={health?.speaker_model_ready ? 'ok' : 'neutral'}>
                  {health ? (health.speaker_model_ready ? 'Ready' : 'Not ready') : 'Unknown'}
                </Pill>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-dim">API version</span>
                <span className="font-mono text-xs text-ink">{health?.version ?? '—'}</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-ink-dim">Mode</span>
                <Pill tone={health?.mode === 'production' ? 'ok' : 'info'}>
                  {health?.mode ?? '—'}
                </Pill>
              </li>
            </ul>
          </Card>

          <Card title="Recent audit events">
            <EmptyState
              title="Audit log not connected yet"
              description="Audit events are recorded by the backend but are not listed in this overview."
            />
          </Card>
        </div>
      </div>
    </div>
  )
}
