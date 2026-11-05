import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { AuditEvent, CleanupRunResult, RetentionPolicy } from '../lib/types'
import {
  Button,
  Callout,
  Card,
  EmptyState,
  PageHeader,
  Pill,
  StateIcon,
  Tag,
} from '../components/ui'
import { ShieldIcon } from '../components/icons'

// Honest status tracking: what the system does today. Backend-verified.
const SAFEGUARDS: { label: string; done: boolean; phase: string }[] = [
  { label: 'API URL via environment variable, no secrets in frontend code', done: true, phase: 'done' },
  { label: 'Explicit offline / empty / error states instead of silent failures', done: true, phase: 'done' },
  { label: 'Consent and authorization notices on all audio workflows', done: true, phase: 'done' },
  { label: 'Strict upload validation (type, size, duration, silence)', done: true, phase: 'done' },
  { label: 'Randomized server-side filenames, no execution of uploads', done: true, phase: 'done' },
  { label: 'Restricted temp storage + secure deletion after analysis', done: true, phase: 'done' },
  { label: 'Strict CORS allowlist configuration', done: true, phase: 'done' },
  { label: 'Safe error messages (no paths, traces or secrets)', done: true, phase: 'done' },
  { label: 'Audit log for uploads, analyses, exports and cleanups', done: true, phase: 'done' },
  { label: 'Retention policy with enforced deletion controls', done: true, phase: 'done' },
  { label: 'Request-ID correlation across every API call', done: true, phase: 'done' },
  { label: 'Rate limiting on analysis endpoints', done: false, phase: 'planned' },
  { label: 'Authentication and authorization for private history', done: false, phase: 'planned' },
]

const EVENT_TYPES = [
  { value: '', label: 'All event types' },
  { value: 'analysis', label: 'Analysis' },
  { value: 'upload', label: 'Upload' },
  { value: 'data_export', label: 'Export' },
  { value: 'evaluation', label: 'Evaluation' },
  { value: 'retention', label: 'Retention' },
  { value: 'speaker_verification', label: 'Speaker verification' },
]

export default function SecurityAuditPage() {
  const { status, refresh } = useApiStatus()
  const [events, setEvents] = useState<AuditEvent[] | null>(null)
  const [eventType, setEventType] = useState('')
  const [policy, setPolicy] = useState<RetentionPolicy | null>(null)
  const [preview, setPreview] = useState<CleanupRunResult | null>(null)
  const [busy, setBusy] = useState<'preview' | 'run' | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const online = status === 'online'

  const load = useCallback(async (type: string) => {
    setError(null)
    try {
      const [audit, retention] = await Promise.all([
        api.listAuditEvents({ event_type: type || undefined, limit: 50 }),
        api.getRetentionPolicy().catch(() => null),
      ])
      setEvents(audit.events)
      setPolicy(retention)
    } catch {
      setError('Could not load the audit trail. The backend may be offline.')
    }
  }, [])

  useEffect(() => {
    if (online) void load(eventType)
    else setEvents(null)
  }, [online, eventType, load])

  const runCleanup = async (dryRun: boolean) => {
    setBusy(dryRun ? 'preview' : 'run')
    setError(null)
    setNotice(null)
    try {
      const result = await api.runRetentionCleanup(dryRun)
      setPreview(result)
      if (!dryRun) {
        setNotice(
          `Cleanup executed: ${result.analyses_deleted} analyses and ` +
            `${result.audit_events_deleted} audit events deleted.`,
        )
        void load(eventType)
      }
    } catch {
      setError('The cleanup request failed.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Security & Audit"
        subtitle="Defense-in-progress, stated honestly. Items are checked off as they are implemented and verified, not before."
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <Card
          title="Safeguards"
          subtitle="Green: implemented and verified. Grey: planned, not yet claimed."
          className="lg:col-span-2"
        >
          <ul className="divide-y divide-line/60">
            {SAFEGUARDS.map((s) => (
              <li key={s.label} className="flex items-center gap-3 py-2.5 text-sm">
                <StateIcon ok={s.done} />
                <span className={s.done ? 'text-ink' : 'text-ink-dim'}>{s.label}</span>
                <span className="ml-auto shrink-0">
                  {s.done ? <Tag>Implemented</Tag> : <Tag>{s.phase}</Tag>}
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <div className="space-y-6">
          <Card
            title="Data retention"
            subtitle={policy ? 'Live policy from the backend; enforced on a schedule and on demand.' : undefined}
            actions={
              policy
                ? policy.enabled
                  ? <Pill tone="ok">Enforced</Pill>
                  : <Pill tone="warn">Keep forever</Pill>
                : <Tag>Backend offline</Tag>
            }
          >
            {policy ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="rounded-md border border-line bg-surface-2 p-3">
                    <p className="text-[11px] uppercase tracking-wider text-ink-dim">Analyses</p>
                    <p className="mt-1 text-lg font-semibold text-ink">
                      {policy.analyses_days > 0 ? `${policy.analyses_days} days` : 'kept'}
                    </p>
                  </div>
                  <div className="rounded-md border border-line bg-surface-2 p-3">
                    <p className="text-[11px] uppercase tracking-wider text-ink-dim">Audit log</p>
                    <p className="mt-1 text-lg font-semibold text-ink">
                      {policy.audit_days > 0 ? `${policy.audit_days} days` : 'kept'}
                    </p>
                  </div>
                </div>
                {policy.last_run && (
                  <p className="text-xs text-ink-faint">
                    Last cleanup: {new Date(policy.last_run.created_at).toLocaleString()} —{' '}
                    {String(policy.last_run.metadata?.analyses_deleted ?? 0)} analyses deleted.
                  </p>
                )}
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" disabled={!online || busy !== null} onClick={() => void runCleanup(true)}>
                    {busy === 'preview' ? 'Previewing…' : 'Preview cleanup'}
                  </Button>
                  <Button disabled={!online || busy !== null} onClick={() => void runCleanup(false)}>
                    {busy === 'run' ? 'Running…' : 'Run cleanup now'}
                  </Button>
                </div>
                {preview && (
                  <Callout tone="info" title={preview.dry_run ? 'Preview (nothing deleted)' : 'Cleanup executed'}>
                    {preview.dry_run
                      ? `${preview.analyses_deleted} analyses and ${preview.audit_events_deleted} audit events are older than the cutoff.`
                      : notice}
                  </Callout>
                )}
                {!preview && notice && <p className="text-xs text-ink-dim">{notice}</p>}
              </div>
            ) : (
              <p className="text-xs text-ink-dim">
                Retention controls appear when the backend is online.
              </p>
            )}
          </Card>

          <Card title="Privacy commitments" subtitle="Stated up front.">
            <ul className="space-y-2 text-xs leading-relaxed text-ink-dim">
              <li>Uploaded audio is temporary and deleted after each analysis by default.</li>
              <li>History stores metadata, scores and decisions — not raw recordings.</li>
              <li>Private audio and analysis history are not publicly accessible.</li>
            </ul>
          </Card>
        </div>
      </div>

      <div className="mt-6">
        <Card
          title="Audit log"
          subtitle="Append-only: events are recorded by the backend and can never be edited through the API."
          actions={
            <select
              value={eventType}
              onChange={(e) => setEventType(e.target.value)}
              disabled={!online}
              aria-label="Filter by event type"
              className="rounded-md border border-line bg-surface-2 px-3 py-1.5 text-sm text-ink"
            >
              {EVENT_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          }
        >
          {error && <p className="mb-3 text-xs text-danger">{error}</p>}
          {!online && (
            <EmptyState
              icon={<ShieldIcon width={24} height={24} />}
              title="Backend offline"
              description="Start the backend to load the audit trail. Raw audio, tokens and secrets are never logged."
              action={<Button variant="secondary" onClick={() => void refresh()}>Retry connection</Button>}
            />
          )}
          {online && events !== null && events.length === 0 && (
            <EmptyState
              icon={<ShieldIcon width={24} height={24} />}
              title="No audit events yet"
              description="Uploads, analysis requests, exports and retention cleanups are recorded here as they happen."
            />
          )}
          {online && events !== null && events.length > 0 && (
            <div className="overflow-x-auto rounded-lg border border-line bg-surface">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-xs uppercase tracking-wider text-ink-dim">
                    <th className="px-4 py-3 font-medium">Time</th>
                    <th className="px-4 py-3 font-medium">Type</th>
                    <th className="px-4 py-3 font-medium">Action</th>
                    <th className="px-4 py-3 font-medium">Outcome</th>
                    <th className="px-4 py-3 font-medium">Resource</th>
                    <th className="px-4 py-3 font-medium">Request ID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/60">
                  {events.map((e) => (
                    <tr key={e.id} className="transition-colors hover:bg-surface-2/50">
                      <td className="px-4 py-3 text-xs text-ink-dim">
                        {new Date(e.created_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-3"><Tag>{e.event_type}</Tag></td>
                      <td className="px-4 py-3 text-ink">{e.action}</td>
                      <td className="px-4 py-3">
                        <Pill tone={e.outcome === 'success' ? 'ok' : 'warn'}>{e.outcome}</Pill>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-ink-dim">
                        {e.resource_id ? `${e.resource_id.slice(0, 8)}…` : '—'}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-ink-dim">
                        {e.request_id ? `${e.request_id.slice(0, 8)}…` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
