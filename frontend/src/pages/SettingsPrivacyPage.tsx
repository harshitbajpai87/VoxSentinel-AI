import { useEffect, useState } from 'react'
import {
  Button,
  Callout,
  Card,
  PageHeader,
  Pill,
} from '../components/ui'
import { CheckIcon, RefreshIcon } from '../components/icons'
import { getConfiguredBaseUrl, setConfiguredBaseUrl } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'

export default function SettingsPrivacyPage() {
  const { status, refresh } = useApiStatus()
  const [urlInput, setUrlInput] = useState(getConfiguredBaseUrl())
  const [saved, setSaved] = useState(false)

  // Keep the field in sync if the effective URL changes elsewhere.
  useEffect(() => {
    setUrlInput(getConfiguredBaseUrl())
  }, [])

  const save = async () => {
    setConfiguredBaseUrl(urlInput)
    setSaved(true)
    await refresh()
    window.setTimeout(() => setSaved(false), 2500)
  }

  const reset = async () => {
    setConfiguredBaseUrl(null)
    setUrlInput(getConfiguredBaseUrl())
    await refresh()
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Settings & Privacy"
        subtitle="Connection settings and the privacy commitments the app makes to you and your users."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Backend connection"
          subtitle="Where the frontend sends analysis requests. Stored locally in your browser; overrides the build-time VITE_API_BASE_URL."
          actions={
            status === 'online' ? <Pill tone="ok">Connected</Pill> : <Pill tone="err">Offline</Pill>
          }
        >
          <label className="block text-xs font-medium uppercase tracking-wider text-ink-dim">
            API base URL
          </label>
          <input
            type="url"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            spellCheck={false}
            className="mt-2 w-full rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-sm text-ink placeholder:text-ink-faint focus:border-accent/50"
            placeholder="http://localhost:8000"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={() => void save()}>
              {saved ? <CheckIcon width={14} height={14} /> : null}
              {saved ? 'Saved' : 'Save & test connection'}
            </Button>
            <Button variant="secondary" onClick={() => void refresh()}>
              <RefreshIcon width={14} height={14} /> Test connection
            </Button>
            <Button variant="ghost" onClick={() => void reset()}>
              Reset to .env value
            </Button>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-ink-faint">
            The test calls GET /health with a 6-second timeout. A failure here means the
            backend is not running, unreachable, or blocking CORS — the frontend shows real
            states, never fabricated data.
          </p>
        </Card>

        <div className="space-y-6">
          <Card title="Consent notice" subtitle="Shown wherever audio is processed.">
            <p className="text-xs leading-relaxed text-ink-dim">
              Recordings are processed only for the requested analysis, are not used for any
              other purpose, and are deleted from temporary storage afterwards. Analysis
              metadata (without the raw audio) is kept in history so results stay auditable.
              You must confirm you are authorized to analyze every recording you submit.
            </p>
          </Card>

          <Card title="Data minimization">
            <ul className="space-y-2 text-xs leading-relaxed text-ink-dim">
              <li>No accounts or personal data required in this research build.</li>
              <li>No analytics, tracking or third-party scripts in the frontend.</li>
              <li>API keys and secrets live in environment variables, never in code.</li>
              <li>Private audio and history are not publicly accessible by default.</li>
            </ul>
          </Card>

          <Callout tone="ethics" title="Retention & deletion">
            Temporary audio is deleted after each analysis. History and audit records are
            purged on a schedule and on demand by the backend — preview or execute a cleanup
            from the Security &amp; Audit page. Live recordings store only what you allow:
            provisional window scores are never persisted, and the final analysis joins
            history only when you leave "Save to History" checked.
          </Callout>
        </div>
      </div>
    </div>
  )
}
