import { useCallback, useEffect, useState } from 'react'
import { api, exportAnalysesCsv } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { AnalysisRecord, RiskCategory } from '../lib/types'
import {
  Button,
  Card,
  EmptyState,
  ErrorState,
  PageHeader,
  Pill,
  Tag,
} from '../components/ui'
import { DownloadIcon, HistoryIcon } from '../components/icons'

const CATEGORY_TONES: Record<string, 'ok' | 'warn' | 'err' | 'info' | 'neutral'> = {
  low_observed_risk: 'ok',
  elevated_risk: 'warn',
  high_risk: 'err',
  inconclusive: 'info',
  unable_to_analyze: 'neutral',
}

export default function HistoryPage() {
  const { status, refresh } = useApiStatus()
  const [analyses, setAnalyses] = useState<AnalysisRecord[] | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [category, setCategory] = useState<string>('')
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState(false)

  const load = useCallback(async (cat?: string) => {
    setLoadFailed(false)
    try {
      const res = await api.listAnalyses(cat ? { category: cat } : undefined)
      setAnalyses(res.analyses)
    } catch {
      setLoadFailed(true)
      setAnalyses([])
    }
  }, [])

  useEffect(() => {
    if (status === 'online') void load(category)
    else setAnalyses(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, category])

  const online = status === 'online'
  const empty = analyses !== null && analyses.length === 0

  /** Downloads the backend-generated CSV (DB-backed, formula-injection safe). */
  const exportCsv = async () => {
    setExporting(true)
    setExportError(false)
    try {
      const blob = await exportAnalysesCsv()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `voxsentinel-history-${new Date().toISOString().slice(0, 10)}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      setExportError(true)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Analysis History"
        subtitle="Chronological log of past audio analyses. Filter, sort, or export records."
        actions={
          <Button
            variant="secondary"
            disabled={!online || exporting || empty}
            onClick={() => void exportCsv()}
            title={online ? 'Download the authorized history as CSV' : 'Backend offline'}
          >
            <DownloadIcon width={14} height={14} /> {exporting ? 'Exporting…' : 'Export CSV'}
          </Button>
        }
      />

      <Card title="Filters">
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            disabled={!online}
            aria-label="Filter by risk category"
            className="rounded-md border border-line bg-surface-2 px-3 py-1.5 text-sm text-ink"
          >
            <option value="">All risk categories</option>
            <option value="low_observed_risk">Low observed risk</option>
            <option value="elevated_risk">Elevated risk</option>
            <option value="high_risk">High risk</option>
            <option value="inconclusive">Inconclusive</option>
            <option value="unable_to_analyze">Unable to analyze</option>
          </select>
          <span className="text-xs text-ink-faint">
            {analyses !== null ? `${analyses.length} shown · sorted newest first` : ''}
          </span>
        </div>
      </Card>
      {exportError && (
        <p className="mt-2 text-xs text-danger">The export failed. The backend may be offline.</p>
      )}

      <div className="mt-6">
        {!online && (
          <ErrorState
            description="Backend connection offline. Start the service to load recorded analysis history."
            onRetry={() => void refresh()}
          />
        )}
        {online && loadFailed && <ErrorState title="Could not load history" onRetry={() => void load()} />}
        {online && !loadFailed && analyses === null && (
          <div className="py-10 text-center text-sm text-ink-dim">Loading history…</div>
        )}
        {empty && (
          <EmptyState
            icon={<HistoryIcon width={26} height={26} />}
            title="No analyses recorded yet"
            description="Analyze a clip first — the record (ID, timestamp, model version, outcome, risk category) will appear here automatically."
          />
        )}
        {online && analyses !== null && analyses.length > 0 && (
          <div className="overflow-x-auto rounded-lg border border-line bg-surface">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs uppercase tracking-wider text-ink-dim">
                  <th className="px-4 py-3 font-medium">ID</th>
                  <th className="px-4 py-3 font-medium">Timestamp</th>
                  <th className="px-4 py-3 font-medium">File</th>
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">Outcome</th>
                  <th className="px-4 py-3 font-medium">Risk</th>
                  <th className="px-4 py-3 font-medium">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/60">
                {analyses.map((a) => (
                  <tr key={a.id} className="transition-colors hover:bg-surface-2/50">
                    <td className="px-4 py-3 font-mono text-xs text-ink-dim">
                      {a.id.slice(0, 8)}…
                    </td>
                    <td className="px-4 py-3 text-xs text-ink-dim">
                      {new Date(a.created_at).toLocaleString()}
                    </td>
                    <td className="max-w-[10rem] truncate px-4 py-3 text-ink">
                      {a.input.display_name}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-dim">
                      {a.model.name} v{a.model.version}
                    </td>
                    <td className="px-4 py-3 text-xs text-ink-dim">{a.decision.label}</td>
                    <td className="px-4 py-3">
                      <Pill tone={CATEGORY_TONES[a.decision.category] ?? 'neutral'}>
                        {a.decision.category}
                      </Pill>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-dim">
                      {a.latency_ms} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
