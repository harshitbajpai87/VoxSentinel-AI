import type { WindowScore } from '../../lib/types'

/**
 * Segment-level scores from the SAME model, rendered as a proportional
 * timeline of bars. Evidence only: the verdict comes from the full-clip
 * analysis, not from these bars.
 */
export default function WindowScoresChart({ windows }: { windows: WindowScore[] }) {
  if (!windows.length) return null
  const maxScore = Math.max(...windows.map((w) => w.score))
  const scale = (v: number) => (maxScore > 0 ? v / maxScore : 0)

  return (
    <div className="space-y-2">
      <div className="flex items-end gap-1" aria-label="Segment-level scores">
        {windows.map((w, i) => {
          const pct = Math.round(scale(w.score) * 100)
          return (
            <div key={i} className="flex-1 min-w-0" title={`${w.start_s}s–${w.end_s}s: ${w.score.toFixed(3)}`}>
              <div
                className={`h-14 rounded-sm ${w.score >= 0.5 ? 'bg-danger/80' : 'bg-success/70'}`}
                style={{ width: `${Math.max(pct, 3)}%`, opacity: 0.85 }}
                role="img"
                aria-label={`Segment ${i + 1}: score ${w.score.toFixed(2)}`}
              />
            </div>
          )
        })}
      </div>
      <div className="flex justify-between text-xs text-ink-faint">
        <span>{windows[0].start_s.toFixed(0)}s</span>
        <span>
          {windows.length} segment{windows.length === 1 ? '' : 's'} · 4s windows · same model
        </span>
        <span>{windows[windows.length - 1].end_s.toFixed(0)}s</span>
      </div>
    </div>
  )
}
