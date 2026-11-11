import type { RiskBadge } from '../../lib/unified'
import { TONE_CLASS } from '../../lib/unified'

/**
 * The live state strip: overall risk, voice authenticity, speaker match, scam
 * intent, which ASR engine ran and which reasoning model answered.
 *
 * Every badge comes from the backend with its own label and detail, so a model
 * that did not run reads "inconclusive"/"not verified" instead of pretending.
 */
export function RiskBadgeRow({ badges, compact = false }: { badges: RiskBadge[]; compact?: boolean }) {
  if (!badges.length) return null
  return (
    <ul className="flex flex-wrap gap-2" aria-label="Analysis badges" data-testid="risk-badges">
      {badges.map((badge) => (
        <li key={badge.id}>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${TONE_CLASS[badge.tone] ?? TONE_CLASS.info}`}
            title={badge.detail ?? undefined}
            data-testid={`badge-${badge.id}`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
            {badge.label}
            {!compact && typeof badge.value === 'number' && badge.id !== 'scam_intent' ? (
              <span className="tabular-nums opacity-70">{badge.value.toFixed(2)}</span>
            ) : null}
          </span>
        </li>
      ))}
    </ul>
  )
}

export default RiskBadgeRow
