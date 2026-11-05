import { NavLink } from 'react-router-dom'
import {
  DashboardIcon,
  FingerprintIcon,
  HistoryIcon,
  ShieldIcon,
  WaveformIcon,
} from '../icons'
import { StatusDot, Tag } from '../ui'
import { useApiStatus } from '../../lib/apiStatus'

export const NAV_ITEMS = [
  { to: '/dashboard', label: 'Overview', icon: DashboardIcon },
  { to: '/analyze', label: 'Analyze Audio', icon: WaveformIcon },
  { to: '/record', label: 'Live Recording', icon: WaveformIcon },
  { to: '/verify', label: 'Speaker Verification', icon: FingerprintIcon },
  { to: '/history', label: 'Analysis History', icon: HistoryIcon },
  { to: '/protection', label: 'Family Emergency & Scam Protection', icon: ShieldIcon },
]

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { status } = useApiStatus()

  return (
    <div className="flex h-full w-64 flex-col border-r border-line bg-surface/60">
      {/* Brand */}
      <div className="flex items-center gap-3 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-accent/30 bg-accent/10">
          <ShieldIcon width={18} height={18} className="text-accent" />
        </div>
        <div>
          <p className="text-sm font-semibold leading-tight text-ink">
            VoxSentinel <span className="text-accent">AI</span>
          </p>
          <div className="mt-0.5 flex items-center gap-1.5">
            <Tag>SIH26104</Tag>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-2" aria-label="Main navigation">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onNavigate}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors duration-150 ${
                isActive
                  ? 'bg-accent/10 text-accent'
                  : 'text-ink-dim hover:bg-surface-2 hover:text-ink'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={`w-0.5 self-stretch rounded-full transition-colors ${
                    isActive ? 'bg-accent' : 'bg-transparent'
                  }`}
                />
                <Icon width={17} height={17} />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Backend status */}
      <div className="border-t border-line px-5 py-4">
        <div className="flex items-center gap-2 text-xs text-ink-dim">
          {status === 'online' && <StatusDot tone="ok" />}
          {status === 'offline' && <StatusDot tone="err" />}
          {status === 'checking' && <StatusDot tone="info" pulse />}
          <span className="font-medium">
            Backend:{' '}
            {status === 'online' ? 'connected' : status === 'offline' ? 'offline' : 'checking…'}
          </span>
        </div>
        <p className="mt-1.5 text-[11px] leading-relaxed text-ink-faint">
          Research prototype. Results never constitute proof of authenticity.
        </p>
      </div>
    </div>
  )
}
