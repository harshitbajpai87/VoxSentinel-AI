import { Link, useLocation } from 'react-router-dom'
import { NAV_ITEMS } from './Sidebar'
import { MenuIcon, RefreshIcon } from '../icons'
import { Pill, StatusDot } from '../ui'
import { useApiStatus } from '../../lib/apiStatus'

export function Topbar({ onOpenMenu }: { onOpenMenu: () => void }) {
  const { pathname } = useLocation()
  const { status, lastCheckedAt, refresh } = useApiStatus()

  const current = NAV_ITEMS.find((item) => pathname.startsWith(item.to))

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-canvas">
      <div className="flex h-14 items-center gap-3 px-4 sm:px-6">
        <button
          className="rounded-md p-2 text-ink-dim hover:bg-surface-2 hover:text-ink lg:hidden"
          onClick={onOpenMenu}
          aria-label="Open navigation"
        >
          <MenuIcon />
        </button>

        {/* Breadcrumb */}
        <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2 text-sm">
          <Link to="/dashboard" className="text-ink-faint transition-colors hover:text-ink-dim">
            VoxSentinel
          </Link>
          <span className="text-ink-faint">/</span>
          <span className="truncate font-medium text-ink">{current?.label ?? 'Overview'}</span>
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={() => void refresh()}
            className="rounded-md p-2 text-ink-dim transition-colors hover:bg-surface-2 hover:text-ink"
            aria-label="Re-check backend connection"
            title={`Last checked ${lastCheckedAt ? lastCheckedAt.toLocaleTimeString() : 'never'}`}
          >
            <RefreshIcon width={15} height={15} />
          </button>
          {status === 'offline' && (
            <Pill tone="err">
              <StatusDot tone="err" /> API offline
            </Pill>
          )}
          {status === 'checking' && (
            <Pill tone="info">
              <StatusDot tone="info" pulse /> Checking
            </Pill>
          )}
        </div>
      </div>
    </header>
  )
}
