import { useEffect, useRef, useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'

export function AppLayout() {
  const [menuOpen, setMenuOpen] = useState(false)
  const dialogRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menuOpen) return
    const previous = document.activeElement as HTMLElement | null
    const oldOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialogRef.current?.querySelector<HTMLElement>('button, a')?.focus()
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuOpen(false)
      if (event.key === 'Tab') {
        const items = dialogRef.current?.querySelectorAll<HTMLElement>('button, a[href]')
        if (!items?.length) return
        const first = items[0], last = items[items.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
      }
    }
    document.addEventListener('keydown', handleKey)
    return () => { document.body.style.overflow = oldOverflow; document.removeEventListener('keydown', handleKey); previous?.focus() }
  }, [menuOpen])

  return (
    <div className="app-shell flex min-h-screen">
      <a className="skip-link" href="#workspace-main">Skip to workspace content</a>
      <aside className="hidden shrink-0 lg:block"><div className="sticky top-0 h-screen"><Sidebar /></div></aside>
      {menuOpen && (
        <div ref={dialogRef} className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-canvas/90 backdrop-blur-sm" onClick={() => setMenuOpen(false)} aria-hidden="true" />
          <div className="absolute left-0 top-0 h-full animate-fade-in shadow-card">
            <button onClick={() => setMenuOpen(false)} className="absolute right-3 top-2 z-10 rounded-full px-2 py-1 text-lg text-ink-dim" aria-label="Close navigation">×</button>
            <Sidebar onNavigate={() => setMenuOpen(false)} />
          </div>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar onOpenMenu={() => setMenuOpen(true)} />
        <main id="workspace-main" className="app-page mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 lg:px-8"><Outlet /></main>
        <footer className="border-t border-line/60 px-6 py-4 text-center text-[11px] text-ink-faint">VoxSentinel AI · AI Foundations · SIH 2026 · SIH26104 · Harshit Bajpai &amp; Oisiky Sinha</footer>
      </div>
    </div>
  )
}
