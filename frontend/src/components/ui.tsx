// VoxSentinel design system: small, reusable, accessible primitives.
// Every color comes from the token palette in tailwind.config.js.
import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react'
import { AlertIcon, CheckIcon, RefreshIcon, ShieldIcon, XIcon } from './icons'

/* ---------- Button ---------- */

type ButtonVariant = 'primary' | 'secondary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
}

const buttonVariants: Record<ButtonVariant, string> = {
  primary:
    'bg-accent text-white font-semibold hover:bg-accent-bright active:brightness-95',
  secondary:
    'bg-surface-2 text-ink border border-line hover:border-accent/40 hover:text-accent',
  ghost: 'text-ink-dim hover:text-ink hover:bg-surface-2',
}

export function Button({ variant = 'primary', className = '', ...props }: ButtonProps) {
  return (
    <button
      className={`action-button inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm
        transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0
        ${buttonVariants[variant]} ${className}`}
      {...props}
    />
  )
}

export function LinkButton({
  variant = 'primary',
  className = '',
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: ButtonVariant }) {
  return (
    <a
      className={`action-button inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm
        transition-colors duration-150 ${buttonVariants[variant]} ${className}`}
      {...props}
    />
  )
}

/* ---------- Card ---------- */

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children?: ReactNode
  className?: string
}) {
  return (
    <section
      className={`app-panel rounded-lg border border-line bg-surface shadow-card transition-colors duration-200 ${className}`}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line/70 px-5 py-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs leading-relaxed text-ink-dim">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="px-5 py-4">{children}</div>
    </section>
  )
}

/* ---------- Pill / Tag / StatusDot ---------- */

type PillTone = 'ok' | 'warn' | 'err' | 'info' | 'neutral'

const pillTones: Record<PillTone, string> = {
  ok: 'bg-success/10 text-success border-success/25',
  warn: 'bg-warning/10 text-warning border-warning/25',
  err: 'bg-danger/10 text-danger border-danger/25',
  info: 'bg-accent/10 text-accent border-accent/25',
  neutral: 'bg-surface-2 text-ink-dim border-line',
}

export function Pill({
  tone = 'neutral',
  children,
  className = '',
}: {
  tone?: PillTone
  children: ReactNode
  className?: string
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${pillTones[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-sm border border-magenta/30 bg-magenta/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-magenta">
      {children}
    </span>
  )
}

export function StatusDot({ tone = 'neutral', pulse = false }: { tone?: PillTone; pulse?: boolean }) {
  const colors: Record<PillTone, string> = {
    ok: 'bg-success',
    warn: 'bg-warning',
    err: 'bg-danger',
    info: 'bg-accent',
    neutral: 'bg-ink-faint',
  }
  return (
    <span className="relative inline-flex h-2 w-2">
      {pulse && (
        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${colors[tone]} opacity-60`} />
      )}
      <span className={`relative inline-flex h-2 w-2 rounded-full ${colors[tone]}`} />
    </span>
  )
}

/* ---------- StatTile ---------- */

export function StatTile({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: PillTone
}) {
  const valueColors: Record<PillTone, string> = {
    ok: 'text-success',
    warn: 'text-warning',
    err: 'text-danger',
    info: 'text-accent',
    neutral: 'text-ink',
  }
  return (
    <div className="app-stat rounded-lg border border-line bg-surface px-5 py-4">
      <p className="text-xs font-medium uppercase tracking-wider text-ink-dim">{label}</p>
      <p className={`mt-2 font-mono text-2xl font-semibold tabular-nums ${valueColors[tone]}`}>
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-ink-faint">{hint}</p>}
    </div>
  )
}

/* ---------- Empty / Error / Loading states ---------- */

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode
  title: string
  description?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-line bg-surface/50 px-6 py-12 text-center animate-fade-in">
      {icon && <div className="text-ink-faint">{icon}</div>}
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {description && (
        <p className="max-w-md text-xs leading-relaxed text-ink-dim">{description}</p>
      )}
      {action}
    </div>
  )
}

export function ErrorState({
  title = 'Backend unavailable',
  description,
  onRetry,
}: {
  title?: string
  description?: ReactNode
  onRetry?: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-danger/20 bg-danger/5 px-6 py-10 text-center animate-fade-in">
      <AlertIcon width={22} height={22} className="text-danger" />
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {description && <p className="max-w-md text-xs leading-relaxed text-ink-dim">{description}</p>}
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          <RefreshIcon width={14} height={14} /> Retry
        </Button>
      )}
    </div>
  )
}

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-10 text-sm text-ink-dim" role="status">
      <svg className="h-4 w-4 animate-spin text-accent" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
        <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      {label}
    </div>
  )
}

/* ---------- Callout ---------- */

type CalloutTone = 'info' | 'warning' | 'ethics'

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: CalloutTone
  title: string
  children: ReactNode
}) {
  const styles: Record<CalloutTone, string> = {
    info: 'border-accent/20 bg-accent/5',
    warning: 'border-warning/25 bg-warning/5',
    ethics: 'border-magenta/25 bg-magenta/5',
  }
  const icons: Record<CalloutTone, ReactNode> = {
    info: <CheckIcon width={16} height={16} className="mt-0.5 text-accent" />,
    warning: <AlertIcon width={16} height={16} className="mt-0.5 text-warning" />,
    ethics: <ShieldIcon width={16} height={16} className="mt-0.5 text-magenta" />,
  }
  return (
    <div className={`flex gap-3 rounded-lg border px-4 py-3.5 ${styles[tone]}`}>
      {icons[tone]}
      <div>
        <p className="text-xs font-semibold text-ink">{title}</p>
        <div className="mt-1 text-xs leading-relaxed text-ink-dim">{children}</div>
      </div>
    </div>
  )
}

/* ---------- PageHeader ---------- */

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string
  subtitle?: ReactNode
  actions?: ReactNode
}) {
  return (
    <header className="app-page-header mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-ink-dim">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}

/* ---------- Helper: guard icon for checklists ---------- */

export function StateIcon({ ok, pending = false }: { ok: boolean; pending?: boolean }) {
  if (ok) return <CheckIcon width={16} height={16} className="text-success" />
  if (pending) return <RefreshIcon width={16} height={16} className="text-ink-faint" />
  return <XIcon width={16} height={16} className="text-ink-faint" />
}
