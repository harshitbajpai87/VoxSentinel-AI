// Marketing landing page — foreground only.
//
// Honesty rules that bind the rest of the app apply here too: the hero
// illustration shows an ILLUSTRATIVE sample card (clearly labelled), never a
// real detection result; the page invents no pricing, testimonials, usage
// counters, certifications or guarantees. Every button routes to a real,
// existing page of the application.
import { Link } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { CheckIcon, PlayIcon, ShieldIcon, WaveformIcon } from '../components/icons'
import { Button, LinkButton, Tag } from '../components/ui'
import { Reveal, StaggerGroup, StaggerItem } from '../components/motion/Reveal'

/* ---------------- hero illustration ---------------- */

/** Decorative animated bars inside the illustrative player card. */
function HeroWave() {
  const reduced = useReducedMotion()
  const heights = [0.45, 0.7, 0.35, 0.85, 0.55, 0.95, 0.4, 0.75, 0.6, 0.9, 0.5, 0.8]
  return (
    <div className="flex h-16 items-center gap-1.5" aria-hidden="true">
      {heights.map((h, i) => (
        <motion.span
          key={i}
          className="w-1.5 rounded-full bg-gradient-to-t from-accent via-magenta to-coral"
          style={{ height: `${h * 100}%` }}
          {...(reduced
            ? {}
            : {
                animate: { scaleY: [1, 0.45, 1] },
                transition: {
                  duration: 1.6 + (i % 4) * 0.35,
                  repeat: Infinity,
                  ease: 'easeInOut',
                  delay: i * 0.12,
                },
              })}
        />
      ))}
    </div>
  )
}

/** Illustrative floating analysis card. Labelled as a demo, never real output. */
function HeroIllustration() {
  const reduced = useReducedMotion()
  const float = (duration: number, delay: number, y: number) =>
    reduced
      ? {}
      : {
          animate: { y: [0, y, 0] },
          transition: { duration, delay, repeat: Infinity, ease: 'easeInOut' as const },
        }

  return (
    <div className="relative mx-auto w-full max-w-md">
      {/* Ambient glows — foreground decoration inside the illustration only. */}
      <div
        aria-hidden="true"
        className="absolute -left-10 -top-10 h-44 w-44 rounded-full bg-lavender/70 blur-3xl"
      />
      <div
        aria-hidden="true"
        className="absolute -bottom-12 -right-6 h-48 w-48 rounded-full bg-soft-coral/80 blur-3xl"
      />

      {/* Main illustrative audio player card */}
      <motion.div
        className="relative rounded-2xl border border-line bg-surface p-5 shadow-card"
        initial={{ opacity: 0, scale: 0.94 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.15, ease: [0.21, 0.6, 0.35, 1] }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-r from-accent-bright via-magenta to-coral text-white">
              <PlayIcon width={14} height={14} />
            </span>
            <div>
              <p className="text-sm font-semibold text-ink">interview clip.wav</p>
              <p className="text-xs text-ink-faint">0:07 · 1.2 MB · 16 kHz</p>
            </div>
          </div>
          <Tag>Illustrative demo</Tag>
        </div>
        <div className="mt-4">
          <HeroWave />
        </div>
        <div className="mt-4 h-1.5 rounded-full bg-surface-3">
          <motion.div
            className="h-1.5 rounded-full bg-gradient-to-r from-accent via-magenta to-coral"
            {...(reduced ? {} : { animate: { width: ['8%', '86%'] }, transition: { duration: 7, repeat: Infinity, ease: 'linear' } })}
            style={{ width: '60%' }}
          />
        </div>
      </motion.div>

      {/* Floating: synthetic-voice warning card */}
      <motion.div
        className="absolute -right-4 top-24 w-56 rounded-xl border border-line bg-surface p-4 shadow-card sm:-right-10"
        {...float(5.2, 0.3, -10)}
      >
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-soft-coral text-danger">
            <ShieldIcon width={13} height={13} />
          </span>
          <p className="text-xs font-semibold text-ink">Synthetic voice signals</p>
        </div>
        <p className="mt-2 text-[11px] leading-relaxed text-ink-dim">
          Spectral flatness and pitch patterns consistent with synthesis.
        </p>
      </motion.div>

      {/* Floating: risk + confidence card */}
      <motion.div
        className="absolute -left-4 bottom-16 w-52 rounded-xl border border-line bg-surface p-4 shadow-card sm:-left-10"
        {...float(6.4, 0.8, 12)}
      >
        <p className="text-[11px] font-medium text-ink-faint">Risk score (demo)</p>
        <div className="mt-2 flex items-baseline gap-2">
          <p className="text-2xl font-semibold text-ink">72%</p>
          <p className="text-[11px] text-ink-dim">model confidence 64%</p>
        </div>
        <div className="mt-2 h-1.5 rounded-full bg-surface-3">
          <div className="h-1.5 w-[72%] rounded-full bg-gradient-to-r from-accent via-magenta to-coral" />
        </div>
      </motion.div>

      {/* Floating: upload action chip */}
      <motion.div
        className="absolute -top-5 right-10 flex items-center gap-2 rounded-full border border-line bg-surface px-3.5 py-2 shadow-card"
        {...float(4.6, 0.6, -8)}
      >
        <WaveformIcon width={14} height={14} className="text-accent" />
        <span className="text-xs font-medium text-ink">Upload Audio</span>
      </motion.div>
    </div>
  )
}

/* ---------------- sections ---------------- */

const FEATURES = [
  {
    title: 'Deepfake & clone detection',
    text: 'A trained detector scores every clip across spectral and prosodic features — and every verdict shows the measured evidence behind it.',
  },
  {
    title: 'Explainable AI results',
    text: 'No black-box percentage: windowed scores, measured features and honest limitations are reported with each analysis.',
  },
  {
    title: 'Secure & privacy-focused',
    text: 'Consent is required before any analysis, audio is processed for the analysis only, and the whole pipeline can run locally.',
  },
]

const STEPS = [
  { title: 'Upload or record', text: 'Drop an MP3, WAV, M4A, FLAC or OGG clip — or record live from your microphone with consent.' },
  { title: 'Analyze with real models', text: 'The detector extracts features, scores the clip and calibrates the probability — no fabricated verdicts.' },
  { title: 'Inspect the evidence', text: 'Review the classification, risk percentage, per-window scores and honest limitations before acting.' },
]

export default function LandingPage() {
  const reduced = useReducedMotion()
  const hoverLift = reduced ? '' : 'transition-transform duration-200 hover:-translate-y-0.5'

  return (
    <div className="min-h-screen bg-canvas">
      {/* Navigation */}
      <Reveal delay={0.05}>
        <header className="border-b border-line bg-surface">
          <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
            <Link to="/dashboard" className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-bright via-magenta to-coral text-white">
                <WaveformIcon width={17} height={17} />
              </span>
              <span>
                <span className="block text-sm font-bold text-ink">VoxSentinel AI</span>
                <span className="block text-[11px] text-ink-faint">Voice Authenticity. Trusted.</span>
              </span>
            </Link>
            <div className="hidden items-center gap-6 text-sm text-ink-dim md:flex">
              <a href="#how" className="hover:text-accent">How It Works</a>
              <a href="#features" className="hover:text-accent">Features</a>
              <Link to="/docs" className="hover:text-accent">Docs</Link>
            </div>
            <div className="flex items-center gap-3">
              <LinkButton href="/dashboard" variant="secondary">Open Dashboard</LinkButton>
              <LinkButton href="/analyze">Analyze Audio</LinkButton>
            </div>
          </nav>
        </header>
      </Reveal>

      {/* Hero */}
      <section className="mx-auto grid max-w-6xl items-center gap-12 px-6 pb-20 pt-16 lg:grid-cols-2 lg:pt-24">
        <div>
          <Reveal delay={0.15}>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1 text-xs font-medium text-accent">
              <ShieldIcon width={12} height={12} /> AI-Powered Voice Security
            </span>
          </Reveal>
          <Reveal delay={0.25}>
            <h1 className="mt-5 text-4xl font-bold leading-tight tracking-tight text-ink sm:text-5xl">
              Know What&rsquo;s Real.{' '}
              <span className="bg-gradient-to-r from-accent-bright via-magenta to-coral bg-clip-text text-transparent">
                Detect What&rsquo;s Synthetic.
              </span>
            </h1>
          </Reveal>
          <Reveal delay={0.35}>
            <p className="mt-4 max-w-md text-base leading-relaxed text-ink-dim">
              AI-powered voice authenticity analysis for a world where audio can no longer be
              trusted at face value.
            </p>
          </Reveal>
          <StaggerGroup className="mt-6 space-y-2.5">
            {FEATURES.map((f) => (
              <StaggerItem key={f.title}>
                <p className="flex items-center gap-2.5 text-sm text-ink">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-lavender text-accent">
                    <CheckIcon width={11} height={11} />
                  </span>
                  {f.title}
                </p>
              </StaggerItem>
            ))}
          </StaggerGroup>
          <Reveal delay={0.55}>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <LinkButton href="/analyze" className="group">
                Analyze Audio
                <span className="transition-transform duration-200 group-hover:translate-x-1">→</span>
              </LinkButton>
              <LinkButton href="/dashboard" variant="secondary">Explore Features</LinkButton>
            </div>
            <p className="mt-4 text-xs text-ink-faint">
              Supported formats: MP3 · WAV · M4A · FLAC · OGG
            </p>
          </Reveal>
        </div>
        <Reveal delay={0.3} className="lg:pl-6">
          <HeroIllustration />
        </Reveal>
      </section>

      {/* How it works */}
      <section id="how" className="border-t border-line bg-surface px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <Reveal inView>
            <h2 className="text-2xl font-bold text-ink">How VoxSentinel AI works</h2>
            <p className="mt-2 max-w-xl text-sm text-ink-dim">
              Three steps, every number from a real model — nothing simulated.
            </p>
          </Reveal>
          <StaggerGroup className="mt-10 grid gap-6 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <StaggerItem key={s.title} className={`rounded-xl border border-line bg-canvas p-6 ${hoverLift}`}>
                <span className="text-xs font-semibold text-ink-faint">0{i + 1}</span>
                <h3 className="mt-2 text-base font-semibold text-ink">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-dim">{s.text}</p>
              </StaggerItem>
            ))}
          </StaggerGroup>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <Reveal inView>
            <h2 className="text-2xl font-bold text-ink">Built for evidence, not vibes</h2>
            <p className="mt-2 max-w-xl text-sm text-ink-dim">
              A research prototype that reports what the models actually measured.
            </p>
          </Reveal>
          <StaggerGroup className="mt-10 grid gap-6 md:grid-cols-3">
            {FEATURES.map((f) => (
              <StaggerItem key={f.title} className={`rounded-xl border border-line bg-surface p-6 shadow-card ${hoverLift}`}>
                <h3 className="text-base font-semibold text-ink">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-dim">{f.text}</p>
              </StaggerItem>
            ))}
          </StaggerGroup>
        </div>
      </section>

      {/* Final CTA */}
      <section className="border-t border-line bg-surface px-6 py-20">
        <Reveal inView className="mx-auto max-w-3xl text-center">
          <h2 className="text-2xl font-bold text-ink">Analyze your first clip</h2>
          <p className="mt-3 text-sm text-ink-dim">
            The full analysis workspace — upload, recording, history and evaluation — is one
            click away.
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <LinkButton href="/analyze" className="group">
              Analyze Audio
              <span className="transition-transform duration-200 group-hover:translate-x-1">→</span>
            </LinkButton>
            <LinkButton href="/record" variant="secondary">Record Live</LinkButton>
          </div>
        </Reveal>
      </section>

      {/* Footer */}
      <footer className="border-t border-line px-6 py-8">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 text-xs text-ink-faint">
          <p>VoxSentinel AI — research prototype. Results are risk signals to investigate, never proof of authenticity.</p>
          <div className="flex gap-4">
            <Link to="/docs" className="hover:text-accent">Documentation</Link>
            <Link to="/security" className="hover:text-accent">Security &amp; Audit</Link>
            <Link to="/settings" className="hover:text-accent">Settings</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
