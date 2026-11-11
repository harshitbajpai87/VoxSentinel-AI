import { useState, type CSSProperties, type PointerEvent } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { Reveal } from '../components/motion/Reveal'

const TOOLS = [
  { name: 'Audio analysis', short: 'Analyze', route: '/analyze', label: 'Inspect synthetic audio signals', body: 'Upload a recording and inspect risk signals, measured audio features, and window-by-window scores.', detail: 'MP3 · WAV · M4A · FLAC · OGG', color: 'peach' },
  { name: 'Speaker comparison', short: 'Compare', route: '/verify', label: 'Compare two speaker recordings', body: 'Compare an authorized reference voice with a second recording and review speaker similarity against the model’s limits.', detail: 'Reference recording + comparison clip', color: 'sage' },
  { name: 'Scam protection', short: 'Protect', route: '/protection', label: 'Screen a suspicious call', body: 'Review a call recording for scam and coercion signals, combining transcription, voice checks and pattern analysis.', detail: 'Family emergency & scam screening', color: 'rose' },
  { name: 'Live recording', short: 'Record', route: '/record', label: 'Record from your microphone', body: 'Capture audio with consent and send it through the same analysis pipeline for a full-clip result.', detail: 'Microphone permission required', color: 'butter' },
] as const

const FAQS = [
  ['Can a score prove a voice is real?', 'No. Model outputs are risk signals, not proof. Recording quality, noise, compression and unfamiliar voices can affect the result. Check the evidence and verify important claims independently.'],
  ['What happens when the backend is offline?', 'The workspace shows a connection error rather than inventing a result. Start the existing backend, check the configured API address, then re-check the connection.'],
  ['Can I record someone without their permission?', 'Use recordings only when you have the right and consent to do so. The recording and upload workspaces include consent controls.'],
  ['Where can I read the technical details?', 'The repository documentation (docs/ and backend/docs/) describes the pipeline, the model card and the known limits of this prototype.'],
]

function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d={diagonal ? 'M5 19 19 5M5 5h14v14' : 'M4 12h15M12 5l7 7-7 7'} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
}

function VoiceMark({ compact = false }: { compact?: boolean }) {
  return <span className={`voice-mark ${compact ? 'compact' : ''}`} aria-hidden="true">{[13, 23, 33, 22, 13].map((h, i) => <i key={i} style={{ height: h }} />)}</span>
}

function VoiceArt() {
  const rings = Array.from({ length: 11 }, (_, k) => {
    const points = Array.from({ length: 181 }, (_, j) => {
      const t = j / 180 * Math.PI * 2
      const r = 105 + k * 9 + Math.sin(t * 5 + k * .36) * (9 + k * 1.1) + Math.cos(t * 3 - k * .22) * 7
      return `${(250 + Math.cos(t) * r).toFixed(2)},${(250 + Math.sin(t) * r).toFixed(2)}`
    })
    return <path key={k} d={`M${points.join(' L')}Z`} fill="none" stroke={k % 3 === 0 ? '#6b7543' : k % 3 === 1 ? '#b54b34' : '#994c75'} strokeWidth={k % 3 === 0 ? '1.4' : '1'} opacity={.3 + k * .055} />
  })
  return <div className="voice-art" aria-hidden="true">
    <span className="art-coordinate top">SPECTRAL STUDY / 001</span>
    <svg className="voice-rings" viewBox="0 0 500 500"><circle cx="250" cy="250" r="238" fill="none" stroke="#d9cbbb" strokeWidth=".65" strokeDasharray="2 8" />{rings}</svg>
    <div className="art-center"><VoiceMark /><span>Signal<br />inspection.</span></div>
    <span className="art-coordinate bottom">DECORATIVE ILLUSTRATION — NOT A RESULT</span>
    <span className="art-orbit"><i /></span>
  </div>
}

function MiniWave({ variant = 0 }: { variant?: number }) {
  return <div className="mini-wave" aria-hidden="true">{Array.from({ length: 43 }, (_, i) => <i key={i} style={{ height: `${12 + (Math.sin(i * .8 + variant) * .5 + .5) * (24 + Math.sin(i * .29) * 25)}%`, '--bar-delay': `${i * .035}s` } as CSSProperties} />)}</div>
}

function FeatureIllustration({ index }: { index: number }) {
  return <div className={`feature-art ${TOOLS[index].color}`} aria-hidden="true">
    <div className="paper-back back-one" /><div className="paper-back back-two" />
    <div className="paper-front">
      <div className="paper-top"><VoiceMark compact /><span>VOXSENTINEL / {TOOLS[index].short.toUpperCase()}</span><span>↗</span></div>
      {index === 0 && <><p className="paper-title">Audio<br />analysis.</p><MiniWave /><div className="paper-rule" /><p className="paper-label">Waveform illustration</p></>}
      {index === 1 && <><p className="paper-title">Speaker<br />comparison.</p><MiniWave variant={1} /><MiniWave variant={2} /><p className="paper-label">Two illustrative audio tracks</p></>}
      {index === 2 && <><p className="paper-title">Scam<br />screening.</p><div className="check-row"><span>01</span> Stop the conversation.</div><div className="check-row"><span>02</span> Call a number you trust.</div><div className="check-row"><span>03</span> Verify before sending.</div></>}
      {index === 3 && <><p className="paper-title">Recording<br />capture.</p><div className="record-illustration"><VoiceMark /><span>Consent before capture</span></div><p className="paper-label">Recording starts only in the workspace</p></>}
    </div>
  </div>
}

export default function LandingPage() {
  const prefersReduced = useReducedMotion()
  const [paused, setPaused] = useState(false)
  const [active, setActive] = useState(0)
  const [menuOpen, setMenuOpen] = useState(false)
  const still = paused || !!prefersReduced

  const tilt = (event: PointerEvent<HTMLDivElement>) => {
    if (still || event.pointerType !== 'mouse') return
    const box = event.currentTarget.getBoundingClientRect()
    event.currentTarget.style.setProperty('--tilt-x', `${(event.clientY - box.top - box.height / 2) / box.height * -7}deg`)
    event.currentTarget.style.setProperty('--tilt-y', `${(event.clientX - box.left - box.width / 2) / box.width * 7}deg`)
  }
  const resetTilt = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.style.setProperty('--tilt-x', '0deg')
    event.currentTarget.style.setProperty('--tilt-y', '0deg')
  }

  return <div className={`landing ${still ? 'motion-paused' : ''}`}>
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="site-header">
      <Link className="site-brand" to="/" aria-label="VoxSentinel AI home"><VoiceMark compact /><span>VoxSentinel<span className="brand-ai"> AI</span></span></Link>
      <nav className="desktop-nav" aria-label="Site navigation"><a href="#tools">Core capabilities</a><a href="#process">Workflow</a></nav>
      <div className="nav-actions"><button className="motion-toggle" onClick={() => setPaused(!paused)} aria-pressed={paused} aria-label={paused ? 'Resume decorative motion' : 'Pause decorative motion'}>{paused ? 'Motion off' : 'Motion on'}<span aria-hidden="true">{paused ? '▷' : 'Ⅱ'}</span></button><Link to="/dashboard" className="nav-workspace">Workspace <Arrow diagonal /></Link><button className="mobile-menu" aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen} aria-controls="mobile-site-nav" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? '−' : '+'}</button></div>
    </header>
    {menuOpen && <nav id="mobile-site-nav" className="mobile-site-nav" aria-label="Mobile site navigation"><a href="#tools" onClick={() => setMenuOpen(false)}>Core capabilities</a><a href="#process" onClick={() => setMenuOpen(false)}>Workflow</a><Link to="/history">Analysis history</Link><Link to="/dashboard">Workspace</Link></nav>}

    <main id="main">
      <section className="hero-section" aria-labelledby="hero-heading">
        <div className="hero-intro"><span className="eyebrow"><span className="tiny-dot" /> Research Prototype v0.1</span><span className="edition">VOXSENTINEL — AUDIO FORENSICS</span></div>
        <div className="hero-grid">
          <div className="hero-copy"><Reveal><h1 id="hero-heading">Synthetic Audio Detection<br /><em>&amp; Speaker Verification</em></h1></Reveal><Reveal delay={.12}><p>An open-source research workspace to inspect audio features, analyze risk signals, and investigate potential synthetic voice manipulation.</p><div className="hero-actions"><Link className="round-link" to="/analyze"><span>Analyze<br />a clip</span><Arrow diagonal /></Link><Link to="/record" className="text-link">Or record your own <Arrow /></Link></div></Reveal></div>
          <div className="hero-visual" onPointerMove={tilt} onPointerLeave={resetTilt}><VoiceArt /><div className="visual-note"><span>01 / THE VOICEPRINT</span><p>Illustrative feature view.<br />Not a computed result.</p></div></div>
        </div>
        <div className="hero-bottom"><span>Research Prototype v0.1 • For evaluation only</span><a href="#tools" className="scroll-prompt">Explore the capabilities <span aria-hidden="true">↓</span></a></div>
      </section>

      <section id="tools" className="tool-section" aria-labelledby="tools-heading">
        <div className="section-top"><span className="eyebrow">01 / CORE CAPABILITIES</span><span>Modular tools designed for technical inspection and forensic audio analysis.</span></div>
        <Reveal inView><h2 id="tools-heading">Core<br /><em>capabilities.</em></h2></Reveal>
        <div className="tool-stage">
          <div className="tool-controls" role="tablist" aria-label="Explore audio tools">{TOOLS.map((tool, i) => <button id={`tool-tab-${i}`} key={tool.short} role="tab" aria-selected={active === i} aria-controls={`tool-panel-${i}`} tabIndex={active === i ? 0 : -1} onClick={() => setActive(i)} onKeyDown={(e) => { let next = i; if (e.key === 'ArrowDown' || e.key === 'ArrowRight') next = (i + 1) % TOOLS.length; else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') next = (i - 1 + TOOLS.length) % TOOLS.length; else if (e.key === 'Home') next = 0; else if (e.key === 'End') next = TOOLS.length - 1; else return; e.preventDefault(); setActive(next); document.getElementById(`tool-tab-${next}`)?.focus() }}><span>0{i + 1}</span>{tool.name}<Arrow diagonal /></button>)}</div>
          <div id={`tool-panel-${active}`} role="tabpanel" aria-labelledby={`tool-tab-${active}`} className="tool-panel" tabIndex={0}>
            <AnimatePresence mode="wait" initial={false}><motion.div key={active} initial={still ? false : { opacity: 0, y: 14, rotate: -1 }} animate={{ opacity: 1, y: 0, rotate: 0 }} exit={{ opacity: 0, y: still ? 0 : -8 }} transition={{ duration: still ? 0 : .25 }}><FeatureIllustration index={active} /></motion.div></AnimatePresence>
            <div className="tool-description"><p className="eyebrow">{TOOLS[active].detail}</p><h3>{TOOLS[active].label}</h3><p>{TOOLS[active].body}</p><Link to={TOOLS[active].route} className="text-link">Open {TOOLS[active].name.toLowerCase()} <Arrow /></Link></div>
          </div>
        </div>
      </section>

      <section id="process" className="process-section" aria-labelledby="process-heading"><div className="section-top"><span className="eyebrow">02 / WORKFLOW</span><span>Three stages, reproducible at every step.</span></div><div className="process-grid"><Reveal inView><h2 id="process-heading">Three-step<br /><em>workflow.</em></h2><p className="process-intro">From input and consent to a window-by-window forensic review.</p><a href="#disclaimer" className="text-link">Model limitations &amp; disclaimer <Arrow /></a></Reveal><div className="process-steps">{[
        ['01', 'Input & Consent', 'Upload an audio clip (MP3, WAV, M4A, FLAC) or record with active consent indicators.'],
        ['02', 'Pipeline Extraction', 'The backend extracts spectral features and runs detection models, outputting transparent error states if a file fails.'],
        ['03', 'Forensic Review', 'Inspect window-by-window risk scores and cross-verify findings independently.'],
      ].map(([n, title, body]) => <Reveal inView key={n}><article className="process-step"><span>{n}</span><div><h3>{title}</h3><p>{body}</p></div><Arrow diagonal /></article></Reveal>)}</div></div></section>

      <section id="disclaimer" className="evidence-section" aria-labelledby="evidence-heading"><div className="evidence-inner"><div className="evidence-stamp"><VoiceMark /><span>DISCLAIMER</span></div><div><span className="eyebrow">READ BEFORE USE</span><h2 id="evidence-heading">Model Limitations<br /><em>&amp; Disclaimer</em></h2><p>This is a research tool. Environmental noise, compression, and language variations can impact accuracy. Risk scores represent model probabilities, not definitive proof. Always verify critical audio evidence through independent channels.</p></div></div></section>

      <section className="faq-section" aria-labelledby="faq-heading"><div><span className="eyebrow">03 / COMMON QUESTIONS</span><h2 id="faq-heading">Frequently<br /><em>asked.</em></h2></div><div className="faq-list">{FAQS.map(([q, a]) => <details key={q}><summary>{q}<span aria-hidden="true">+</span></summary><p>{a}</p></details>)}</div></section>
      <section className="closing-section"><p className="eyebrow">NEXT STEP</p><h2>Start with an<br /><em>audio analysis.</em></h2><div><Link to="/analyze" className="pill-link">Open audio analysis <Arrow diagonal /></Link><Link to="/dashboard" className="text-link">Explore the workspace <Arrow /></Link></div></section>
    </main>
    <footer className="site-footer"><div className="footer-top"><Link className="site-brand" to="/"><VoiceMark compact /><span>VoxSentinel AI</span></Link><div><Link to="/analyze">Analyze audio</Link><Link to="/verify">Speaker verification</Link><Link to="/history">Analysis history</Link></div><a href="#main">Back to top ↑</a></div><div className="footer-bottom"><span>AI Foundations · SIH 2026 · SIH26104</span><span>Harshit Bajpai & Oisiky Sinha</span><span>Research Prototype • For Evaluation Only</span></div></footer>
  </div>
}
