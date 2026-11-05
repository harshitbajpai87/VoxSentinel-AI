import { useEffect, useRef, useState } from 'react'
import { analyzeAudio, ApiError, ApiUnavailableError } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { AnalysisRecord, WindowScore } from '../lib/types'
import { detectRecorderSupport, meanScore } from '../lib/audio/recorderSupport'
import {
  Button,
  Callout,
  Card,
  PageHeader,
  Pill,
} from '../components/ui'
import { WaveformIcon, ShieldIcon } from '../components/icons'
import { AnalysisResultPanel } from '../components/audio/AnalysisResultPanel'
import WindowScoresChart from '../components/audio/WindowScoresChart'

/** How often a provisional prefix is scored while recording (seconds). */
const WINDOW_SECONDS = 4

type Phase = 'idle' | 'recording' | 'paused' | 'processing' | 'done'

interface WindowResult {
  score: number
  durationS: number
}

export default function RecordAudioPage() {
  const { status } = useApiStatus()
  const support = detectRecorderSupport()

  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [windows, setWindows] = useState<WindowResult[]>([])
  const [provisionalMean, setProvisionalMean] = useState<number | null>(null)
  const [provisionalRisk, setProvisionalRisk] = useState<number | null>(null)
  const [final, setFinal] = useState<AnalysisRecord | null>(null)
  const [persist, setPersist] = useState(true)

  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  // Rolling duration timer while recording.
  useEffect(() => {
    if (phase !== 'recording') return
    const t = window.setInterval(() => setElapsed((e) => e + 1), 1000)
    return () => window.clearInterval(t)
  }, [phase])

  const cleanupStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    recorderRef.current = null
  }

  const resetAll = () => {
    setError(null)
    setElapsed(0)
    setWindows([])
    setProvisionalMean(null)
    setProvisionalRisk(null)
    setFinal(null)
    chunksRef.current = []
  }

  /** Scores the audio recorded so far WITHOUT persisting it (provisional). */
  const scoreProvisional = async (mimeType: string, extension: string) => {
    if (!chunksRef.current.length) return
    const blob = new Blob(chunksRef.current, { type: mimeType })
    try {
      const file = new File([blob], `live-prefix.${extension}`, { type: mimeType })
      const record = await analyzeAudio(file, { persist: false, timeoutMs: 15000 })
      const evidenceWindows = (record.evidence?.time_window_scores ?? []) as WindowScore[]
      const durationS = record.input.duration_s
      // Provisional prefix analysis: full-clip score of everything so far.
      const last = evidenceWindows.length ? evidenceWindows[evidenceWindows.length - 1] : null
      const score = record.scores.raw_score ?? (last ? last.score : 0)
      setProvisionalRisk(record.scores.calibrated_probability)
      setWindows((prev) => {
        const next = [...prev, { score, durationS }]
        setProvisionalMean(meanScore(next.map((w) => w.score)))
        return next
      })
    } catch (e) {
      // Provisional windows are best-effort evidence; surface but do not stop recording.
      if (e instanceof ApiError || e instanceof ApiUnavailableError) setError(e.message)
    }
  }

  const start = async () => {
    resetAll()
    if (!support.supported) {
      setError('This browser does not support audio recording (MediaRecorder).')
      return
    }
    if (status !== 'online') {
      setError('Backend is offline. Recording needs the analysis API for live scores.')
      return
    }
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Microphone permission was denied or no microphone is available.')
      return
    }
    streamRef.current = stream
    chunksRef.current = []

    const recorder = new MediaRecorder(stream, { mimeType: support.mimeType })
    recorderRef.current = recorder
    let windowSeconds = 0
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.onstop = () => {
      window.clearInterval(windowTimer)
      void finish(support.mimeType, support.extension)
    }
    // 1s timeslices so prefix blobs stay decodable; score every WINDOW_SECONDS.
    const windowTimer = window.setInterval(() => {
      windowSeconds += 1
      if (windowSeconds % WINDOW_SECONDS === 0) void scoreProvisional(support.mimeType, support.extension)
    }, 1000)

    recorder.start(1000)
    setPhase('recording')
  }

  const pause = () => {
    const r = recorderRef.current
    if (!r || r.state !== 'recording') return
    r.pause()
    setPhase('paused')
  }

  const resume = () => {
    const r = recorderRef.current
    if (!r || r.state !== 'paused') return
    r.resume()
    setPhase('recording')
  }

  const stop = () => {
    const r = recorderRef.current
    if (!r || r.state === 'inactive') return
    setPhase('processing')
    try {
      r.stop()
    } catch {
      cleanupStream()
      setPhase('idle')
    }
  }

  /** Final full-clip analysis; persisted only if the user allowed saving. */
  const finish = async (mimeType: string, extension: string) => {
    cleanupStream()
    if (!chunksRef.current.length) {
      setError('No audio was captured.')
      setPhase('idle')
      return
    }
    const blob = new Blob(chunksRef.current, { type: mimeType })
    const file = new File([blob], `live-recording.${extension}`, { type: mimeType })
    try {
      const record = await analyzeAudio(file, { persist })
      setFinal(record)
      setPhase('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The analysis failed.')
      setPhase('idle')
    }
  }

  // Ecosystems hygiene: stop the mic if the user navigates away mid-recording.
  useEffect(() => () => cleanupStream(), [])

  const chartWindows: WindowScore[] = windows.map((w) => ({
    start_s: 0,
    end_s: w.durationS,
    score: w.score,
  }))

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Live Recording"
        subtitle="Record from the microphone with honest, model-derived risk signals as the audio grows. Provisional scores are computed live; the final verdict always comes from the full clip."
        actions={
          phase === 'recording' ? (
            <Pill tone="err" >
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-danger" />
                Recording
              </span>
            </Pill>
          ) : phase === 'paused' ? (
            <Pill tone="warn">Paused</Pill>
          ) : phase === 'processing' ? (
            <Pill tone="info">Analyzing…</Pill>
          ) : null
        }
      />

      {!support.supported && (
        <div className="mb-4">
          <Callout tone="warning" title="Live recording unavailable in this browser">
            Your browser does not expose MediaRecorder for audio. Use the Analyze Audio page to
            upload a file instead — every other feature works there.
          </Callout>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card
            title="Recorder"
            subtitle={support.supported ? 'Audio is analyzed locally by the backend and never leaves your machine.' : undefined}
            actions={
              <span className="font-mono text-sm text-ink-dim">
                {String(Math.floor(elapsed / 60)).padStart(2, '0')}:
                {String(elapsed % 60).padStart(2, '0')}
              </span>
            }
          >
            <div className="flex flex-wrap items-center gap-3">
              {phase === 'idle' || phase === 'done' ? (
                <Button onClick={() => void start()} disabled={!support.supported || status !== 'online'}>
                  <WaveformIcon width={14} height={14} />
                  {phase === 'done' ? 'Record again' : 'Start recording'}
                </Button>
              ) : (
                <>
                  {phase === 'recording' ? (
                    <Button variant="secondary" onClick={pause}>Pause</Button>
                  ) : (
                    <Button onClick={resume}>Resume</Button>
                  )}
                  <Button variant="primary" onClick={stop}>
                    <WaveformIcon width={14} height={14} /> Stop & analyze
                  </Button>
                </>
              )}
            </div>

            <label className="mt-4 flex cursor-pointer items-start gap-3 text-sm">
              <input
                type="checkbox"
                checked={persist}
                onChange={(e) => setPersist(e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-accent"
              />
              <span className="text-ink-dim">
                Save the final analysis to History (uncheck for one-off checks — provisional
                live windows are never stored either way).
              </span>
            </label>

            {error && (
              <div className="mt-4">
                <Callout tone="warning" title="Recording problem">{error}</Callout>
              </div>
            )}
          </Card>

          {windows.length > 0 && (
            <Card
              title="Provisional window scores"
              subtitle="Each bar is the model's score for the audio recorded so far (a 4s checkpoint). Provisional — the final verdict uses the full clip."
            >
              <WindowScoresChart windows={chartWindows} />
              <div className="mt-4 grid grid-cols-2 gap-3 text-center md:grid-cols-4">
                <div className="rounded-md border border-line bg-surface-2 p-3">
                  <p className="text-[11px] uppercase tracking-wider text-ink-dim">Provisional risk</p>
                  <p className="mt-1 text-lg font-semibold text-ink">
                    {provisionalRisk !== null ? `${(provisionalRisk * 100).toFixed(1)}%` : '—'}
                  </p>
                  <p className="mt-0.5 text-[10px] text-ink-faint">calibrated, prefix only</p>
                </div>
                <div className="rounded-md border border-line bg-surface-2 p-3">
                  <p className="text-[11px] uppercase tracking-wider text-ink-dim">Checkpoints</p>
                  <p className="mt-1 text-lg font-semibold text-ink">{windows.length}</p>
                </div>
                <div className="rounded-md border border-line bg-surface-2 p-3">
                  <p className="text-[11px] uppercase tracking-wider text-ink-dim">Mean score</p>
                  <p className="mt-1 text-lg font-semibold text-ink">
                    {provisionalMean !== null ? provisionalMean.toFixed(2) : '—'}
                  </p>
                </div>
                <div className="rounded-md border border-line bg-surface-2 p-3">
                  <p className="text-[11px] uppercase tracking-wider text-ink-dim">Recorded</p>
                  <p className="mt-1 text-lg font-semibold text-ink">{elapsed}s</p>
                </div>
              </div>
            </Card>
          )}

          {final && (
            <Card title="Final analysis (full recording)">
              <AnalysisResultPanel record={final} />
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Callout tone="ethics" title="What live scores mean — and don't">
            Live window scores come from the same detection model as the full-clip verdict, over
            the audio recorded so far. They are evidence for a human reviewer, not a verdict:
            scores on short prefixes can be misleading, and the model only discriminates
            human vs Piper-TTS synthesis in its current training. It is not a general
            clone detector.
          </Callout>
          <Card title="Privacy while recording">
            <ul className="space-y-2.5 text-xs leading-relaxed text-ink-dim">
              <li className="flex items-start gap-2">
                <ShieldIcon width={14} height={14} className="mt-0.5 shrink-0 text-accent" />
                Audio is processed only for this analysis; provisional prefixes are computed and
                discarded, never stored.
              </li>
              <li className="flex items-start gap-2">
                <ShieldIcon width={14} height={14} className="mt-0.5 shrink-0 text-accent" />
                The final recording is saved to History only when you leave the box checked.
              </li>
              <li className="flex items-start gap-2">
                <ShieldIcon width={14} height={14} className="mt-0.5 shrink-0 text-accent" />
                Stopping the page or navigating away stops the microphone immediately.
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )
}
