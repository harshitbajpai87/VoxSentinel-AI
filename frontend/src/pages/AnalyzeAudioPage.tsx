import { useEffect, useRef, useState } from 'react'
import {
  analyzeAudio,
  ApiError,
  ApiUnavailableError,
  mapAnalyzeError,
} from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import type { AnalysisRecord } from '../lib/types'
import {
  analyzeDecodedAudio,
  checkSelectedFile,
  computePeaks,
  formatBytes,
  formatTime,
  type AudioQualityReport,
  type WaveformPeaks,
} from '../lib/audio/validation'
import { decodeAudioFile, decodeSupportHint } from '../lib/audio/decode'
import { extensionOf } from '../lib/audio/validation'
import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { AudioDropzone } from '../components/audio/AudioDropzone'
import { WaveformCanvas } from '../components/audio/WaveformCanvas'
import { AnalysisResultPanel } from '../components/audio/AnalysisResultPanel'
import {
  Button,
  Callout,
  Card,
  ErrorState,
  LoadingState,
  PageHeader,
  Pill,
} from '../components/ui'
import {
  CheckIcon,
  PauseIcon,
  PlayIcon,
  SparkIcon,
  XIcon,
} from '../components/icons'

/* ------------------------------------------------------------------ */
/* Local state types                                                   */
/* ------------------------------------------------------------------ */

type SelectionStatus = 'decoding' | 'selection-error' | 'decode-error' | 'validated'

interface FileSelection {
  file: File
  key: string
  url: string
  status: SelectionStatus
  error: string | null
  /** Non-fatal notices from the selection checks (e.g. odd MIME type). */
  notices: string[]
  quality: AudioQualityReport | null
  peaks: WaveformPeaks | null
}

type AnalysisState =
  | { status: 'idle' }
  | { status: 'submitting'; progress: number | null }
  | { status: 'completed'; record: AnalysisRecord }
  | { status: 'failed'; message: string }

const OUTCOMES = [
  {
    label: 'Potentially synthetic speech detected',
    tone: 'err' as const,
    text: 'Model evidence suggests synthesis. This is a risk signal to investigate, never proof.',
  },
  {
    label: 'No strong synthetic-speech evidence detected',
    tone: 'ok' as const,
    text: 'No suspicious evidence was found. This is NOT proof that the recording is genuine.',
  },
  {
    label: 'Inconclusive',
    tone: 'info' as const,
    text: 'Audio quality, duration or conflicting evidence prevents a defensible assessment.',
  },
  {
    label: 'Unable to analyze',
    tone: 'neutral' as const,
    text: 'The file failed validation (silent, corrupt, too short, unsupported) and was not processed.',
  },
]

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function AnalyzeAudioPage() {
  const { status: connection, health, refresh } = useApiStatus()

  const [selection, setSelection] = useState<FileSelection | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisState>({ status: 'idle' })
  const [consent, setConsent] = useState(false)
  const [analyzedKeys, setAnalyzedKeys] = useState<Set<string>>(new Set())

  // Revoke the object URL on unmount (the ref always holds the latest URL).
  const urlRef = useRef<string | null>(null)
  useEffect(() => {
    urlRef.current = selection?.url ?? null
  }, [selection])
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    },
    [],
  )

  const player = useAudioPlayer(
    selection?.status === 'validated' ? selection.url : null,
  )

  /* ---------- selection & validation ---------- */

  const selectFile = async (file: File) => {
    // One selection at a time; release the previous object URL.
    if (selection) URL.revokeObjectURL(selection.url)
    setAnalysis({ status: 'idle' })

    const check = checkSelectedFile(file)
    const url = URL.createObjectURL(file)

    if (!check.ok) {
      setSelection({
        file,
        key: check.fileKey,
        url,
        status: 'selection-error',
        error: check.errorMessage,
        notices: [],
        quality: null,
        peaks: null,
      })
      return
    }

    const notices = [...check.warnings]
    const support = decodeSupportHint(extensionOf(file.name))
    if (support.message) notices.push(support.message)

    setSelection({
      file,
      key: check.fileKey,
      url,
      status: 'decoding',
      error: null,
      notices,
      quality: null,
      peaks: null,
    })

    try {
      const buffer = await decodeAudioFile(file)
      const quality = analyzeDecodedAudio(buffer)
      const peaks = computePeaks(buffer.getChannelData(0), 400)
      setSelection((prev) =>
        prev && prev.key === check.fileKey
          ? { ...prev, status: 'validated', quality, peaks }
          : prev,
      )
    } catch (err) {
      setSelection((prev) =>
        prev && prev.key === check.fileKey
          ? {
              ...prev,
              status: 'decode-error',
              error:
                err instanceof Error
                  ? err.message
                  : 'The browser could not decode this file as audio.',
            }
          : prev,
      )
    }
  }

  const removeFile = () => {
    if (selection) URL.revokeObjectURL(selection.url)
    setSelection(null)
    setAnalysis({ status: 'idle' })
    setConsent(false)
  }

  /* ---------- analysis ---------- */

  const backendReady = connection === 'online'
  const modelReady = backendReady && (health?.model_ready ?? false)
  const blockers = selection?.status === 'validated' ? selection.quality?.blockers ?? [] : []
  const alreadyAnalyzed = selection ? analyzedKeys.has(selection.key) : false

  const canAnalyze =
    selection?.status === 'validated' &&
    consent &&
    blockers.length === 0 &&
    backendReady &&
    modelReady &&
    !alreadyAnalyzed &&
    analysis.status !== 'submitting'

  const startAnalysis = async () => {
    if (!selection || !canAnalyze) return
    setAnalysis({ status: 'submitting', progress: null })
    try {
      // analyzeAudio() prepares the upload: the ORIGINAL file is sent when the
      // backend can decode it, and only converted in the browser when it cannot.
      const record = await analyzeAudio(selection.file, {
        onProgress: ({ loaded, total }) =>
          setAnalysis({
            status: 'submitting',
            progress: total > 0 ? loaded / total : null,
          }),
      })
      setAnalysis({ status: 'completed', record })
      setAnalyzedKeys((prev) => new Set(prev).add(selection.key))
    } catch (err) {
      let message: string
      if (err instanceof ApiUnavailableError) message = err.message
      else if (err instanceof ApiError) message = mapAnalyzeError(err.status, err.code, err.message)
      else message = 'The analysis request failed unexpectedly.'
      setAnalysis({ status: 'failed', message })
    }
  }

  /* ---------- derived view values ---------- */

  const durationS =
    selection?.quality?.durationS && selection.quality.durationS > 0
      ? selection.quality.durationS
      : player.durationS
  const progress = durationS > 0 ? player.currentTimeS / durationS : 0

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Analyze Audio"
        subtitle="Upload a clip, inspect it locally, then send it for synthetic-speech analysis."
        actions={
          backendReady ? (
            modelReady ? (
              <Pill tone="ok">
                <CheckIcon width={13} height={13} /> Backend &amp; model ready
              </Pill>
            ) : (
              <Pill tone="warn">Backend online · model not ready</Pill>
            )
          ) : (
            <Pill tone="err">Backend offline — local preview only</Pill>
          )
        }
      />

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Left: upload, inspection, analysis */}
        <div className="space-y-6 lg:col-span-3">
          <Card
            title="Upload & inspect"
            subtitle="Decoding and preview happen entirely in your browser."
            actions={
              selection ? (
                <Button variant="ghost" onClick={removeFile}>
                  <XIcon width={14} height={14} /> Remove file
                </Button>
              ) : undefined
            }
          >
            {!selection ? (
              <AudioDropzone onFileSelected={(f) => void selectFile(f)} />
            ) : (
              <div className="space-y-4">
                {/* File summary */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="max-w-full truncate text-sm font-medium text-ink">
                    {selection.file.name}
                  </span>
                  <span className="text-xs text-ink-faint">
                    {formatBytes(selection.file.size)}
                  </span>
                  {selection.status === 'decoding' && <Pill tone="info">Decoding…</Pill>}
                  {selection.status === 'validated' && (
                    <Pill tone="ok">
                      <CheckIcon width={13} height={13} /> Decoded &amp; validated
                    </Pill>
                  )}
                  {(selection.status === 'selection-error' ||
                    selection.status === 'decode-error') && <Pill tone="err">Invalid</Pill>}
                </div>

                {selection.status === 'decoding' && (
                  <LoadingState label="Decoding audio in your browser…" />
                )}

                {selection.status === 'selection-error' && (
                  <ErrorState title="File not accepted" description={selection.error ?? undefined} />
                )}
                {selection.status === 'decode-error' && (
                  <ErrorState title="Could not decode audio" description={selection.error ?? undefined} />
                )}

                {selection.status === 'validated' &&
                  selection.quality &&
                  selection.peaks && (
                    <>
                      {/* Waveform + playback */}
                      <div className="rounded-lg border border-line bg-canvas/60 p-3">
                        <WaveformCanvas
                          peaks={selection.peaks}
                          progress={progress}
                          onSeek={player.seekToFraction}
                          height={110}
                        />
                        <div className="mt-3 flex items-center gap-3">
                          <button
                            type="button"
                            onClick={player.toggle}
                            aria-label={player.playing ? 'Pause' : 'Play'}
                            className="flex h-9 w-9 items-center justify-center rounded-full border border-accent/40 bg-accent/10 text-accent transition-colors hover:bg-accent/20"
                          >
                            {player.playing ? (
                              <PauseIcon width={15} height={15} />
                            ) : (
                              <PlayIcon width={15} height={15} />
                            )}
                          </button>
                          <span className="font-mono text-xs tabular-nums text-ink-dim">
                            {formatTime(player.currentTimeS)} / {formatTime(durationS)}
                          </span>
                          <span className="ml-auto text-[11px] text-ink-faint">
                            Click the waveform to seek
                          </span>
                        </div>
                      </div>

                      {/* Metadata */}
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                        {[
                          ['Duration', `${selection.quality.durationS.toFixed(1)} s`],
                          ['Sample rate', `${selection.quality.sampleRateHz.toLocaleString()} Hz`],
                          ['Channels', String(selection.quality.channels)],
                          ['Peak level', selection.quality.peak.toFixed(3)],
                          ['RMS level', selection.quality.rms.toFixed(4)],
                        ].map(([label, value]) => (
                          <div
                            key={label}
                            className="rounded-md border border-line bg-surface-2/40 px-3 py-2"
                          >
                            <p className="text-[10px] uppercase tracking-wider text-ink-dim">
                              {label}
                            </p>
                            <p className="mt-1 font-mono text-sm text-ink">{value}</p>
                          </div>
                        ))}
                      </div>

                      {/* Quality findings */}
                      {blockers.length > 0 && (
                        <Callout tone="warning" title="Unable to analyze this recording">
                          <ul className="space-y-1">
                            {blockers.map((b) => (
                              <li key={b}>· {b}</li>
                            ))}
                          </ul>
                          <p className="mt-2">
                            You can still preview the audio above, but analysis would be
                            meaningless. Remove the file and select a usable recording.
                          </p>
                        </Callout>
                      )}
                      {selection.quality.warnings.length > 0 && (
                        <Callout tone="info" title="Quality warnings">
                          <ul className="space-y-1">
                            {selection.quality.warnings.map((w) => (
                              <li key={w}>· {w}</li>
                            ))}
                          </ul>
                        </Callout>
                      )}
                      {selection.notices.length > 0 && (
                        <ul className="space-y-1 text-[11px] text-ink-faint">
                          {selection.notices.map((n) => (
                            <li key={n}>· {n}</li>
                          ))}
                        </ul>
                      )}
                    </>
                  )}
              </div>
            )}
          </Card>

          {/* Start analysis */}
          <Card
            title="Start analysis"
            subtitle="Client-side validation is a convenience filter. The backend independently re-validates everything."
          >
            <div className="space-y-4" aria-live="polite">
              {!selection && (
                <p className="text-xs text-ink-faint">Select and validate a file first.</p>
              )}

              {selection?.status === 'decoding' && (
                <p className="text-xs text-ink-dim">Waiting for the file to finish decoding…</p>
              )}

              {selection &&
                (selection.status === 'selection-error' ||
                  selection.status === 'decode-error') && (
                  <p className="text-xs text-ink-dim">
                    This file could not be validated, so it cannot be analyzed.
                  </p>
                )}

              {selection?.status === 'validated' && blockers.length > 0 && (
                <p className="text-xs text-danger">
                  Unable to analyze: {blockers[0]} Remove the file to select another recording.
                </p>
              )}

              {selection?.status === 'validated' &&
                blockers.length === 0 &&
                !backendReady && (
                  <p className="text-xs text-warning">
                    The backend is offline. Local preview keeps working; detection will be
                    available once the Python backend runs (Phase 4) at{' '}
                    <span className="font-mono">VITE_API_BASE_URL</span>.
                  </p>
                )}

              {selection?.status === 'validated' &&
                blockers.length === 0 &&
                backendReady &&
                !modelReady && (
                  <p className="text-xs text-warning">
                    The backend is reachable, but its health check reports the detection model
                    as not ready. Analysis is disabled rather than returning fake predictions.
                  </p>
                )}

              {alreadyAnalyzed && (
                <p className="text-xs text-accent">
                  This exact file was already analyzed in this session — see the result below.
                  Remove the file to analyze a different recording.
                </p>
              )}

              {/* Progress while submitting */}
              {analysis.status === 'submitting' && (
                <div>
                  <p className="mb-2 text-xs text-ink-dim">
                    Uploading and analyzing
                    {analysis.progress !== null
                      ? ` — ${Math.round(analysis.progress * 100)}%`
                      : '…'}
                  </p>
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
                    <div
                      className="h-full rounded-full bg-accent transition-all"
                      style={{
                        width:
                          analysis.progress !== null
                            ? `${Math.max(5, analysis.progress * 100)}%`
                            : '100%',
                        opacity: analysis.progress !== null ? 1 : 0.35,
                      }}
                    />
                  </div>
                </div>
              )}

              {analysis.status === 'failed' && (
                <ErrorState
                  title="Analysis request failed"
                  description={analysis.message}
                  onRetry={() => void startAnalysis()}
                />
              )}

              <label className="flex cursor-pointer items-start gap-3 text-sm">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                  disabled={!selection || selection.status !== 'validated'}
                  className="mt-0.5 h-4 w-4 accent-accent"
                />
                <span className="text-ink-dim">
                  I am authorized to analyze this recording and consent to it being sent to
                  the backend for analysis. It is deleted from server storage afterwards.
                </span>
              </label>

              <div>
                <Button disabled={!canAnalyze} onClick={() => void startAnalysis()}>
                  <SparkIcon width={15} height={15} />
                  {analysis.status === 'submitting' ? 'Analyzing…' : 'Start analysis'}
                </Button>
                {analysis.status === 'idle' && !canAnalyze && selection?.status === 'validated' && (
                  <span className="ml-3 text-[11px] text-ink-faint">
                    {!consent && blockers.length === 0 && backendReady && modelReady
                      ? 'Tick the consent box to enable analysis.'
                      : undefined}
                  </span>
                )}
              </div>
            </div>
          </Card>

          {/* Real result — only after a real backend response */}
          {analysis.status === 'completed' && (
            <AnalysisResultPanel record={analysis.record} />
          )}
        </div>

        {/* Right: result vocabulary (unchanged from Phase 2) */}
        <div className="space-y-6 lg:col-span-2">
          <Card
            title="Possible outcomes"
            subtitle="Four categories only. No result is ever worded as definitive proof."
          >
            <ul className="space-y-3">
              {OUTCOMES.map((o) => (
                <li key={o.label} className="rounded-md border border-line bg-surface/40 p-3">
                  <Pill tone={o.tone}>{o.label}</Pill>
                  <p className="mt-2 text-xs leading-relaxed text-ink-dim">{o.text}</p>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )
}
