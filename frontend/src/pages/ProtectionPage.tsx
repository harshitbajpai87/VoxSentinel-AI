// Family Emergency & Scam Protection — additive module.
// Everything here is new: the existing analysis workflow is untouched.
// Findings come only from the backend response; nothing is fabricated.
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Card,
  Button,
  Pill,
  StatTile,
  PageHeader,
  EmptyState,
  ErrorState,
  LoadingState,
  StatusDot,
  Tag,
} from '../components/ui'
import {
  AlertIcon,
  CheckIcon,
  DownloadIcon,
  RefreshIcon,
  ShieldIcon,
  UploadIcon,
  XIcon,
} from '../components/icons'
import { ApiError, protectionApi } from '../lib/api'
import type {
  FamilyVoiceProfile,
  ProtectionAnalysis,
  ProtectionAnalysisSummary,
} from '../lib/types'
import { ACCEPTED_EXTENSIONS_LABEL, DEFAULT_AUDIO_LIMITS, formatBytes } from '../lib/audio/validation'
import { RiskBadgeRow } from '../components/protection/RiskBadgeRow'
import { ThreatExplanationCard } from '../components/protection/ThreatExplanationCard'

type Stage = 'idle' | 'selected' | 'running' | 'done' | 'error'

const RISK_TONE: Record<string, 'ok' | 'warn' | 'err' | 'neutral' | 'info'> = {
  LOW: 'ok',
  MEDIUM: 'warn',
  HIGH: 'err',
  CRITICAL: 'err',
  INCONCLUSIVE: 'info',
}

function riskTone(category: string | undefined) {
  return RISK_TONE[(category ?? '').toUpperCase()] ?? 'neutral'
}

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  hi: 'Hindi',
  mr: 'Marathi',
  ta: 'Tamil',
  te: 'Telugu',
  bn: 'Bengali',
  gu: 'Gujarati',
  kn: 'Kannada',
  ml: 'Malayalam',
  pa: 'Punjabi',
  ur: 'Urdu',
  or: 'Odia',
  as: 'Assamese',
}

/** Whisper reports short codes ("hi"); people read names ("Hindi"). */
function languageName(code: string | null | undefined) {
  if (!code) return '—'
  const base = code.split(/[-_]/)[0].toLowerCase()
  return LANGUAGE_NAMES[base] ?? base.toUpperCase()
}

function timeOf(iso: string) {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export default function ProtectionPage() {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [consent, setConsent] = useState(false)
  const [profiles, setProfiles] = useState<FamilyVoiceProfile[]>([])
  const [profileId, setProfileId] = useState('')
  const [stage, setStage] = useState<Stage>('idle')
  const [failure, setFailure] = useState<string | null>(null)
  const [result, setResult] = useState<ProtectionAnalysis | null>(null)
  const [history, setHistory] = useState<ProtectionAnalysisSummary[] | null>(null)
  const [historyError, setHistoryError] = useState(false)
  const [newProfileName, setNewProfileName] = useState('')
  const [newProfileRelation, setNewProfileRelation] = useState('son')
  const [newProfileFile, setNewProfileFile] = useState<File | null>(null)
  const [profileBusy, setProfileBusy] = useState(false)
  const [profileMsg, setProfileMsg] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  // Text-only path: the same content, LLM and risk layers, no audio needed.
  const [inputMode, setInputMode] = useState<'audio' | 'text'>('audio')
  const [pastedText, setPastedText] = useState('')
  const profileInputRef = useRef<HTMLInputElement | null>(null)

  const loadHistory = useCallback(async () => {
    setHistoryError(false)
    try {
      const res = await protectionApi.listAnalyses(50)
      setHistory(res.analyses)
    } catch {
      setHistoryError(true)
    }
  }, [])

  const loadProfiles = useCallback(async () => {
    try {
      const res = await protectionApi.listProfiles()
      setProfiles(res.profiles)
    } catch {
      setProfiles([])
    }
  }, [])

  useEffect(() => {
    void loadHistory()
    void loadProfiles()
  }, [loadHistory, loadProfiles])

  const pickFile = (f: File | null) => {
    if (!f) return
    setResult(null)
    setFailure(null)
    setFile(f)
    setStage('selected')
  }

  const runAnalysis = async () => {
    if (!file || !consent) return
    setStage('running')
    setFailure(null)
    setResult(null)
    try {
      const res = await protectionApi.analyze(file, { consent, profileId: profileId || null })
      setResult(res)
      setStage('done')
      void loadHistory()
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : 'The analysis could not be completed.'
      setFailure(message)
      setStage('error')
    }
  }

  const runTextAnalysis = async () => {
    if (!consent || pastedText.trim().length < 10) return
    setStage('running')
    setFailure(null)
    setResult(null)
    try {
      const res = await protectionApi.analyzeText(pastedText.trim(), {
        consent,
        profileId: profileId || null,
      })
      setResult(res)
      setStage('done')
      void loadHistory()
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : 'The analysis could not be completed.'
      setFailure(message)
      setStage('error')
    }
  }

  const addProfile = async () => {
    if (!newProfileFile || !newProfileName.trim()) return
    setProfileBusy(true)
    setProfileMsg(null)
    try {
      await protectionApi.createProfile(newProfileName.trim(), newProfileRelation, newProfileFile)
      setProfileMsg('Family voice profile saved.')
      setNewProfileName('')
      setNewProfileFile(null)
      void loadProfiles()
    } catch (err) {
      setProfileMsg(err instanceof Error ? err.message : 'Could not save the profile.')
    } finally {
      setProfileBusy(false)
    }
  }

  const removeProfile = async (id: string) => {
    try {
      await protectionApi.deleteProfile(id)
      if (profileId === id) setProfileId('')
      void loadProfiles()
    } catch {
      setProfileMsg('Could not delete the profile.')
    }
  }

  const removeAnalysis = async (id: string) => {
    try {
      await protectionApi.deleteAnalysis(id)
      if (result?.id === id) setResult(null)
      void loadHistory()
    } catch {
      setHistoryError(true)
    }
  }

  const openAnalysis = async (id: string) => {
    try {
      const res = await protectionApi.getAnalysis(id)
      setResult(res)
      setStage('done')
      setFile(null)
    } catch {
      setHistoryError(true)
    }
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Family Emergency & Scam Protection"
        subtitle="Analyze suspicious call recordings for AI voice cloning, impersonation, and payment fraud using transcription, voice verification, and pattern analysis."
        actions={<Tag>Protection</Tag>}
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card
            title="1 · Provide the suspicious call"
            subtitle={
              inputMode === 'audio'
                ? `Audio, up to ${Math.round(
                    DEFAULT_AUDIO_LIMITS.maxFileSizeBytes / (1024 * 1024),
                  )} MB; ${ACCEPTED_EXTENSIONS_LABEL} and similar formats.`
                : 'Paste a transcript to check the content layers without audio. Voice authenticity and speaker match cannot be assessed without a recording.'
            }
          >
            <div className="mb-4 inline-flex rounded-lg border border-line p-0.5" role="tablist" aria-label="Input type">
              {(['audio', 'text'] as const).map((mode) => (
                <button
                  key={mode}
                  role="tab"
                  aria-selected={inputMode === mode}
                  data-testid={`input-mode-${mode}`}
                  onClick={() => {
                    setInputMode(mode)
                    setResult(null)
                    setFailure(null)
                    setStage(mode === 'audio' ? (file ? 'selected' : 'idle') : 'idle')
                  }}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    inputMode === mode ? 'bg-accent/10 text-accent' : 'text-ink-dim hover:text-ink'
                  }`}
                >
                  {mode === 'audio' ? 'Call recording' : 'Pasted transcript'}
                </button>
              ))}
            </div>
            {inputMode === 'text' ? (
              <div>
                <label htmlFor="protection-transcript" className="mb-1 block text-xs font-medium text-ink-faint">
                  Transcript to check
                </label>
                <textarea
                  id="protection-transcript"
                  data-testid="protection-transcript"
                  value={pastedText}
                  onChange={(e) => setPastedText(e.target.value)}
                  rows={7}
                  placeholder="Paste what the caller said. English, Hindi or Hinglish."
                  className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink"
                />
                <p className="mt-1 text-xs text-ink-faint">
                  {pastedText.trim().length} characters · content, scam intent, threat reasoning and risk are all computed
                </p>
              </div>
            ) : (
            <div
              role="button"
              tabIndex={0}
              aria-label="Upload a call recording"
              data-testid="protection-dropzone"
              onDragOver={(e) => {
                e.preventDefault()
                setDragActive(true)
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragActive(false)
                pickFile(e.dataTransfer.files?.[0] ?? null)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
              }}
              onClick={() => inputRef.current?.click()}
              className={`cursor-pointer rounded-xl border border-dashed px-6 py-10 text-center transition-colors ${
                dragActive ? 'border-accent bg-accent/5' : 'border-line hover:border-accent/60'
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.mp4,.aac,.flac,.ogg,.oga,.opus,.webm,.aiff,.aif,.caf,.wma,.amr,.mka,.3gp,.spx"
                className="sr-only"
                onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
              />
              <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full border border-accent/25 bg-accent/10 text-accent">
                <UploadIcon width={20} height={20} />
              </div>
              <p className="mt-3 text-sm font-medium text-ink">
                {file ? file.name : 'Drop a call recording here, or click to browse'}
              </p>
              <p className="mt-1 text-xs text-ink-faint">
                {file
                  ? `${formatBytes(file.size)} · ready to analyze`
                  : 'Nothing is uploaded until you start the analysis'}
              </p>
            </div>
            )}

            <div className="mt-5 space-y-4">
              <label className="flex items-start gap-2.5 text-sm text-ink-dim">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-accent"
                />
                <span>I confirm I am authorized to analyze this recording.</span>
              </label>

              <div className="flex flex-wrap items-end gap-3">
                <label className="text-sm text-ink-dim">
                  <span className="mb-1 block text-xs font-medium text-ink-faint">
                    Compare with a family voice profile (optional)
                  </span>
                  <select
                    value={profileId}
                    onChange={(e) => setProfileId(e.target.value)}
                    className="w-64 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink"
                  >
                    <option value="">No comparison</option>
                    {profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({p.relationship})
                      </option>
                    ))}
                  </select>
                </label>
                <Button
                  onClick={() => void (inputMode === 'audio' ? runAnalysis() : runTextAnalysis())}
                  disabled={
                    !consent ||
                    stage === 'running' ||
                    (inputMode === 'audio' ? !file : pastedText.trim().length < 10)
                  }
                >
                  <ShieldIcon width={16} height={16} />
                  {stage === 'running'
                    ? 'Analyzing…'
                    : inputMode === 'audio'
                      ? 'Analyze call'
                      : 'Analyze transcript'}
                </Button>
                {file && (
                  <Button
                    variant="ghost"
                    onClick={() => {
                      setFile(null)
                      setStage('idle')
                      setResult(null)
                    }}
                  >
                    Clear
                  </Button>
                )}
              </div>
            </div>
          </Card>

          {stage === 'running' && (
            <Card title="Analysis in progress">
              <LoadingState label="Transcribing audio, running the voice detector and scam analysis…" />
              <ul className="mt-4 space-y-2 text-sm text-ink-dim">
                {[
                  'Audio validation',
                  'Speech transcription (multilingual)',
                  'Voice authenticity (existing detector)',
                  'Family voice comparison',
                  'Scam & ransom content analysis',
                  'Risk assessment',
                ].map((s) => (
                  <li key={s} className="flex items-center gap-2">
                    <StatusDot tone="info" pulse />
                    {s}
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {stage === 'error' && failure && (
            <ErrorState
              title="Analysis failed"
              description={failure}
              onRetry={() => void runAnalysis()}
            />
          )}

          {result && <ResultPanel result={result} onExport={() => window.open(protectionApi.reportUrl(result.id), '_blank')} />}
        </div>

        <div className="space-y-6">
          <Card
            title="Family voice profiles"
            subtitle="Enroll consented reference audio to enable optional family voice matching."
          >
            {profiles.length === 0 ? (
              <p className="text-xs leading-relaxed text-ink-faint">No profiles enrolled yet.</p>
            ) : (
              <ul className="space-y-2">
                {profiles.map((p) => (
                  <li
                    key={p.id}
                    className="flex items-center justify-between gap-2 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm"
                  >
                    <span className="min-w-0 truncate text-ink">
                      {p.name} <span className="text-ink-faint">· {p.relationship}</span>
                    </span>
                    <button
                      onClick={() => void removeProfile(p.id)}
                      aria-label={`Delete profile ${p.name}`}
                      className="rounded p-1 text-ink-faint hover:bg-danger/10 hover:text-danger"
                    >
                      <XIcon width={14} height={14} />
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <div className="mt-4 space-y-2 border-t border-line/70 pt-4">
              <input
                value={newProfileName}
                onChange={(e) => setNewProfileName(e.target.value)}
                placeholder="Name (e.g. Aarav)"
                className="w-full rounded-lg border border-input bg-surface px-3 py-2 text-sm text-ink"
              />
              <select
                value={newProfileRelation}
                onChange={(e) => setNewProfileRelation(e.target.value)}
                className="w-full rounded-lg border border-input bg-surface px-3 py-2 text-sm text-ink"
              >
                {['son', 'daughter', 'parent', 'spouse', 'sibling', 'other'].map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
              <input
                ref={profileInputRef}
                type="file"
                accept="audio/*"
                className="sr-only"
                onChange={(e) => setNewProfileFile(e.target.files?.[0] ?? null)}
              />
              <Button
                variant="secondary"
                className="w-full"
                onClick={() => profileInputRef.current?.click()}
              >
                {newProfileFile ? newProfileFile.name : 'Choose reference audio'}
              </Button>
              <Button
                className="w-full"
                disabled={profileBusy || !newProfileName.trim() || !newProfileFile}
                onClick={() => void addProfile()}
              >
                {profileBusy ? 'Saving…' : 'Enroll profile (consent required)'}
              </Button>
              {profileMsg && <p className="text-xs text-ink-faint">{profileMsg}</p>}
            </div>
          </Card>

          <Card
            title="Analysis history"
            subtitle="Saved to the backend; transcripts follow the retention policy."
            actions={
              <Button variant="ghost" onClick={() => void loadHistory()} aria-label="Refresh history">
                <RefreshIcon width={14} height={14} />
              </Button>
            }
          >
            {historyError && <p className="text-xs text-danger">Could not load history.</p>}
            {!historyError && history === null && <LoadingState label="Loading…" />}
            {!historyError && history?.length === 0 && (
              <EmptyState title="No protection analyses yet" description="Results will appear here after the first call is analyzed." />
            )}
            {history && history.length > 0 && (
              <ul className="divide-y divide-line/60">
                {history.map((h) => (
                  <li key={h.id} className="flex items-center justify-between gap-2 py-2.5 text-sm">
                    <button
                      onClick={() => void openAnalysis(h.id)}
                      className="min-w-0 flex-1 text-left hover:text-accent"
                    >
                      <span className="block truncate text-ink">{h.display_name}</span>
                      <span className="block text-[11px] text-ink-faint">{timeOf(h.created_at)}</span>
                    </button>
                    <Pill tone={riskTone(h.risk_category)}>{h.risk_category}</Pill>
                    <button
                      onClick={() => void removeAnalysis(h.id)}
                      aria-label={`Delete analysis ${h.display_name}`}
                      className="rounded p-1 text-ink-faint hover:bg-danger/10 hover:text-danger"
                    >
                      <XIcon width={14} height={14} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}

function ResultPanel({ result, onExport }: { result: ProtectionAnalysis; onExport: () => void }) {
  // Provider-neutral: `llm` for current records, `gemini` for older ones.
  const ai = result.ai.llm ?? result.ai.gemini
  const aiLabel = result.ai.provider_label ?? 'Gemini'
  const risk = result.risk
  return (
    <div className="space-y-6">
      <Card
      title="Analysis result"
      subtitle={`${result.display_name} · ${timeOf(result.created_at)} · ${result.processing_ms} ms`}
      actions={
        <Button variant="secondary" onClick={onExport}>
          <DownloadIcon width={14} height={14} /> JSON report
        </Button>
      }
    >
      <div className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          <Pill tone={riskTone(risk.category)}>{risk.category} risk</Pill>
          <p className="text-sm font-medium text-ink">{result.content.summary}</p>
        </div>

        {result.risk_badges?.length ? <RiskBadgeRow badges={result.risk_badges} /> : null}

        <div className="rounded-xl border border-line bg-surface-2/40 p-4">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile
              label="Voice authenticity"
            value={
              result.voice.suspected_synthetic === null
                ? '—'
                : result.voice.suspected_synthetic
                  ? 'Synthetic signs'
                  : 'No strong signs'
            }
            hint={
              result.voice.calibrated_probability != null
                ? `Calibrated detector probability ${(result.voice.calibrated_probability * 100).toFixed(1)}%`
                : typeof result.voice.score === 'number'
                  ? `${result.voice.score_label ?? 'Raw detector score'} ${result.voice.score.toFixed(3)} · ${result.voice.score_definition ?? 'not a calibrated probability'}`
                  : result.voice.status
            }
            tone={result.voice.suspected_synthetic ? 'err' : result.voice.status === 'completed' ? 'ok' : 'neutral'}
          />
          <StatTile
            label="Detected language"
            value={languageName(result.transcript.language)}
            hint={result.transcript.language_probability !== null ? `${(result.transcript.language_probability * 100).toFixed(0)}% confidence` : 'not reported'}
          />
          <StatTile
            label="Content analysis"
            value={result.content.primary_category?.replace(/_/g, ' ') ?? 'no findings'}
            hint={
              result.content.categories.length
                ? `${result.content.categories.length} categories · strength ${result.content.signal_strength}`
                : 'no scam indicators'
            }
            tone={result.content.signal_strength >= 6 ? 'err' : result.content.signal_strength >= 3 ? 'warn' : 'ok'}
          />
          <StatTile
            label="Speaker match"
            value={
              result.speaker_verification
                ? result.speaker_verification.match === true
                  ? 'Match'
                  : result.speaker_verification.match === false
                    ? 'Mismatch'
                    : 'Not verified'
                : result.speaker_comparison
                  ? result.speaker_comparison.outcome.replace(/_/g, ' ')
                  : '—'
            }
            hint={
              result.speaker_verification?.similarity != null
                ? `similarity ${result.speaker_verification.similarity.toFixed(3)}${
                    result.speaker_verification.threshold != null
                      ? ` · threshold ${result.speaker_verification.threshold.toFixed(2)}`
                      : ''
                  }`
                : (result.speaker_verification?.detail ?? 'no family reference supplied')
            }
            tone={
              result.speaker_verification?.match === true
                ? 'ok'
                : result.speaker_verification?.match === false
                  ? 'err'
                  : 'neutral'
            }
          />
          </div>
        </div>

        {/* {aiLabel} contextual analysis — clearly labelled, with honest fallback status */}
        <div className="rounded-lg border border-line bg-surface p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">
              {aiLabel} contextual analysis
            </p>
            <Pill
              tone={
                ai.status === 'completed'
                  ? 'ok'
                  : ai.status === 'failed'
                    ? 'err'
                    : 'warn'
              }
            >
              {ai.status === 'completed'
                ? `${aiLabel} used${ai.model ? ` · ${ai.model}` : ''}`
                : ai.status === 'failed'
                  ? `${aiLabel} failed · rule-based fallback`
                  : `${aiLabel} not configured · rule-based fallback`}
            </Pill>
          </div>

          {ai.status === 'completed' && ai.result ? (
            <div className="mt-3 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Pill
                  tone={
                    ai.result.classification === 'LIKELY_SCAM'
                      ? 'err'
                      : ai.result.classification === 'SUSPICIOUS'
                        ? 'warn'
                        : ai.result.classification === 'INCONCLUSIVE'
                          ? 'neutral'
                          : 'ok'
                  }
                >
                  {ai.result.classification.replace(/_/g, ' ')}
                </Pill>
                <span className="text-sm text-ink">{ai.result.summary}</span>
              </div>

              {ai.result.scam_categories.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {ai.result.scam_categories.map((c) => (
                    <span key={c} className="rounded-sm border border-line bg-surface-2 px-2 py-0.5 text-[11px] text-ink-dim">
                      {c.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              )}

              {ai.result.indicators.length > 0 && (
                <ul className="space-y-2">
                  {ai.result.indicators.map((ind, i) => (
                    <li key={i} className="rounded-lg border border-line px-3 py-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Pill tone={ind.severity === 'CRITICAL' || ind.severity === 'HIGH' ? 'err' : 'warn'}>
                          {ind.severity}
                        </Pill>
                        <span className="text-xs font-medium text-ink-dim">
                          {ind.category.replace(/_/g, ' ')}
                        </span>
                      </div>
                      <blockquote className="mt-1 border-l-2 border-magenta/40 pl-2 text-sm italic text-ink">
                        “{ind.evidence_quote}”
                      </blockquote>
                      <p className="mt-1 text-xs text-ink-faint">{ind.explanation}</p>
                    </li>
                  ))}
                </ul>
              )}

              {ai.result.manipulation_tactics.length > 0 && (
                <p className="text-xs text-ink-dim">
                  <span className="font-medium text-ink">Manipulation tactics:</span>{' '}
                  {ai.result.manipulation_tactics.join(', ')}
                </p>
              )}
              {ai.result.uncertainty_reasons.length > 0 && (
                <p className="text-xs text-ink-faint">
                  <span className="font-medium text-ink-dim">Uncertainty:</span>{' '}
                  {ai.result.uncertainty_reasons.join('; ')}
                </p>
              )}
              {ai.result.protective_recommendations.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-ink-dim">Model recommendations</p>
                  <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs text-ink-faint">
                    {ai.result.protective_recommendations.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
              {!!ai.dropped_quotes && (
                <p className="text-xs text-warning">
                  {ai.dropped_quotes} model quote(s) could not be traced to the
                  transcript and were discarded.
                </p>
              )}
            </div>
          ) : (
            <p className="mt-2 text-sm text-ink-dim">
              {ai.detail ??
                'Gemini contextual analysis was not used for this analysis.'}
              {' '}The assessment below relies on the transparent rule-based baseline and the
              voice-authenticity detector.
            </p>
          )}
        </div>

        {result.speaker_comparison && (
          <div className="rounded-lg border border-line bg-surface-2 p-4 text-sm">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">Family voice comparison</p>
            <p className="mt-1 text-ink">
              {result.speaker_comparison.outcome.replace(/_/g, ' ')}
              {result.speaker_comparison.similarity !== null &&
                ` · raw cosine similarity ${result.speaker_comparison.similarity.toFixed(3)}`}
            </p>
            <p className="mt-1 text-xs text-ink-faint">{result.speaker_comparison.detail}</p>
          </div>
        )}

        {result.content.categories.length > 0 && (
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">Detected indicators</p>
            <ul className="mt-2 space-y-2">
              {result.content.categories.map((c, i) => (
                <li key={i} className="rounded-lg border border-line px-3 py-2 text-sm">
                  <span className="font-medium text-ink">{c.label}</span>
                  <span className="text-ink-dim"> — {c.category.replace(/_/g, ' ')}</span>
                  {c.matches.length > 0 && (
                    <p className="mt-0.5 text-xs text-ink-faint">
                      Matched phrases: {c.matches.slice(0, 3).map((m) => `“${m.phrase}”`).join(', ')}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="rounded-lg border border-warning/40 bg-warning/10 p-4">
          <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-warning">
            <AlertIcon width={14} height={14} /> Recommended actions
          </p>
          <ul className="mt-2 space-y-1.5 text-sm text-ink">
            {risk.recommended_actions.map((a, i) => (
              <li key={i} className="flex gap-2">
                <CheckIcon width={14} height={14} className="mt-0.5 shrink-0 text-warning" />
                {a}
              </li>
            ))}
          </ul>
        </div>

      </div>
    </Card>

      {result.explanation ? <ThreatExplanationCard explanation={result.explanation} /> : null}
    </div>
  )
}
