import { useRef, useState } from 'react'
import { verifySpeakers } from '../lib/api'
import { useApiStatus } from '../lib/apiStatus'
import { checkSelectedFile } from '../lib/audio/validation'
import type { SpeakerVerifyResponse } from '../lib/types'
import {
  Button,
  Callout,
  Card,
  PageHeader,
  Pill,
  Tag,
} from '../components/ui'
import { FingerprintIcon, UploadIcon } from '../components/icons'

function UploadSlot({
  title,
  description,
  file,
  onPick,
  accept,
  disabled,
}: {
  title: string
  description: string
  file: File | null
  onPick: (f: File) => void
  accept: string
  disabled?: boolean
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [err, setErr] = useState<string | null>(null)
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line bg-surface/40 px-5 py-9 text-center">
      <FingerprintIcon width={20} height={20} className="text-magenta" />
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="max-w-xs text-xs leading-relaxed text-ink-dim">{description}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={(e) => {
          setErr(null)
          const f = e.target.files?.[0] ?? null
          if (!f) return
          const check = checkSelectedFile(f)
          if (!check.ok) setErr(check.errorMessage ?? 'This file cannot be used.')
          else onPick(f)
        }}
      />
      {file ? (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="font-mono text-xs text-accent underline underline-offset-2"
        >
          {file.name}
        </button>
      ) : (
        <Button variant="secondary" onClick={() => inputRef.current?.click()} disabled={disabled}>
          <UploadIcon width={14} height={14} /> Choose file
        </Button>
      )}
      {err && <p className="text-xs text-danger">{err}</p>}
    </div>
  )
}

export default function SpeakerVerificationPage() {
  const { status, health } = useApiStatus()
  const [authorized, setAuthorized] = useState(false)
  const [consent, setConsent] = useState(false)
  const [reference, setReference] = useState<File | null>(null)
  const [test, setTest] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SpeakerVerifyResponse | null>(null)

  const speakerReady = status === 'online' && (health?.speaker_model_ready ?? false)
  const canRequest = authorized && consent && speakerReady && reference !== null && test !== null && !busy

  const run = async () => {
    if (!reference || !test) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await verifySpeakers(reference, test, { consent, referenceAuthorized: authorized }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The comparison failed.')
    } finally {
      setBusy(false)
    }
  }

  const accept = '.wav,.mp3,.flac,.m4a,.aac,.ogg,.oga,.opus,.webm'
  const similarityPct = result?.similarity !== null && result?.similarity !== undefined
    ? Math.round(result.similarity * 100)
    : null

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Speaker Verification"
        subtitle="Optional: compare a recording against an authorized reference clip from the same person, using speaker embeddings."
        actions={
          speakerReady ? (
            <Pill tone="ok">Embedding model ready</Pill>
          ) : (
            <Pill tone="err">Embedding model unavailable</Pill>
          )
        }
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card
            title="Authorized comparison"
            subtitle="Both recordings must be provided with explicit authorization. We never identify a person from voice alone."
            actions={<Tag>ECAPA-TDNN embeddings</Tag>}
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <UploadSlot
                title="Reference recording"
                description="An authorized clip of the known speaker, collected with their consent."
                file={reference}
                onPick={setReference}
                accept={accept}
              />
              <UploadSlot
                title="Recording to verify"
                description="The clip whose speaker you want to compare against the reference."
                file={test}
                onPick={setTest}
                accept={accept}
              />
            </div>

            <div className="mt-5 space-y-3 rounded-md border border-line bg-surface/40 p-4">
              <label className="flex cursor-pointer items-start gap-3 text-sm">
                <input
                  type="checkbox"
                  checked={authorized}
                  onChange={(e) => setAuthorized(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-accent"
                />
                <span className="text-ink-dim">
                  I am authorized to use the reference recording, and the reference speaker has
                  consented to this comparison.
                </span>
              </label>
              <label className="flex cursor-pointer items-start gap-3 text-sm">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-accent"
                />
                <span className="text-ink-dim">
                  I consent to both recordings being processed for this analysis. Audio is used
                  only for the comparison and is deleted afterwards.
                </span>
              </label>
            </div>

            <div className="mt-4 flex items-center gap-3">
              <Button onClick={() => void run()} disabled={!canRequest}>
                <UploadIcon width={14} height={14} /> {busy ? 'Comparing…' : 'Run comparison'}
              </Button>
              {!speakerReady && (
                <span className="text-xs text-ink-faint">
                  Requires the backend with a loaded speaker-embedding model. If the model is
                  unavailable, the API returns an explicit unavailable status — never a
                  fabricated similarity score.
                </span>
              )}
            </div>

            {error && (
              <div className="mt-4">
                <Callout tone="warning" title="Comparison failed">{error}</Callout>
              </div>
            )}
          </Card>

          {result && (
            <Card title="Comparison result">
              {result.status === 'unavailable' ? (
                <Callout tone="warning" title="Embedding model unavailable">
                  The backend has no speaker model loaded, so no similarity score was computed.
                  {result.warnings.length > 0 && ` Warnings: ${result.warnings.join('; ')}.`}
                </Callout>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="rounded-md border border-line bg-surface-2 p-3">
                      <p className="text-[11px] uppercase tracking-wider text-ink-dim">Similarity</p>
                      <p className="mt-1 text-lg font-semibold text-ink">
                        {similarityPct !== null ? `${similarityPct}%` : '—'}
                      </p>
                    </div>
                    <div className="rounded-md border border-line bg-surface-2 p-3">
                      <p className="text-[11px] uppercase tracking-wider text-ink-dim">Threshold</p>
                      <p className="mt-1 text-lg font-semibold text-ink">
                        {result.threshold !== null ? result.threshold.toFixed(2) : '—'}
                      </p>
                    </div>
                    <div className="rounded-md border border-line bg-surface-2 p-3">
                      <p className="text-[11px] uppercase tracking-wider text-ink-dim">Outcome</p>
                      <p className="mt-1 text-sm font-semibold text-ink">{result.outcome ?? '—'}</p>
                    </div>
                  </div>
                  <p className="text-xs text-ink-dim">
                    Embedding model: {result.embedding_model ?? 'unknown'}.
                  </p>
                  {result.warnings.length > 0 && (
                    <Callout tone="warning" title="Quality warnings">{result.warnings.join('; ')}</Callout>
                  )}
                  {result.limitations.length > 0 && (
                    <Callout tone="ethics" title="Limitations">{result.limitations.join(' ')}</Callout>
                  )}
                </div>
              )}
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Callout tone="warning" title="Similarity is not authenticity">
            A cloned voice can score high similarity to its target, and a genuine clip can score
            low because of noise, illness, emotion, language or recording conditions. Speaker
            verification answers “is this consistent with the reference speaker?”, never “is this
            audio synthetic?” — that is the separate Analyze Audio workflow.
          </Callout>
        </div>
      </div>
    </div>
  )
}
