import { useRef, useState, type DragEvent, type ReactNode } from 'react'
import { UploadIcon } from '../icons'
import {
  ACCEPTED_EXTENSIONS,
  ACCEPTED_EXTENSIONS_LABEL,
  DEFAULT_AUDIO_LIMITS,
  formatBytes,
} from '../../lib/audio/validation'

interface AudioDropzoneProps {
  onFileSelected: (file: File) => void
  disabled?: boolean
  /** Extra hint rendered inside the zone (e.g. support warnings). */
  hint?: ReactNode
}

/**
 * Drag-and-drop + native file picker upload zone. Keyboard accessible: the
 * hidden input keeps native focus behavior, with a visible focus ring via
 * focus-within. Hands exactly one file to the parent for full validation.
 */
export function AudioDropzone({ onFileSelected, disabled = false, hint }: AudioDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const acceptAttr = ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(',') + ',audio/*'

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return
    // Only one file at a time; extras are ignored with a status hint.
    onFileSelected(files[0])
    if (files.length > 1 && inputRef.current) {
      // surface nothing destructive; the parent's file view is single-file
      inputRef.current.dataset.extraFiles = String(files.length - 1)
    }
    if (inputRef.current) inputRef.current.value = '' // allow re-selecting the same file
  }

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (!disabled) setDragActive(true)
  }
  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragActive(false)
  }
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragActive(false)
    if (disabled || !e.dataTransfer) return
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div
      data-testid="audio-dropzone"
      role="button"
      aria-disabled={disabled}
      tabIndex={0}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      className={`rounded-lg border border-dashed transition-colors duration-200 ${
        dragActive ? 'border-accent bg-accent/5' : 'border-line hover:border-ink-faint'
      } bg-surface/40 ${disabled ? 'opacity-50' : ''}`}
    >
      <label className="flex cursor-pointer flex-col items-center gap-3 px-6 py-12 text-center focus:outline-none">
        <input
          ref={inputRef}
          type="file"
          accept={acceptAttr}
          className="sr-only"
          disabled={disabled}
          aria-label="Choose an audio file for analysis"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div
          className={`rounded-full border p-3 transition-colors ${
            dragActive ? 'border-accent/50 bg-accent/15' : 'border-accent/25 bg-accent/10'
          } text-accent`}
        >
          <UploadIcon width={22} height={22} />
        </div>
        <p className="text-sm font-medium text-ink">
          Drag &amp; drop an audio file, or{' '}
          <span className="text-accent underline decoration-accent/40 underline-offset-4">
            browse
          </span>
        </p>
        <p className="max-w-sm text-xs leading-relaxed text-ink-dim">
          {ACCEPTED_EXTENSIONS_LABEL} · up to {formatBytes(DEFAULT_AUDIO_LIMITS.maxFileSizeBytes)}.
          The file stays on your device until you explicitly start an analysis.
        </p>
        {hint}
      </label>
    </div>
  )
}
