/**
 * Types for the unified pipeline contract (POST /api/analyze, schema 1.0).
 *
 * These mirror backend/app/protection/unified.py exactly: one analysis returns
 * transcription, speaker verification, acoustic forensics, scam risk and LLM
 * reasoning, plus the badges and the structured threat explanation the
 * dashboard renders.
 */

export type RiskTone = 'critical' | 'warn' | 'ok' | 'info'

export interface RiskBadge {
  id: string
  label: string
  tone: RiskTone
  value?: string | number | null
  detail?: string | null
}

export interface UnifiedTranscription {
  status: string
  text: string
  language: string | null
  language_probability: number | null
  language_source: string | null
  engine: string | null
  model: string | null
  segments: { start: number; end: number; text: string }[]
  engines: {
    engine: string
    model?: string | null
    status: string
    latency_ms?: number | null
    characters?: number | null
    language?: string | null
    detail?: string | null
  }[]
  error: string | null
  limitations: string[]
}

export interface UnifiedSpeakerVerification {
  status: string
  outcome: string
  match: boolean | null
  similarity: number | null
  threshold: number | null
  profile_id: string | null
  profile_name: string | null
  backend: string
  model: string | null
  embedding_dim: number | null
  sample_rate_hz: number | null
  detail: string | null
  warnings: string[]
}

export interface UnifiedForensicsComponent {
  status: string
  model?: string | null
  score?: number | null
  calibrated_probability?: number | null
  threshold?: number | null
  verdict?: string | null
  dim?: number | null
  detail?: string | null
  signals?: string[]
  evidence?: Record<string, unknown>
  explanation?: string | null
  head_version?: string | null
}

export interface UnifiedAcousticForensics {
  status: string
  detector: UnifiedForensicsComponent
  handcrafted: UnifiedForensicsComponent
  wav2vec2_head: UnifiedForensicsComponent
  fusion: {
    score: number | null
    verdict: string
    threshold?: number
    weights?: Record<string, number>
    components_used: string[]
    components_missing?: string[]
    definition?: string
  }
  explanation: string | null
  detail?: string | null
}

export interface UnifiedReasoningAttempt {
  provider: string
  status: string
  code?: string | null
  model?: string | null
  latency_ms?: number | null
  detail?: string | null
}

export interface UnifiedReasoning {
  provider: string | null
  provider_label: string | null
  model: string | null
  status: string
  fallback_used: boolean
  fallback_reason: string | null
  latency_ms: number | null
  attempts: UnifiedReasoningAttempt[]
  threat_type: string | null
  severity: string | null
  confidence: number | null
  explanation: string | null
  indicators: { category: string; quote: string; explanation: string; severity: string }[]
  manipulation_tactics: string[]
  requested_actions: string[]
  recommended_actions: string[]
  uncertainty_reasons: string[]
}

export interface UnifiedScamRisk {
  score: number | null
  score_scale: string
  signal_strength: number | null
  signal_strength_max: number
  primary_category: string | null
  summary: string | null
  categories: { category: string; label: string; severity: string; matches: string[] }[]
  flags: Record<string, boolean>
  discussion_context: boolean
  evidence_quotes: string[]
  trained_classifier: Record<string, unknown> | null
  transformer_ensemble: {
    status: string
    score: number | null
    label: string | null
    threshold?: number
    branches?: { branch: string; model: string; status: string; score: number | null; detail?: string | null }[]
    detail?: string | null
    raised_signal?: boolean
  } | null
  detector: {
    model: string | null
    score: number | null
    calibrated_probability: number | null
    suspected_synthetic: boolean | null
    verdict: string | null
  }
  limitations: string[]
}

export interface ThreatExplanation {
  verdict: string | null
  threat_type: string | null
  severity: string | null
  confidence: number | null
  summary: string | null
  evidence: string[]
  transcript_excerpt: string
  manipulation_tactics: string[]
  requested_actions: string[]
  speaker: { outcome: string | null; similarity: number | null; threshold: number | null }
  authenticity: { fused_score: number | null; verdict: string | null; components_used: string[] | null }
  why_not_conclusive: string[]
  recommended_actions: string[]
  model_trail: Record<string, unknown>
}

export interface PipelineCapabilities {
  schema_version: string
  endpoint: string
  input_kinds: string[]
  detector_ready: boolean
  models: {
    transcription?: {
      available?: boolean
      engines?: { engine: string; model_id?: string; available: boolean; reason?: string | null }[]
    }
    speaker_verification?: { ready: boolean; model: string; embedding_dim: number }
    acoustic_forensics?: {
      fusion_weights?: Record<string, number>
      embedding_model?: { model: string; status: string; detail?: string | null }
      head?: { artifact: string; status: string; detail?: string | null }
      handcrafted_available?: boolean
    } | null
    scam_text?: {
      ensemble?: {
        available?: boolean
        threshold?: number
        branches?: { branch: string; model: string; status: string; detail?: string | null }[]
      } | null
    }
    llm_reasoning?: {
      configured?: boolean
      primary_model?: string
      fallback_model?: string
      chain?: string[]
      resolved_chain?: {
        provider: string
        configured: boolean
        model: string | null
        reachable?: boolean
        detail?: string | null
      }[]
    } | null
  }
}

export const TONE_CLASS: Record<RiskTone, string> = {
  critical: 'border-danger/40 bg-danger/10 text-danger',
  warn: 'border-warning/40 bg-warning/10 text-warning',
  ok: 'border-success/40 bg-success/10 text-success',
  info: 'border-line bg-surface text-ink-soft',
}

export function riskToneForCategory(category: string | null | undefined): RiskTone {
  switch ((category ?? '').toUpperCase()) {
    case 'CRITICAL':
    case 'HIGH':
      return 'critical'
    case 'MEDIUM':
      return 'warn'
    case 'LOW':
      return 'ok'
    default:
      return 'info'
  }
}
