// API contract types — mirrors docs/API_CONTRACT.md (v1).
import type {
  PipelineCapabilities,
  RiskBadge,
  ThreatExplanation,
  UnifiedAcousticForensics,
  UnifiedReasoning,
  UnifiedScamRisk,
  UnifiedSpeakerVerification,
  UnifiedTranscription,
} from './unified'

// These are the shapes the FastAPI backend will serve (Phase 4).
// The frontend is written against these types only, never against ad-hoc shapes.

export type ApiMode = 'baseline' | 'demo' | 'production'

export type RiskCategory =
  | 'low_observed_risk'
  | 'elevated_risk'
  | 'high_risk'
  | 'inconclusive'

export interface AudioDecoders {
  ffmpeg: boolean
  ffprobe: boolean
  ffmpeg_version: string | null
  ffprobe_version: string | null
  libsndfile: string | null
  libsndfile_formats: string[]
  m4a_aac_webm_opus: boolean
  notes: string[]
}

/** Additive capability block: lets the UI tell "no decoder" from "offline". */
export interface AudioCapabilities {
  decoders: AudioDecoders
  decodable_extensions: string[]
  max_upload_bytes: number
  effective_max_upload_bytes: number
  hosting_body_limit_bytes: number | null
  ffmpeg_binary?: string
  ffprobe_binary?: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down'
  backend_ready: boolean
  model_ready: boolean
  speaker_model_ready: boolean
  version: string
  mode: ApiMode
  // Optional on the client: an older backend may not report these yet.
  audio_decoder_ready?: boolean
  max_upload_bytes?: number
  effective_max_upload_bytes?: number
  audio?: AudioCapabilities
  services?: Record<string, boolean>
  environment?: Record<string, string | boolean>
}

export interface ModelInfo {
  id: string
  name: string
  version: string
  mode: ApiMode
  kind: 'detection' | 'speaker'
  description: string
  loaded: boolean
}

export interface AnalysisInput {
  display_name: string
  duration_s: number
  sample_rate_hz: number
  channels: number
  quality_warnings: string[]
}

export interface AnalysisScores {
  raw_score: number | null
  raw_score_definition: string
  calibrated_probability: number | null
  uncertainty: number | null
  uncertainty_method: string | null
}

export interface AnalysisDecision {
  category: RiskCategory | 'unable_to_analyze'
  label: string
  reasons: string[]
  recommendations: string[]
}

export interface MeasuredFeature {
  name: string
  value: number
  unit: string
}

export interface WindowScore {
  start_s: number
  end_s: number
  score: number
}

export interface AnalysisEvidence {
  measured_features: MeasuredFeature[]
  model_explanation: string | null
  /** Same-model scores over consecutive segments (evidence, not the verdict). */
  time_window_scores?: WindowScore[]
  limitations: string[]
}

export interface AnalysisRecord {
  id: string
  created_at: string
  input: AnalysisInput
  model: { name: string; version: string; mode: ApiMode }
  scores: AnalysisScores
  decision: AnalysisDecision
  /** Present when the analysis completed (optional in early list responses). */
  evidence?: AnalysisEvidence
  latency_ms: number
}

export interface AnalysisListResponse {
  analyses: AnalysisRecord[]
  total: number
}

export interface EvaluationMetrics {
  evaluated: boolean
  accuracy?: number
  precision?: number
  recall?: number
  f1?: number
  far?: number
  frr?: number
  roc_auc?: number
  eer?: number
  latency_ms_avg?: number
  n_samples?: number
  positive_class?: string
  model_version?: string
  dataset?: string
  evaluated_at?: string
}


/* ------------------------------------------------------------------ */
/* Speaker verification (Module 3)                                    */
/* ------------------------------------------------------------------ */

export interface SpeakerVerifyResponse {
  status: 'completed' | 'unavailable'
  similarity: number | null
  threshold: number | null
  outcome: string | null
  embedding_model: string | null
  warnings: string[]
  limitations: string[]
}

/* ------------------------------------------------------------------ */
/* Audit + retention (Module 7)                                       */
/* ------------------------------------------------------------------ */

export interface AuditEvent {
  id: string
  created_at: string
  event_type: string
  action: string
  outcome: string
  actor: string
  resource_id: string | null
  request_id: string | null
  metadata: Record<string, unknown>
}

export interface AuditListResponse {
  events: AuditEvent[]
  total: number
}

export interface RetentionPolicy {
  enabled: boolean
  analyses_days: number
  audit_days: number
  temp_file_max_age_minutes: number
  interval_hours: number
  last_run: AuditEvent | null
}

export interface CleanupRunResult {
  dry_run: boolean
  analyses_deleted: number
  audit_events_deleted: number
  cutoff_analyses_iso: string | null
  cutoff_audit_iso: string | null
}

/* ------------------------------------------------------------------ */
/* Evaluation runs (Module 5)                                         */
/* ------------------------------------------------------------------ */

export interface EvaluationSummary {
  evaluated: boolean
  message?: string
  model?: { name: string; version: string } | Record<string, unknown>
  split?: Record<string, unknown>
  evaluated_at?: string
  metrics?: Record<string, unknown> | null
}

export interface EvaluationJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: { done: number; total: number }
  error: string | null
  started_at: string
  finished_at: string | null
  metrics: Record<string, unknown> | null
}

/* ---------- Family Emergency & Scam Protection (additive module) ---------- */

export interface ProtectionStage {
  name: string
  status: string
  detail?: string | null
}

export interface ProtectionTranscript {
  status: string
  text: string
  language: string | null
  language_probability: number | null
  model: string
  segments: { start: number; end: number; text: string }[]
  error: string | null
  limitations: string[]
}

export interface ProtectionVoice {
  status: string
  score?: number | null
  score_label?: string
  score_definition?: string | null
  calibrated_probability?: number | null
  threshold?: number | null
  suspected_synthetic?: boolean | null
  model?: string | null
  mode?: string
  detail?: string | null
}

export interface ProtectionSpeaker {
  status: string
  outcome: string
  similarity: number | null
  threshold?: number | null
  warnings?: string[]
  detail?: string | null
}

export interface ProtectionFindingMatch {
  phrase: string
  label: string
  start: number
  end: number
}

export interface ProtectionCategory {
  category: string
  label: string
  severity: number
  matches: ProtectionFindingMatch[]
}

export interface ProtectionContent {
  summary: string
  primary_category: string
  signal_strength: number
  discussion_context: boolean
  flags: Record<string, boolean>
  categories: ProtectionCategory[]
  limitations: string[]
}

export interface ProtectionRisk {
  category: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'INCONCLUSIVE'
  reasoning: string[]
  recommended_actions: string[]
  limitations: string[]
}

export interface GeminiIndicator {
  category: string
  evidence_quote: string
  explanation: string
  severity: string
}

export interface GeminiResult {
  classification: string
  risk_level: string
  scam_categories: string[]
  summary: string
  indicators: GeminiIndicator[]
  manipulation_tactics: string[]
  requested_actions: string[]
  protective_recommendations: string[]
  uncertainty_reasons: string[]
  transcript_quality: string
}

export interface GeminiBlock {
  status: 'completed' | 'unavailable' | 'failed'
  code?: string | null
  detail?: string | null
  latency_ms?: number | null
  model?: string | null
  result?: GeminiResult | null
  dropped_quotes?: number
}

export interface ProtectionAI {
  /** Provider-neutral block (current records). */
  llm?: GeminiBlock
  /** Legacy key kept for records written before the Grok switch. */
  gemini: GeminiBlock
  provider?: string
  provider_label?: string
  role: string
}

export interface ProtectionAnalysis {
  id: string
  created_at: string
  status: string
  display_name: string
  size_bytes: number
  duration_s: number | null
  profile_id: string | null
  profile_name: string | null
  stages: ProtectionStage[]
  transcript: ProtectionTranscript
  voice: ProtectionVoice
  speaker_comparison: ProtectionSpeaker | null
  content: ProtectionContent
  ai: ProtectionAI
  risk: ProtectionRisk
  warnings: string[]
  processing_ms: number
  pipeline: Record<string, unknown>
  /* ---- unified pipeline contract (POST /api/analyze, schema 1.0) ---- */
  schema_version?: string
  input_kind?: 'audio' | 'text'
  transcription?: UnifiedTranscription
  speaker_verification?: UnifiedSpeakerVerification
  acoustic_forensics?: UnifiedAcousticForensics
  scam_risk?: UnifiedScamRisk
  llm_reasoning?: UnifiedReasoning
  risk_badges?: RiskBadge[]
  explanation?: ThreatExplanation
  units?: Record<string, string>
}

export interface ProtectionAnalysisSummary {
  id: string
  created_at: string
  display_name: string
  duration_s: number | null
  status: string
  risk_category: string
  profile_id: string | null
}

export interface FamilyVoiceProfile {
  id: string
  created_at: string
  name: string
  relationship: string
  consent: boolean
  duration_s: number
}
