// ---------------------------------------------------------------------------
// API types — mirrors backend/schemas.py
// ---------------------------------------------------------------------------

// --- Meta ---

export interface NicheOption {
  cluster_id: number
  label: string
  tier: 'strong' | 'medium' | 'weak'
}

export interface ContentScoreRequest {
  caption:  string
  hashtags?: string
}

export interface ContentScoreBreakdown {
  hook_strength:      number
  cta_presence:       number
  hashtag_quality:    number
  caption_length:     number
  engagement_signals: number
}

export interface ContentScoreResponse {
  quality_score: number
  grade: 'Excellent' | 'Good' | 'Average' | 'Needs Work'
  breakdown: ContentScoreBreakdown
  tips: string[]
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  models: Record<string, { loaded: boolean; file: string }>
  models_dir: string
}

// --- Stage-1 ---

export interface Stage1Request {
  rolling_weighted_median: number
  rolling_volatility: number
  posting_frequency: number
  cluster_entropy: number
  content_quality: number
  cluster_id: number
  hour_of_day?: number
}

export interface Stage1Response {
  prediction_id: number
  survival_probability: number
  survives: boolean
  confidence: 'high' | 'medium' | 'low'
  posting_time_bucket: number
  model: string
}

// --- Stage-2 ---

export interface Stage2Request {
  prediction_id: number
  stage1_prior: number
  rolling_weighted_median: number
  likes_1h: number
  comments_1h: number
  cluster_tier: 'strong' | 'medium' | 'weak'
}

export interface Stage2Response {
  prediction_id: number
  survival_probability: number
  survives: boolean
  stage1_prior: number
  correction: number
  confidence: 'high' | 'medium' | 'low'
  velocity_features: {
    norm_likes_1h: number
    comment_ratio_1h: number
    on_track_score: number
  }
  model: string
}

// --- Predictions list ---

export interface PredictionSummary {
  prediction_id: number
  account_id: number | null
  post_id: number | null
  stage1_prob: number | null
  stage1_survives: boolean | null
  stage2_prob: number | null
  stage2_survives: boolean | null
  stage2_correction: number | null
  actual_survived: boolean | null
  stage1_correct: boolean | null
  stage2_correct: boolean | null
  velocity_features: {
    norm_likes_1h: number
    comment_ratio_1h: number
    on_track_score: number
  } | null
  stage1_called_at: string | null
  stage2_called_at: string | null
  outcome_recorded_at: string | null
}
