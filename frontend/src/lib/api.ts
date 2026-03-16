/**
 * Typed API client for the CreatorDynamix backend.
 *
 * In development, requests go to /api/* which Vite proxies to localhost:8000.
 * In production, set VITE_API_BASE to the deployed backend URL.
 */

import type {
  ContentScoreRequest,
  ContentScoreResponse,
  HealthResponse,
  NicheDetectRequest,
  NicheDetectResponse,
  NicheOption,
  PredictionSummary,
  Stage1Request,
  Stage1Response,
  Stage2Request,
  Stage2Response,
} from '@/types/api'

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${detail}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  health: () =>
    request<HealthResponse>('/health'),

  getNiches: () =>
    request<NicheOption[]>('/meta/niches'),

  scoreContent: (body: ContentScoreRequest) =>
    request<ContentScoreResponse>('/meta/score-content', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  detectNiche: (body: NicheDetectRequest) =>
    request<NicheDetectResponse>('/meta/detect-niche', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  predictStage1: (body: Stage1Request) =>
    request<Stage1Response>('/predict/stage1', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  predictStage2: (body: Stage2Request) =>
    request<Stage2Response>('/predict/stage2', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  listPredictions: (params?: {
    limit?: number
    account_id?: number
    has_outcome?: boolean
  }) => {
    const qs = new URLSearchParams()
    if (params?.limit !== undefined) qs.set('limit', String(params.limit))
    if (params?.account_id !== undefined) qs.set('account_id', String(params.account_id))
    if (params?.has_outcome !== undefined) qs.set('has_outcome', String(params.has_outcome))
    const query = qs.toString() ? `?${qs.toString()}` : ''
    return request<PredictionSummary[]>(`/predictions${query}`)
  },

  recordOutcome: (predictionId: number, actualSurvived: boolean) =>
    request(`/predictions/${predictionId}/outcome`, {
      method: 'PATCH',
      body: JSON.stringify({ actual_survived: actualSurvived }),
    }),
}
