export interface Stage1Context {
  prediction_id: number
  stage1_prior: number
  survival_probability: number
  rolling_weighted_median: number
  cluster_tier: 'strong' | 'medium' | 'weak'
  survives: boolean
  confidence: 'high' | 'medium' | 'low'
  saved_at: string
}

const KEY = 'cd_last_stage1'

export function saveStage1Context(ctx: Omit<Stage1Context, 'saved_at'>): void {
  const full: Stage1Context = { ...ctx, saved_at: new Date().toISOString() }
  localStorage.setItem(KEY, JSON.stringify(full))
}

export function loadStage1Context(): Stage1Context | null {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Stage1Context) : null
  } catch {
    return null
  }
}

export function clearStage1Context(): void {
  localStorage.removeItem(KEY)
}
