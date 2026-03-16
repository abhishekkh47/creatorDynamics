export function pct(prob: number): string {
  return `${Math.round(prob * 100)}%`
}

export function fmt(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export function confidenceColor(c: 'high' | 'medium' | 'low'): string {
  return c === 'high' ? 'text-emerald-400' : c === 'medium' ? 'text-yellow-400' : 'text-red-400'
}

export function survivalColor(prob: number): string {
  if (prob >= 0.65) return 'text-emerald-400'
  if (prob >= 0.40) return 'text-yellow-400'
  return 'text-red-400'
}

export function correctionLabel(correction: number): string {
  if (correction > 0.1) return `+${pct(correction)} boost`
  if (correction < -0.1) return `${pct(correction)} drag`
  return 'no change'
}
