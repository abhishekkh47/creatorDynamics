// NicheOption type and niche list are served from the backend at GET /meta/niches.
// Import NicheOption from @/types/api — never hardcode cluster IDs or tiers here.

export interface FreqOption {
  label: string
  value: number // posts per 14 days
}

export const POSTING_FREQ_OPTIONS: FreqOption[] = [
  { label: 'Multiple times a day',  value: 14 },
  { label: 'Daily',                 value: 7  },
  { label: 'Every few days',        value: 4  },
  { label: 'Twice a week',          value: 3  },
  { label: 'Weekly',                value: 2  },
  { label: 'Occasionally',          value: 1  },
]

export const TIER_LABELS: Record<string, string> = {
  strong: '🔥 Strong',
  medium: '📈 Medium',
  weak:   '📉 Weak',
}
