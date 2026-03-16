interface BadgeProps {
  label: string
  variant?: 'green' | 'yellow' | 'red' | 'blue' | 'gray'
}

const variants: Record<string, string> = {
  green:  'bg-emerald-900/50 text-emerald-300 border border-emerald-700/40',
  yellow: 'bg-yellow-900/50 text-yellow-300 border border-yellow-700/40',
  red:    'bg-red-900/50 text-red-300 border border-red-700/40',
  blue:   'bg-brand-700/30 text-brand-100 border border-brand-600/40',
  gray:   'bg-gray-800 text-gray-400 border border-gray-700',
}

export default function Badge({ label, variant = 'gray' }: BadgeProps) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${variants[variant]}`}>
      {label}
    </span>
  )
}
