import { pct } from '@/lib/utils'

interface Props {
  probability: number
  label?: string
  size?: 'sm' | 'lg'
}

export default function ProbabilityMeter({ probability, label, size = 'lg' }: Props) {
  const pct100 = Math.round(probability * 100)

  const barColor =
    pct100 >= 65 ? 'bg-emerald-500' :
    pct100 >= 40 ? 'bg-yellow-500' :
    'bg-red-500'

  const textColor =
    pct100 >= 65 ? 'text-emerald-400' :
    pct100 >= 40 ? 'text-yellow-400' :
    'text-red-400'

  return (
    <div className="space-y-2">
      {label && <p className="text-sm text-gray-400">{label}</p>}
      <div className="flex items-center gap-4">
        <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${barColor}`}
            style={{ width: `${pct100}%` }}
          />
        </div>
        <span className={`${size === 'lg' ? 'text-3xl font-bold' : 'text-xl font-semibold'} ${textColor} tabular-nums`}>
          {pct(probability)}
        </span>
      </div>
    </div>
  )
}
