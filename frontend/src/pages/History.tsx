import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import type { PredictionSummary } from '@/types/api'
import { fmt, pct } from '@/lib/utils'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Spinner from '@/components/ui/Spinner'

type Filter = 'all' | 'pending' | 'correct' | 'wrong'

export default function History() {
  const [rows, setRows] = useState<PredictionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('all')

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listPredictions({ limit: 200 })
      setRows(data)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const filtered = rows.filter(r => {
    if (filter === 'pending') return r.actual_survived === null
    if (filter === 'correct') return r.stage2_correct === true || (r.stage2_correct === null && r.stage1_correct === true)
    if (filter === 'wrong')   return r.stage2_correct === false || (r.stage2_correct === null && r.stage1_correct === false)
    return true
  })

  // Quick stats
  const withOutcome = rows.filter(r => r.actual_survived !== null)
  const s1Acc = withOutcome.length
    ? withOutcome.filter(r => r.stage1_correct).length / withOutcome.length
    : null
  const withS2 = withOutcome.filter(r => r.stage2_correct !== null)
  const s2Acc = withS2.length
    ? withS2.filter(r => r.stage2_correct).length / withS2.length
    : null

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Prediction History</h1>
          <p className="text-gray-400 mt-1 text-sm">All Stage-1 and Stage-2 predictions with recorded outcomes.</p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:border-gray-600 transition-colors"
        >
          {loading ? <Spinner size="sm" /> : '↻'} Refresh
        </button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4">
        <StatCard label="Total predictions" value={String(rows.length)} />
        <StatCard
          label="Stage-1 accuracy"
          value={s1Acc !== null ? pct(s1Acc) : '—'}
          sub={`${withOutcome.length} with outcomes`}
        />
        <StatCard
          label="Stage-2 accuracy"
          value={s2Acc !== null ? pct(s2Acc) : '—'}
          sub={`${withS2.length} with Stage-2`}
        />
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2">
        {(['all', 'pending', 'correct', 'wrong'] as Filter[]).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded-lg text-sm font-medium capitalize transition-colors ${
              filter === f
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {error && (
        <p className="text-sm text-red-400 bg-red-900/20 rounded-lg px-4 py-3">{error}</p>
      )}

      {loading && rows.length === 0 ? (
        <div className="flex justify-center py-16">
          <Spinner size="lg" />
        </div>
      ) : filtered.length === 0 ? (
        <Card className="text-center py-12 text-gray-500">
          No predictions found for this filter.
        </Card>
      ) : (
        <div className="space-y-3">
          {filtered.map(r => (
            <PredictionRow key={r.prediction_id} row={r} onOutcomeRecorded={load} />
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Prediction row
// ---------------------------------------------------------------------------

function PredictionRow({
  row,
  onOutcomeRecorded,
}: {
  row: PredictionSummary
  onOutcomeRecorded: () => void
}) {
  const [recording, setRecording] = useState(false)

  async function recordOutcome(survived: boolean) {
    setRecording(true)
    try {
      await api.recordOutcome(row.prediction_id, survived)
      onOutcomeRecorded()
    } finally {
      setRecording(false)
    }
  }

  const bestCorrect = row.stage2_correct ?? row.stage1_correct

  return (
    <Card className="!p-4">
      <div className="flex flex-wrap items-start gap-4">
        {/* ID + time */}
        <div className="w-24 shrink-0">
          <p className="font-mono text-xs text-gray-500">#{row.prediction_id}</p>
          <p className="text-xs text-gray-600 mt-0.5">{fmt(row.stage1_called_at)}</p>
        </div>

        {/* Probabilities */}
        <div className="flex gap-3 items-center flex-1 min-w-0">
          <ProbBar label="S1" prob={row.stage1_prob} />
          {row.stage2_prob !== null && (
            <>
              <span className="text-gray-600 text-xs">→</span>
              <ProbBar label="S2" prob={row.stage2_prob} />
            </>
          )}
        </div>

        {/* Correction */}
        {row.stage2_correction !== null && (
          <div className="text-right shrink-0">
            <p className="text-xs text-gray-500">correction</p>
            <p className={`text-sm font-semibold ${row.stage2_correction >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {row.stage2_correction >= 0 ? '+' : ''}{pct(row.stage2_correction)}
            </p>
          </div>
        )}

        {/* Outcome */}
        <div className="shrink-0 flex items-center gap-2">
          {row.actual_survived !== null ? (
            <>
              <Badge
                label={row.actual_survived ? 'Survived' : 'Did not survive'}
                variant={row.actual_survived ? 'green' : 'red'}
              />
              {bestCorrect !== null && (
                <Badge
                  label={bestCorrect ? '✓ correct' : '✗ wrong'}
                  variant={bestCorrect ? 'blue' : 'gray'}
                />
              )}
            </>
          ) : (
            <div className="flex gap-1">
              <button
                disabled={recording}
                onClick={() => recordOutcome(true)}
                className="rounded-lg border border-emerald-700/50 px-2 py-1 text-xs text-emerald-400 hover:bg-emerald-900/30 disabled:opacity-50 transition-colors"
              >
                Survived ✓
              </button>
              <button
                disabled={recording}
                onClick={() => recordOutcome(false)}
                className="rounded-lg border border-red-700/50 px-2 py-1 text-xs text-red-400 hover:bg-red-900/30 disabled:opacity-50 transition-colors"
              >
                Didn't ✗
              </button>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

function ProbBar({ label, prob }: { label: string; prob: number | null }) {
  if (prob === null) return null
  const pct100 = Math.round(prob * 100)
  const color = pct100 >= 65 ? 'bg-emerald-500' : pct100 >= 40 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="min-w-[80px]">
      <p className="text-xs text-gray-500 mb-1">{label} {pct100}%</p>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct100}%` }} />
      </div>
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card className="!p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
    </Card>
  )
}
