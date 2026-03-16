import React, { useEffect, useState } from 'react'
import {
  ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis,
} from 'recharts'
import { api } from '@/lib/api'
import type { Stage2Response } from '@/types/api'
import { loadStage1Context, type Stage1Context } from '@/lib/storage'
import { confidenceColor, pct, correctionLabel } from '@/lib/utils'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Spinner from '@/components/ui/Spinner'
import ProbabilityMeter from '@/components/ProbabilityMeter'

const CLUSTER_TIERS = ['strong', 'medium', 'weak'] as const

interface AdvancedForm {
  prediction_id:           string
  stage1_prior:            string
  rolling_weighted_median: string
  cluster_tier:            'strong' | 'medium' | 'weak'
}

function defaultAdvForm(ctx: Stage1Context | null): AdvancedForm {
  return {
    prediction_id:           String(ctx?.prediction_id           ?? ''),
    stage1_prior:            String(ctx?.stage1_prior            ?? ''),
    rolling_weighted_median: String(ctx?.rolling_weighted_median ?? ''),
    cluster_tier:            ctx?.cluster_tier ?? 'medium',
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function LiveTracker() {
  const [ctx, setCtx] = useState<Stage1Context | null>(null)

  // The only two things a normal user types
  const [likes,    setLikes]    = useState('')
  const [comments, setComments] = useState('')

  // Advanced panel
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [advForm,      setAdvForm]      = useState<AdvancedForm | null>(null)

  // Result
  const [result,  setResult]  = useState<Stage2Response | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  useEffect(() => {
    setCtx(loadStage1Context())
  }, [])

  function toggleAdvanced() {
    if (!advancedOpen) {
      setAdvForm(prev => prev ?? defaultAdvForm(ctx))
    }
    setAdvancedOpen(v => !v)
  }

  function setAdv(key: keyof AdvancedForm) {
    return (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
      setAdvForm(f => (f ? { ...f, [key]: e.target.value as AdvancedForm[typeof key] } : f))
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      let prediction_id:           number
      let stage1_prior:            number
      let rolling_weighted_median: number
      let cluster_tier:            'strong' | 'medium' | 'weak'

      if (advancedOpen && advForm) {
        prediction_id           = Number(advForm.prediction_id)
        stage1_prior            = Number(advForm.stage1_prior)
        rolling_weighted_median = Number(advForm.rolling_weighted_median)
        cluster_tier            = advForm.cluster_tier
      } else if (ctx) {
        prediction_id           = ctx.prediction_id
        stage1_prior            = ctx.stage1_prior
        rolling_weighted_median = ctx.rolling_weighted_median
        cluster_tier            = ctx.cluster_tier
      } else {
        setError(
          'No Stage-1 prediction found. Run a Pre-Post prediction first, ' +
          'or use Advanced options to enter values manually.'
        )
        setLoading(false)
        return
      }

      const res = await api.predictStage2({
        prediction_id,
        stage1_prior,
        rolling_weighted_median,
        likes_1h:    Number(likes),
        comments_1h: Number(comments),
        cluster_tier,
      })
      setResult(res)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  const onTrackScore = result?.velocity_features.on_track_score ?? null

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">
          1h Check-in <span className="text-gray-500 text-lg font-normal">T+1h</span>
        </h1>
        <p className="text-gray-400 mt-1 text-sm">
          About 60 minutes after posting, enter your like and comment count to update the prediction.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Form ─────────────────────────────────────────────────────── */}
        <Card>
          <form onSubmit={submit} className="space-y-5">

            {/* Stage-1 context banner */}
            {ctx ? (
              <div className="rounded-xl border border-gray-700/50 bg-gray-800/50 px-4 py-3 flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Pre-Post prediction loaded</p>
                  <p className="text-sm font-semibold text-gray-200">
                    #{ctx.prediction_id} &mdash; {Math.round(ctx.survival_probability * 100)}% survival
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5 capitalize">
                    {ctx.cluster_tier} niche
                  </p>
                </div>
                <Badge
                  label={ctx.survives ? 'Pre-post: ✓' : 'Pre-post: ✗'}
                  variant={ctx.survives ? 'green' : 'red'}
                />
              </div>
            ) : (
              <div className="rounded-xl border border-yellow-700/30 bg-yellow-900/10 px-4 py-3">
                <p className="text-xs text-yellow-400/80 leading-relaxed">
                  No Stage-1 prediction found. Run a{' '}
                  <a href="/" className="underline text-yellow-400 hover:text-yellow-300">
                    Pre-Post prediction
                  </a>{' '}
                  first, or use Advanced options below to enter values manually.
                </p>
              </div>
            )}

            {/* The only two inputs a normal user needs */}
            <div className="border-t border-gray-800 pt-4">
              <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">
                Your Engagement Right Now
              </h2>
            </div>

            <SimpleField label="Likes" hint="Like count ~60 minutes after posting">
              <input
                type="number"
                step="1"
                min="0"
                required
                value={likes}
                onChange={e => setLikes(e.target.value)}
                placeholder="e.g. 340"
              />
            </SimpleField>

            <SimpleField label="Comments" hint="Comment count at the same time">
              <input
                type="number"
                step="1"
                min="0"
                required
                value={comments}
                onChange={e => setComments(e.target.value)}
                placeholder="e.g. 18"
              />
            </SimpleField>

            {/* Advanced toggle */}
            <div className="pt-1">
              <button
                type="button"
                onClick={toggleAdvanced}
                className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                <span
                  className="inline-block transition-transform duration-200"
                  style={{ transform: advancedOpen ? 'rotate(90deg)' : 'rotate(0deg)' }}
                >
                  ▶
                </span>
                <span>Advanced options</span>
                {advancedOpen && (
                  <span className="text-yellow-500/70 font-medium">
                    — overrides auto-loaded context
                  </span>
                )}
              </button>
            </div>

            {/* Advanced panel */}
            {advancedOpen && advForm && (
              <div className="rounded-xl border border-gray-700/50 bg-gray-800/40 p-4 space-y-3">
                <p className="text-xs text-gray-500 leading-relaxed">
                  Pre-filled from your Stage-1 prediction. Edit to override.
                </p>

                <AdvField label="Prediction ID" hint="From the Stage-1 result">
                  <input type="number" step="1" min="1" value={advForm.prediction_id} onChange={setAdv('prediction_id')} />
                </AdvField>

                <AdvField label="Stage-1 prior" hint="Survival probability from Stage-1 (0–1)">
                  <input type="number" step="0.001" min="0" max="1" value={advForm.stage1_prior} onChange={setAdv('stage1_prior')} />
                </AdvField>

                <AdvField label="Rolling weighted median" hint="Account baseline reach">
                  <input type="number" step="any" value={advForm.rolling_weighted_median} onChange={setAdv('rolling_weighted_median')} />
                </AdvField>

                <div className="space-y-0.5">
                  <label className="block text-xs font-medium text-gray-400">Cluster tier</label>
                  <p className="text-xs text-gray-600">Your niche performance category</p>
                  <select
                    value={advForm.cluster_tier}
                    onChange={setAdv('cluster_tier')}
                    className={advSelectCls}
                  >
                    {CLUSTER_TIERS.map(t => (
                      <option key={t} value={t}>
                        {t.charAt(0).toUpperCase() + t.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {error && (
              <p className="text-sm text-red-400 bg-red-900/20 rounded-lg px-3 py-2">{error}</p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-50 px-4 py-2.5 text-sm font-semibold transition-colors"
            >
              {loading && <Spinner size="sm" />}
              {loading ? 'Updating…' : 'Update Prediction'}
            </button>
          </form>
        </Card>

        {/* ── Result ───────────────────────────────────────────────────── */}
        <div className="space-y-4">
          {result ? (
            <>
              <Card>
                <div className="space-y-5">
                  <div className="flex items-center justify-between">
                    <h2 className="font-semibold">Stage-2 Result</h2>
                    <Badge
                      label={result.survives ? 'On track' : 'Below baseline'}
                      variant={result.survives ? 'green' : 'red'}
                    />
                  </div>

                  <ProbabilityMeter probability={result.survival_probability} label="Updated survival probability" />

                  {/* Correction indicator */}
                  <div
                    className={`rounded-xl px-4 py-3 flex items-center justify-between ${
                      result.correction > 0.05
                        ? 'bg-emerald-900/20 border border-emerald-700/30'
                        : result.correction < -0.05
                        ? 'bg-red-900/20 border border-red-700/30'
                        : 'bg-gray-800 border border-gray-700'
                    }`}
                  >
                    <div>
                      <p className="text-xs text-gray-400 mb-0.5">Velocity correction</p>
                      <p className={`font-semibold ${result.correction >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {correctionLabel(result.correction)}
                      </p>
                    </div>
                    <span
                      className={`text-2xl font-bold tabular-nums ${
                        result.correction >= 0 ? 'text-emerald-400' : 'text-red-400'
                      }`}
                    >
                      {result.correction >= 0 ? '+' : ''}{pct(result.correction)}
                    </span>
                  </div>

                  {/* On-track gauge */}
                  {onTrackScore !== null && (
                    <div className="text-center py-2">
                      <p className="text-xs text-gray-500 mb-2">On-track score</p>
                      <div className="relative inline-flex items-center justify-center w-28 h-28">
                        <ResponsiveContainer width={112} height={112}>
                          <RadialBarChart
                            innerRadius="70%"
                            outerRadius="100%"
                            startAngle={180}
                            endAngle={0}
                            data={[{ value: Math.min(onTrackScore / 2, 1) * 100 }]}
                          >
                            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                            <RadialBar
                              dataKey="value"
                              cornerRadius={6}
                              fill={
                                onTrackScore >= 1
                                  ? '#34d399'
                                  : onTrackScore >= 0.7
                                  ? '#facc15'
                                  : '#f87171'
                              }
                              background={{ fill: '#1f2937' }}
                            />
                          </RadialBarChart>
                        </ResponsiveContainer>
                        <div className="absolute inset-0 flex flex-col items-center justify-end pb-3">
                          <span
                            className={`text-xl font-bold ${
                              onTrackScore >= 1
                                ? 'text-emerald-400'
                                : onTrackScore >= 0.7
                                ? 'text-yellow-400'
                                : 'text-red-400'
                            }`}
                          >
                            {onTrackScore.toFixed(2)}×
                          </span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">of baseline pace</p>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-3 pt-1">
                    <Stat label="Prior (Stage-1)">{pct(result.stage1_prior)}</Stat>
                    <Stat label="Updated (Stage-2)">{pct(result.survival_probability)}</Stat>
                    <Stat label="Confidence">
                      <span className={`font-semibold ${confidenceColor(result.confidence)}`}>
                        {result.confidence}
                      </span>
                    </Stat>
                    <Stat label="Norm likes/1h">
                      {result.velocity_features.norm_likes_1h.toFixed(4)}
                    </Stat>
                  </div>
                </div>
              </Card>

              <Card className="bg-gray-900/50">
                <p className="text-xs text-gray-500 leading-relaxed">
                  <span className="text-gray-300 font-medium">On-track score</span> — 1.0× means the
                  post is exactly on pace to hit the baseline by 24h. 1.4× means 40% ahead. Below
                  0.7× is an early warning sign.
                </p>
              </Card>
            </>
          ) : (
            <Card className="flex flex-col items-center justify-center h-64 text-center">
              <div className="text-4xl mb-3">⚡</div>
              <p className="text-gray-400 text-sm">
                Post your Reel, wait ~1 hour, then enter<br />
                your likes and comments to update the prediction.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------

const inputCls =
  'mt-1 block w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm ' +
  'text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 ' +
  'focus:border-transparent'

const advInputCls =
  'mt-0.5 block w-full rounded-lg bg-gray-900 border border-gray-700/60 px-2.5 py-1.5 text-xs ' +
  'text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-brand-500/50 ' +
  'focus:border-transparent'

const advSelectCls =
  'mt-0.5 block w-full rounded-lg bg-gray-900 border border-gray-700/60 px-2.5 py-1.5 text-xs ' +
  'text-gray-300 focus:outline-none focus:ring-1 focus:ring-brand-500/50 focus:border-transparent'

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SimpleField({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactElement
}) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-gray-300">{label}</label>
      {hint && <p className="text-xs text-gray-500">{hint}</p>}
      {React.cloneElement(
        children as React.ReactElement<React.InputHTMLAttributes<HTMLInputElement>>,
        { className: inputCls },
      )}
    </div>
  )
}

function AdvField({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactElement
}) {
  return (
    <div className="space-y-0.5">
      <label className="block text-xs font-medium text-gray-400">{label}</label>
      {hint && <p className="text-xs text-gray-600">{hint}</p>}
      {React.cloneElement(
        children as React.ReactElement<React.InputHTMLAttributes<HTMLInputElement>>,
        { className: advInputCls },
      )}
    </div>
  )
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-gray-800/60 px-3 py-2">
      <p className="text-xs text-gray-500 mb-0.5">{label}</p>
      <div className="text-sm">{children}</div>
    </div>
  )
}
