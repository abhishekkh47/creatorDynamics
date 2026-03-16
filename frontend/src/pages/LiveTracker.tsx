import React, { useState } from 'react'
import {
  ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis,
} from 'recharts'
import { api } from '@/lib/api'
import type { Stage2Response } from '@/types/api'
import { confidenceColor, pct, correctionLabel } from '@/lib/utils'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Spinner from '@/components/ui/Spinner'
import ProbabilityMeter from '@/components/ProbabilityMeter'

const CLUSTER_TIERS = ['strong', 'medium', 'weak'] as const


interface FormState {
  prediction_id: string
  stage1_prior: string
  rolling_weighted_median: string
  likes_1h: string
  comments_1h: string
  cluster_tier: 'strong' | 'medium' | 'weak'
}

const defaultForm: FormState = {
  prediction_id: '',
  stage1_prior: '',
  rolling_weighted_median: '8500',
  likes_1h: '',
  comments_1h: '',
  cluster_tier: 'medium',
}

export default function LiveTracker() {
  const [form, setForm] = useState<FormState>(defaultForm)
  const [result, setResult] = useState<Stage2Response | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function set(field: keyof FormState) {
    return (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
      setForm(f => ({ ...f, [field]: e.target.value as FormState[typeof field] }))
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await api.predictStage2({
        prediction_id:           Number(form.prediction_id),
        stage1_prior:            Number(form.stage1_prior),
        rolling_weighted_median: Number(form.rolling_weighted_median),
        likes_1h:                Number(form.likes_1h),
        comments_1h:             Number(form.comments_1h),
        cluster_tier:            form.cluster_tier,
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
        <h1 className="text-2xl font-bold">Live Tracker <span className="text-gray-500 text-lg font-normal">T+1h</span></h1>
        <p className="text-gray-400 mt-1 text-sm">
          Enter first-hour engagement to update the Stage-1 prediction with real velocity data.
          93% of Stage-2's lift is available within the first hour.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <Card>
          <form onSubmit={submit} className="space-y-4">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
              Stage-1 Reference
            </h2>

            <Field label="Prediction ID" hint="From the Pre-Post prediction you saved">
              <input type="number" step="1" min="1" required value={form.prediction_id} onChange={set('prediction_id')} placeholder="e.g. 42" />
            </Field>

            <Field label="Stage-1 prior" hint="survival_probability from Stage-1">
              <input type="number" step="0.001" min="0" max="1" required value={form.stage1_prior} onChange={set('stage1_prior')} placeholder="e.g. 0.62" />
            </Field>

            <Field label="Rolling weighted median" hint="Same value used in Stage-1">
              <input type="number" step="any" required value={form.rolling_weighted_median} onChange={set('rolling_weighted_median')} />
            </Field>

            <div className="border-t border-gray-800 pt-4">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
                1h Engagement
              </h2>
            </div>

            <Field label="Likes at 1h" hint="Raw like count ~60 min after posting">
              <input type="number" step="1" min="0" required value={form.likes_1h} onChange={set('likes_1h')} placeholder="e.g. 340" />
            </Field>

            <Field label="Comments at 1h" hint="Raw comment count at 1h">
              <input type="number" step="1" min="0" required value={form.comments_1h} onChange={set('comments_1h')} placeholder="e.g. 18" />
            </Field>

            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">Cluster tier</label>
              <p className="text-xs text-gray-500">Your niche's performance tier</p>
              <select
                value={form.cluster_tier}
                onChange={set('cluster_tier')}
                className="mt-1 block w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
              >
                {CLUSTER_TIERS.map(t => (
                  <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
                ))}
              </select>
            </div>

            {error && (
              <p className="text-sm text-red-400 bg-red-900/20 rounded-lg px-3 py-2">{error}</p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-50 px-4 py-2.5 text-sm font-semibold transition-colors"
            >
              {loading ? <Spinner size="sm" /> : null}
              {loading ? 'Updating…' : 'Update Prediction'}
            </button>
          </form>
        </Card>

        {/* Result */}
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
                  <div className={`rounded-xl px-4 py-3 flex items-center justify-between
                    ${result.correction > 0.05 ? 'bg-emerald-900/20 border border-emerald-700/30'
                      : result.correction < -0.05 ? 'bg-red-900/20 border border-red-700/30'
                      : 'bg-gray-800 border border-gray-700'}`}>
                    <div>
                      <p className="text-xs text-gray-400 mb-0.5">Velocity correction</p>
                      <p className={`font-semibold ${result.correction > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {correctionLabel(result.correction)}
                      </p>
                    </div>
                    <span className={`text-2xl font-bold tabular-nums ${result.correction >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
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
                              fill={onTrackScore >= 1 ? '#34d399' : onTrackScore >= 0.7 ? '#facc15' : '#f87171'}
                              background={{ fill: '#1f2937' }}
                            />
                          </RadialBarChart>
                        </ResponsiveContainer>
                        <div className="absolute inset-0 flex flex-col items-center justify-end pb-3">
                          <span className={`text-xl font-bold ${onTrackScore >= 1 ? 'text-emerald-400' : onTrackScore >= 0.7 ? 'text-yellow-400' : 'text-red-400'}`}>
                            {onTrackScore.toFixed(2)}×
                          </span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">of baseline pace</p>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-3 pt-1">
                    <Stat label="Prior (Stage-1)">{pct(result.stage1_prior)}</Stat>
                    <Stat label="Posterior (Stage-2)">{pct(result.survival_probability)}</Stat>
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
                  <span className="text-gray-300 font-medium">On-track score</span> — 1.0× means the post is
                  exactly on pace to hit the baseline by 24h. 1.4× means 40% ahead. Below 0.7× is a
                  strong early warning sign.
                </p>
              </Card>
            </>
          ) : (
            <Card className="flex flex-col items-center justify-center h-64 text-center">
              <div className="text-4xl mb-3">⚡</div>
              <p className="text-gray-400 text-sm">
                Post your Reel, wait 1 hour, then enter the<br />
                engagement data to update the prediction.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

function Field({
  label, hint, children,
}: {
  label: string
  hint?: string
  children: React.ReactElement
}) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-gray-300">{label}</label>
      {hint && <p className="text-xs text-gray-500">{hint}</p>}
      {React.cloneElement(children as React.ReactElement<React.InputHTMLAttributes<HTMLInputElement>>, {
        className:
          'mt-1 block w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm ' +
          'text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 ' +
          'focus:border-transparent',
      })}
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
