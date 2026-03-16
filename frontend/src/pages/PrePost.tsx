import React, { useState } from 'react'
import { api } from '@/lib/api'
import type { Stage1Response } from '@/types/api'
import { confidenceColor } from '@/lib/utils'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Spinner from '@/components/ui/Spinner'
import ProbabilityMeter from '@/components/ProbabilityMeter'

interface FormState {
  rolling_weighted_median: string
  rolling_volatility: string
  posting_frequency: string
  cluster_entropy: string
  content_quality: string
  cluster_id: string
  hour_of_day: string
}

const defaultForm: FormState = {
  rolling_weighted_median: '8500',
  rolling_volatility: '1200',
  posting_frequency: '5',
  cluster_entropy: '1.8',
  content_quality: '0.72',
  cluster_id: '3',
  hour_of_day: '14',
}

export default function PrePost() {
  const [form, setForm] = useState<FormState>(defaultForm)
  const [result, setResult] = useState<Stage1Response | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function set(field: keyof FormState) {
    return (e: React.ChangeEvent<HTMLInputElement>) =>
      setForm(f => ({ ...f, [field]: e.target.value }))
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await api.predictStage1({
        rolling_weighted_median: Number(form.rolling_weighted_median),
        rolling_volatility:      Number(form.rolling_volatility),
        posting_frequency:       Number(form.posting_frequency),
        cluster_entropy:         Number(form.cluster_entropy),
        content_quality:         Number(form.content_quality),
        cluster_id:              Number(form.cluster_id),
        hour_of_day:             form.hour_of_day ? Number(form.hour_of_day) : undefined,
      })
      setResult(res)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Pre-Post Prediction</h1>
        <p className="text-gray-400 mt-1 text-sm">
          Enter account history features to get a Stage-1 survival probability before you post.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <Card>
          <form onSubmit={submit} className="space-y-4">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
              Account Features
            </h2>

            <Field label="Rolling weighted median reach" hint="Account's recent baseline reach">
              <input type="number" step="any" required value={form.rolling_weighted_median} onChange={set('rolling_weighted_median')} />
            </Field>

            <Field label="Rolling volatility" hint="Std dev of recent log-reach">
              <input type="number" step="any" min="0" required value={form.rolling_volatility} onChange={set('rolling_volatility')} />
            </Field>

            <Field label="Posting frequency" hint="Posts in the past 14 days">
              <input type="number" step="1" min="0" required value={form.posting_frequency} onChange={set('posting_frequency')} />
            </Field>

            <Field label="Cluster entropy" hint="Shannon entropy of topic distribution">
              <input type="number" step="any" min="0" required value={form.cluster_entropy} onChange={set('cluster_entropy')} />
            </Field>

            <div className="border-t border-gray-800 pt-4">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
                Post Attributes
              </h2>
            </div>

            <Field label="Content quality" hint="Score 0–1 (hook, caption, hashtags)">
              <input type="number" step="0.01" min="0" max="1" required value={form.content_quality} onChange={set('content_quality')} />
            </Field>

            <Field label="Cluster ID" hint="Topic cluster (0–19)">
              <input type="number" step="1" min="0" max="19" required value={form.cluster_id} onChange={set('cluster_id')} />
            </Field>

            <Field label="Hour of day" hint="0–23, optional">
              <input type="number" step="1" min="0" max="23" value={form.hour_of_day} onChange={set('hour_of_day')} />
            </Field>

            {error && (
              <p className="text-sm text-red-400 bg-red-900/20 rounded-lg px-3 py-2">{error}</p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-brand-500 hover:bg-brand-600 disabled:opacity-50 px-4 py-2.5 text-sm font-semibold transition-colors"
            >
              {loading ? <Spinner size="sm" /> : null}
              {loading ? 'Predicting…' : 'Predict Survival'}
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
                    <h2 className="font-semibold">Stage-1 Result</h2>
                    <Badge
                      label={result.survives ? 'Will survive' : 'Won\'t survive'}
                      variant={result.survives ? 'green' : 'red'}
                    />
                  </div>

                  <ProbabilityMeter probability={result.survival_probability} label="Survival probability" />

                  <div className="grid grid-cols-2 gap-3 pt-2">
                    <Stat label="Confidence">
                      <span className={`font-semibold ${confidenceColor(result.confidence)}`}>
                        {result.confidence}
                      </span>
                    </Stat>
                    <Stat label="Prediction ID">
                      <span className="font-mono text-gray-300">#{result.prediction_id}</span>
                    </Stat>
                    <Stat label="Model">{result.model}</Stat>
                    <Stat label="Time bucket">{result.posting_time_bucket}</Stat>
                  </div>
                </div>
              </Card>

              <Card className="bg-gray-900/50">
                <p className="text-xs text-gray-500 leading-relaxed">
                  <span className="text-gray-300 font-medium">Decision threshold:</span> 0.35 — posts with probability above
                  this are predicted to outperform the rolling baseline.{' '}
                  <span className="text-gray-300 font-medium">Confidence</span> reflects distance from threshold:
                  high = {'>'}0.25 away, medium = 0.10–0.25, low = &lt;0.10.
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Save the <span className="font-mono text-gray-300">prediction_id</span> — you'll need it in the Live Tracker
                  to run the Stage-2 update at T+1h.
                </p>
              </Card>
            </>
          ) : (
            <Card className="flex flex-col items-center justify-center h-64 text-center">
              <div className="text-4xl mb-3">📊</div>
              <p className="text-gray-400 text-sm">
                Fill in the account features and click <br />
                <span className="text-white font-medium">Predict Survival</span> to see the result.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

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
      {/* Clone child to apply styling */}
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
      <div className="text-sm capitalize">{children}</div>
    </div>
  )
}
