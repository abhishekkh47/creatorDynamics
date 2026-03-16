import React, { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import type { ContentScoreResponse, NicheDetectResponse, NicheOption, Stage1Request, Stage1Response } from '@/types/api'
import { confidenceColor } from '@/lib/utils'
import { POSTING_FREQ_OPTIONS, TIER_LABELS } from '@/lib/constants'
import { saveStage1Context } from '@/lib/storage'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Spinner from '@/components/ui/Spinner'
import ProbabilityMeter from '@/components/ProbabilityMeter'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseHour(timeStr: string): number {
  const h = parseInt(timeStr.split(':')[0], 10)
  return isNaN(h) ? 14 : h
}

function buildAdvancedDefaults(
  niche: NicheOption,
  reach: number,
  freqValue: number,
  contentQuality: number,
  hour: number,
): AdvancedForm {
  return {
    rolling_weighted_median: String(reach),
    rolling_volatility:      String(Math.max(100, Math.round(reach * 0.15))),
    posting_frequency:       String(freqValue),
    cluster_entropy:         '1.5',
    content_quality:         String(contentQuality.toFixed(3)),
    cluster_id:              String(niche.cluster_id),
    hour_of_day:             String(hour),
  }
}

interface AdvancedForm {
  rolling_weighted_median: string
  rolling_volatility:      string
  posting_frequency:       string
  cluster_entropy:         string
  content_quality:         string
  cluster_id:              string
  hour_of_day:             string
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PrePost() {
  // Niches loaded from backend — never hardcoded
  const [niches,       setNiches]       = useState<NicheOption[]>([])
  const [nichesLoading, setNichesLoading] = useState(true)
  const [nichesError,  setNichesError]  = useState<string | null>(null)

  useEffect(() => {
    api.getNiches()
      .then(data => {
        setNiches(data)
        if (data.length > 0) setNiche(data[0])
      })
      .catch(err => setNichesError(String(err)))
      .finally(() => setNichesLoading(false))
  }, [])

  // Simple inputs
  const [niche, setNiche]           = useState<NicheOption | null>(null)
  const [reach, setReach]           = useState('8500')
  const [freqValue, setFreqValue]   = useState(4)
  const [timeStr, setTimeStr]       = useState('14:00')

  // Content analyzer
  const [caption,       setCaption]       = useState('')
  const [hashtags,      setHashtags]      = useState('')
  const [scoreResult,   setScoreResult]   = useState<ContentScoreResponse | null>(null)
  const [detectedNiche, setDetectedNiche] = useState<NicheDetectResponse | null>(null)
  const [scoring,       setScoring]       = useState(false)
  const [scoreError,    setScoreError]    = useState<string | null>(null)

  // Advanced panel — null means "not yet opened / synced"
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [advForm, setAdvForm]           = useState<AdvancedForm | null>(null)

  // Prediction result
  const [result,  setResult]  = useState<Stage1Response | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  async function analyzeContent() {
    if (!caption.trim()) return
    setScoring(true)
    setScoreError(null)
    try {
      // Run both in parallel — one button, two results
      const [scoreRes, nicheRes] = await Promise.all([
        api.scoreContent({ caption, hashtags }),
        api.detectNiche({ caption, hashtags }),
      ])
      setScoreResult(scoreRes)
      setDetectedNiche(nicheRes)
      // Auto-select the detected niche but don't lock it — user can still change
      const found = niches.find(n => n.cluster_id === nicheRes.cluster_id)
      if (found) setNiche(found)
    } catch (err) {
      setScoreError(String(err))
    } finally {
      setScoring(false)
    }
  }

  function toggleAdvanced() {
    if (!advancedOpen && niche) {
      const quality = scoreResult?.quality_score ?? 0.6
      setAdvForm(
        buildAdvancedDefaults(niche, Number(reach) || 8500, freqValue, quality, parseHour(timeStr))
      )
    }
    setAdvancedOpen(v => !v)
  }

  function setAdv(key: keyof AdvancedForm) {
    return (e: React.ChangeEvent<HTMLInputElement>) =>
      setAdvForm(f => (f ? { ...f, [key]: e.target.value } : f))
  }

  function buildRequest(): Stage1Request {
    if (advancedOpen && advForm) {
      return {
        rolling_weighted_median: Number(advForm.rolling_weighted_median),
        rolling_volatility:      Number(advForm.rolling_volatility),
        posting_frequency:       Number(advForm.posting_frequency),
        cluster_entropy:         Number(advForm.cluster_entropy),
        content_quality:         Number(advForm.content_quality),
        cluster_id:              Number(advForm.cluster_id),
        hour_of_day:             Number(advForm.hour_of_day),
      }
    }
    const r = Number(reach) || 8500
    return {
      rolling_weighted_median: r,
      rolling_volatility:      Math.max(100, Math.round(r * 0.15)),
      posting_frequency:       freqValue,
      cluster_entropy:         1.5,
      content_quality:         scoreResult?.quality_score ?? 0.6,
      cluster_id:              niche?.cluster_id ?? 0,
      hour_of_day:             parseHour(timeStr),
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const req = buildRequest()
      const res = await api.predictStage1(req)
      setResult(res)
      saveStage1Context({
        prediction_id:           res.prediction_id,
        stage1_prior:            res.survival_probability,
        survival_probability:    res.survival_probability,
        rolling_weighted_median: req.rolling_weighted_median,
        cluster_tier:            niche?.tier ?? 'medium',
        survives:                res.survives,
        confidence:              res.confidence,
      })
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
          Answer a few quick questions — we'll predict whether your Reel will beat your baseline.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Form ─────────────────────────────────────────────────────── */}
        <Card>
          <form onSubmit={submit} className="space-y-5">

            {/* ── Section 1: This Post ───────────────────────────────── */}
            <SectionHeader>This Post</SectionHeader>

            <div>
              <label className="block text-sm font-medium text-gray-300">Caption</label>
              <p className="text-xs text-gray-500 mt-0.5">Paste your Reel caption</p>
              <textarea
                value={caption}
                onChange={e => {
                  setCaption(e.target.value)
                  setScoreResult(null)
                  setDetectedNiche(null)
                }}
                rows={4}
                placeholder="Your caption text here…"
                className={inputCls + ' resize-none mt-1'}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300">Hashtags</label>
              <p className="text-xs text-gray-500 mt-0.5">Optional — separate with spaces or commas</p>
              <input
                type="text"
                value={hashtags}
                onChange={e => {
                  setHashtags(e.target.value)
                  setScoreResult(null)
                  setDetectedNiche(null)
                }}
                placeholder="#fitness #workout #health"
                className={inputCls + ' mt-1'}
              />
            </div>

            {/* Analyze button */}
            <button
              type="button"
              disabled={!caption.trim() || scoring}
              onClick={analyzeContent}
              className="flex items-center gap-2 rounded-lg border border-brand-600/50 bg-brand-900/20 px-3 py-1.5 text-xs font-medium text-brand-300 hover:bg-brand-900/40 disabled:opacity-40 transition-colors"
            >
              {scoring ? <Spinner size="sm" /> : '✦'}
              {scoring ? 'Analyzing content & detecting niche…' : 'Analyze Post'}
            </button>

            {scoreError && (
              <p className="text-xs text-red-400 bg-red-900/20 rounded-lg px-3 py-2">{scoreError}</p>
            )}

            {/* Content quality score card */}
            {scoreResult && <ContentScoreCard score={scoreResult} />}

            {!scoreResult && !scoring && (
              <p className="text-xs text-gray-600 italic">
                {caption.trim()
                  ? 'Click Analyze to score quality and auto-detect your niche.'
                  : "Paste your caption above and we'll handle the rest."}
              </p>
            )}

            {/* ── Section 2: Niche (auto-detected, still editable) ───── */}
            <div className="border-t border-gray-800 pt-4">
              <div className="flex items-center gap-2 mb-3">
                <SectionHeader>Content Niche</SectionHeader>
                {detectedNiche && niche?.cluster_id === detectedNiche.cluster_id && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-brand-900/40 border border-brand-700/40 px-2 py-0.5 text-xs text-brand-300">
                    ✦ Auto-detected · {Math.round(detectedNiche.confidence * 100)}% confident
                  </span>
                )}
              </div>

              {detectedNiche && (
                <p className="text-xs text-gray-500 mb-2 italic">
                  {detectedNiche.reasoning}
                  {' '}
                  <span className="text-gray-600">Not right? Pick from the list below.</span>
                </p>
              )}

              {nichesError ? (
                <p className="text-xs text-red-400 bg-red-900/20 rounded-lg px-3 py-2">
                  Could not load niches: {nichesError}
                </p>
              ) : (
                <select
                  value={niche?.cluster_id ?? ''}
                  disabled={nichesLoading}
                  onChange={e => {
                    const found = niches.find(n => n.cluster_id === Number(e.target.value))
                    if (found) {
                      setNiche(found)
                      // Clear auto-detect badge if user manually changes
                      if (detectedNiche && found.cluster_id !== detectedNiche.cluster_id) {
                        setDetectedNiche(null)
                      }
                    }
                  }}
                  className={inputCls + (nichesLoading ? ' opacity-50' : '')}
                >
                  {nichesLoading ? (
                    <option>Loading niches…</option>
                  ) : (
                    niches.map(n => (
                      <option key={n.cluster_id} value={n.cluster_id}>
                        {n.label} — {TIER_LABELS[n.tier]}
                      </option>
                    ))
                  )}
                </select>
              )}
            </div>

            {/* ── Section 3: Your Account ────────────────────────────── */}
            <div className="border-t border-gray-800 pt-4">
              <SectionHeader>Your Account</SectionHeader>
            </div>

            <SimpleField label="Typical reach per Reel" hint="Your average views on recent posts">
              <input
                type="number"
                step="1"
                min="1"
                required
                value={reach}
                onChange={e => setReach(e.target.value)}
                placeholder="e.g. 8500"
              />
            </SimpleField>

            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">How often do you post?</label>
              <select
                value={freqValue}
                onChange={e => setFreqValue(Number(e.target.value))}
                className={inputCls}
              >
                {POSTING_FREQ_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>

            {/* ── Section 4: Posting time ────────────────────────────── */}
            <div className="border-t border-gray-800 pt-4">
              <SectionHeader>Posting Time</SectionHeader>
            </div>

            <SimpleField label="What time will you post?" hint="Your local time">
              <input
                type="time"
                value={timeStr}
                onChange={e => setTimeStr(e.target.value)}
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
                    — manual values override auto-calculation
                  </span>
                )}
              </button>
            </div>

            {/* Advanced panel */}
            {advancedOpen && advForm && (
              <div className="rounded-xl border border-gray-700/50 bg-gray-800/40 p-4 space-y-3">
                <p className="text-xs text-gray-500 leading-relaxed">
                  Pre-filled from your answers above. Edit any field to override.
                </p>

                <AdvField
                  label="Rolling weighted median"
                  hint="Account's baseline reach (auto: your typical reach)"
                >
                  <input type="number" step="any" value={advForm.rolling_weighted_median} onChange={setAdv('rolling_weighted_median')} />
                </AdvField>

                <AdvField
                  label="Rolling volatility"
                  hint="Reach variability (auto: 15% of median)"
                >
                  <input type="number" step="any" min="0" value={advForm.rolling_volatility} onChange={setAdv('rolling_volatility')} />
                </AdvField>

                <AdvField
                  label="Posting frequency"
                  hint="Posts in last 14 days (auto: from dropdown)"
                >
                  <input type="number" step="1" min="0" value={advForm.posting_frequency} onChange={setAdv('posting_frequency')} />
                </AdvField>

                <AdvField
                  label="Cluster entropy"
                  hint="Content variety — 0.5 = focused niche, 2.5+ = very mixed (auto: 1.5)"
                >
                  <input type="number" step="0.1" min="0" value={advForm.cluster_entropy} onChange={setAdv('cluster_entropy')} />
                </AdvField>

                <AdvField
                  label="Content quality score"
                  hint="0–1 (auto: from content analyzer, default 0.6 if not analyzed)"
                >
                  <input type="number" step="0.01" min="0" max="1" value={advForm.content_quality} onChange={setAdv('content_quality')} />
                </AdvField>

                <AdvField
                  label="Cluster ID"
                  hint="Topic cluster 0–19 (auto: from niche)"
                >
                  <input type="number" step="1" min="0" max="19" value={advForm.cluster_id} onChange={setAdv('cluster_id')} />
                </AdvField>

                <AdvField
                  label="Hour of day"
                  hint="0–23 (auto: from post time)"
                >
                  <input type="number" step="1" min="0" max="23" value={advForm.hour_of_day} onChange={setAdv('hour_of_day')} />
                </AdvField>
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
              {loading ? 'Predicting…' : 'Predict Survival'}
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
                    <h2 className="font-semibold">Stage-1 Result</h2>
                    <Badge
                      label={result.survives ? 'Will beat baseline' : 'Below baseline'}
                      variant={result.survives ? 'green' : 'red'}
                    />
                  </div>

                  <ProbabilityMeter probability={result.survival_probability} label="Survival probability" />

                  <div className="grid grid-cols-2 gap-3 pt-1">
                    <Stat label="Confidence">
                      <span className={`font-semibold ${confidenceColor(result.confidence)}`}>
                        {result.confidence}
                      </span>
                    </Stat>
                    <Stat label="Prediction ID">
                      <span className="font-mono text-gray-300">#{result.prediction_id}</span>
                    </Stat>
                  </div>
                </div>
              </Card>

              <Card className="bg-gray-900/50">
                <p className="text-xs text-gray-500 leading-relaxed">
                  Prediction saved. Head to the{' '}
                  <a href="/tracker" className="text-brand-400 hover:text-brand-300 underline">
                    1h Check-in
                  </a>{' '}
                  page about 60 minutes after posting — your context will be pre-loaded.
                </p>
              </Card>
            </>
          ) : (
            <Card className="flex flex-col items-center justify-center h-64 text-center">
              <div className="text-4xl mb-3">📊</div>
              <p className="text-gray-400 text-sm">
                Fill in the details and click<br />
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
// Shared style
// ---------------------------------------------------------------------------

const inputCls =
  'mt-1 block w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm ' +
  'text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 ' +
  'focus:border-transparent'

const advInputCls =
  'mt-0.5 block w-full rounded-lg bg-gray-900 border border-gray-700/60 px-2.5 py-1.5 text-xs ' +
  'text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-brand-500/50 ' +
  'focus:border-transparent'

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{children}</h2>
  )
}

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

function ContentScoreCard({ score }: { score: ContentScoreResponse }) {
  const gradeColor =
    score.grade === 'Excellent' ? 'text-emerald-400' :
    score.grade === 'Good'      ? 'text-brand-400'   :
    score.grade === 'Average'   ? 'text-yellow-400'  : 'text-red-400'

  const signals: { key: keyof ContentScoreResponse['breakdown']; label: string }[] = [
    { key: 'hook_strength',      label: 'Hook strength'      },
    { key: 'cta_presence',       label: 'Call-to-action'     },
    { key: 'hashtag_quality',    label: 'Hashtag quality'    },
    { key: 'caption_length',     label: 'Caption length'     },
    { key: 'engagement_signals', label: 'Engagement signals' },
  ]

  return (
    <div className="rounded-xl border border-gray-700/50 bg-gray-800/40 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className={`text-sm font-bold ${gradeColor}`}>{score.grade}</span>
          <span className="text-xs text-gray-500 ml-2">content quality</span>
        </div>
        <span className={`text-xl font-bold tabular-nums ${gradeColor}`}>
          {Math.round(score.quality_score * 100)}
          <span className="text-sm font-normal text-gray-500">/100</span>
        </span>
      </div>

      <div className="space-y-1.5">
        {signals.map(({ key, label }) => {
          const val = score.breakdown[key]
          const pct = Math.round(val * 100)
          const barColor = pct >= 75 ? 'bg-emerald-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500'
          return (
            <div key={key} className="flex items-center gap-2">
              <span className="text-xs text-gray-500 w-36 shrink-0">{label}</span>
              <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
              </div>
              <span className="text-xs text-gray-400 w-8 text-right tabular-nums">{pct}%</span>
            </div>
          )
        })}
      </div>

      {score.tips.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-gray-700/50">
          {score.tips.map((tip, i) => (
            <p key={i} className="text-xs text-gray-400 leading-relaxed">
              <span className="text-yellow-400 mr-1">💡</span>{tip}
            </p>
          ))}
        </div>
      )}
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
