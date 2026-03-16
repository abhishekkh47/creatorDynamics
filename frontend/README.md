# Frontend — CreatorDynamix Dashboard

Creator-facing prediction dashboard. Built with React + TypeScript + Vite.

---

## Status

**Current (Phase 5 — UX rebuild complete).** All three views rebuilt for real end users. No raw ML parameters are exposed. AI-assisted content scoring and niche detection integrated via the backend provider system.

---

## Quickstart

**Prerequisite:** The backend must be running at `localhost:8000`.

```bash
cd frontend
npm install
npm run dev
```

The app will be live at `http://localhost:3000`.

All `/api/*` requests are proxied to `http://localhost:8000` via the Vite dev server — no CORS issues in development.

---

## Views

### Pre-Post Predict (`/`)

The main prediction form. Designed for non-technical creators — no ML vocabulary exposed.

**User fills in:**
1. **Caption** — paste the Reel caption
2. **Hashtags** — optional, separate with spaces or commas
3. Click **Analyze Post** → fires two API calls in parallel:
   - `POST /meta/score-content` — AI scores content quality (0–1)
   - `POST /meta/detect-niche` — AI detects the content niche and auto-selects the dropdown
4. **Niche dropdown** — auto-selected from detection, still fully editable; shows confidence badge
5. **Typical reach per Reel** — account's average views (maps to `rolling_weighted_median`)
6. **How often do you post?** — dropdown (maps to `posting_frequency`)
7. **What time will you post?** — time picker (maps to `hour_of_day`)
8. Click **Predict Survival** → calls `POST /predict/stage1`

**Result:** survival probability, confidence, and a saved `prediction_id` stored to `localStorage` for the 1h Check-in.

**Advanced toggle:** reveals all 7 raw ML fields pre-filled from the inputs above. Any field can be overridden — used by technical users or when testing with known values.

---

### 1h Check-in (`/tracker`)

Called ~60 minutes after posting.

**Auto-loaded from `localStorage`:** prediction ID, Stage-1 prior, rolling weighted median, cluster tier — set when the Pre-Post prediction was made. The user sees a context banner confirming which prediction is being updated.

**User fills in (only two fields):**
1. **Likes** — current like count at ~60 min
2. **Comments** — current comment count at ~60 min

Calls `POST /predict/stage2` → shows updated probability, velocity correction, and on-track gauge (1.4× = tracking 40% above baseline).

**Advanced toggle:** reveals all 4 Stage-2 input fields (prediction ID, prior, median, cluster tier) pre-filled from the stored context, editable as overrides.

---

### History (`/history`)

All past predictions with outcomes.

- Calls `GET /predictions`
- Shows Stage-1 → Stage-2 probability bars and corrections per row
- Record outcomes inline (Survived / Didn't survive)
- Displays aggregate Stage-1 and Stage-2 accuracy at the top

---

## Code Structure

```
frontend/
├── src/
│   ├── App.tsx               — BrowserRouter + routes (catch-all redirects to /)
│   ├── main.tsx              — React entry point
│   ├── index.css             — Tailwind base styles
│   ├── types/
│   │   └── api.ts            — TypeScript types mirroring all backend schemas
│   ├── lib/
│   │   ├── api.ts            — Typed fetch wrappers for all backend endpoints
│   │   ├── constants.ts      — POSTING_FREQ_OPTIONS, TIER_LABELS (static UI helpers only)
│   │   ├── storage.ts        — localStorage helpers for Stage-1 context persistence
│   │   └── utils.ts          — pct(), fmt(), confidenceColor(), correctionLabel()
│   ├── components/
│   │   ├── NavBar.tsx        — Sticky top nav with active route highlighting
│   │   ├── ProbabilityMeter.tsx — Animated probability bar + percentage
│   │   └── ui/               — Card, Badge, Spinner
│   └── pages/
│       ├── PrePost.tsx       — Stage-1 form: caption → AI analyze → predict
│       ├── LiveTracker.tsx   — Stage-2 form: auto-loaded context + 1h engagement
│       └── History.tsx       — Predictions table + outcome recording
├── vite.config.ts            — Vite config + /api proxy to backend
├── tailwind.config.js
├── tsconfig.app.json
└── package.json
```

**Key design rules:**
- `lib/constants.ts` holds only static UI helpers (dropdown labels, tier display strings). Niche cluster data is fetched from `GET /meta/niches` at runtime — never hardcoded here.
- `lib/storage.ts` persists Stage-1 context to `localStorage` so the 1h Check-in page can auto-load it without the user typing anything.
- `types/api.ts` is the single source of truth for all API shapes — no inline type definitions in components.
- The Advanced toggle is always available but hidden by default. It opens pre-filled with auto-computed values. When closed, the simple inputs drive the ML request. When open, the raw field values are used verbatim.

---

## AI Integration (frontend side)

The frontend has no knowledge of whether OpenAI is active or not — it calls the same two endpoints regardless:

| Endpoint | What it returns |
|---|---|
| `POST /meta/score-content` | `quality_score` (0–1), grade, per-signal breakdown, improvement tips |
| `POST /meta/detect-niche` | `cluster_id`, `confidence`, `reasoning` sentence |

If OpenAI is enabled on the backend, responses are GPT-powered. If not, the heuristic provider runs transparently. The frontend UX is identical either way.

See `backend/README.md` → AI Provider section for the toggle instructions.

---

## Stage-1 Context — How Data Flows Between Pages

After a Pre-Post prediction succeeds, this object is saved to `localStorage`:

```ts
{
  prediction_id:           number
  stage1_prior:            number   // survival_probability from Stage-1
  survival_probability:    number
  rolling_weighted_median: number
  cluster_tier:            'strong' | 'medium' | 'weak'
  survives:                boolean
  confidence:              'high' | 'medium' | 'low'
  saved_at:                string   // ISO timestamp
}
```

The 1h Check-in page loads this on mount and uses it to auto-fill the Stage-2 request. The user only needs to type likes and comments.

---

## Relationship to Backend

The frontend talks **only** to the backend API — no direct ML engine access, no model files. All prediction logic stays server-side.

For production, set the environment variable:

```
VITE_API_BASE=https://your-backend-domain.com
```

Without it, requests go to `/api/*` which Vite proxies to `localhost:8000` during development.

---

## Parameter Source Reference

This table documents what each ML input maps to in the UI so no developer ever re-exposes these as raw form fields.

| ML parameter | UI source | Auto or manual |
|---|---|---|
| `rolling_weighted_median` | "Typical reach per Reel" number input | Manual (one number) |
| `rolling_volatility` | Auto-computed as 15% of median | Auto |
| `posting_frequency` | "How often do you post?" dropdown | Auto |
| `cluster_entropy` | Default 1.5 (future: computed from post history) | Auto |
| `cluster_id` | Niche dropdown — auto-filled by `POST /meta/detect-niche` | Auto (user can override) |
| `cluster_tier` | Derived from niche selection via `GET /meta/niches` | Auto |
| `content_quality` | `POST /meta/score-content` result | Auto |
| `hour_of_day` | "What time will you post?" time picker | Manual (one field) |
| `stage1_prior` | Stored in `localStorage` from prior Stage-1 result | Auto |
| `likes_1h` | "Likes" number input on 1h Check-in | Manual |
| `comments_1h` | "Comments" number input on 1h Check-in | Manual |
