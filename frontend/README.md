# Frontend — CreatorDynamix Dashboard

Creator-facing prediction dashboard. Built with React + TypeScript + Vite.

---

## Status

**Implemented (Phase 5 MVP).** Three views wired to the live backend API.

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

### Pre-Post (`/`)
Enter account history features before a Reel goes live.
- Calls `POST /predict/stage1`
- Shows survival probability, confidence badge, and decision context
- Returns a `prediction_id` to carry forward to the Live Tracker

### Live Tracker (`/tracker`)
Enter first-hour engagement ~60 minutes after posting.
- Calls `POST /predict/stage2`
- Shows updated probability, velocity correction, and on-track gauge (1.4× = tracking 40% above baseline)
- 93% of Stage-2's total lift is available at this checkpoint

### History (`/history`)
All past predictions with outcomes.
- Calls `GET /predictions`
- Shows Stage-1 → Stage-2 probability bars and corrections per row
- Record outcomes inline (Survived / Didn't survive buttons)
- Displays aggregate Stage-1 and Stage-2 accuracy at the top

---

## Code Structure

```
frontend/
├── src/
│   ├── App.tsx               — BrowserRouter + route definitions
│   ├── main.tsx              — React entry point
│   ├── index.css             — Tailwind base styles
│   ├── types/
│   │   └── api.ts            — TypeScript types mirroring backend schemas
│   ├── lib/
│   │   ├── api.ts            — Typed fetch wrappers for all backend endpoints
│   │   └── utils.ts          — pct(), fmt(), color helpers
│   ├── components/
│   │   ├── NavBar.tsx        — Sticky top nav with active route highlighting
│   │   ├── ProbabilityMeter.tsx — Animated probability bar + percentage
│   │   └── ui/               — Card, Badge, Spinner
│   └── pages/
│       ├── PrePost.tsx       — Stage-1 prediction form + result
│       ├── LiveTracker.tsx   — Stage-2 velocity form + on-track gauge
│       └── History.tsx       — Predictions table + outcome recording
├── vite.config.ts            — Vite config + /api proxy to backend
├── tailwind.config.js
├── tsconfig.app.json
└── package.json
```

---

## Why the Current Forms Are Wrong — UX Rebuild Plan

The current Pre-Post and Live Tracker pages expose raw ML feature inputs directly to the end user. **This is a developer test harness, not a product.** No real creator knows what "rolling weighted median reach" or "cluster entropy" means.

### What the forms currently ask vs. what they should ask

| Parameter shown today | What it actually is | How it should be sourced in v2 |
|---|---|---|
| **Rolling weighted median reach** | Exponentially-weighted median of the account's last ~20 posts' reach | Computed automatically by `feature_engine.py` from post history stored in the DB. Exposed via `GET /accounts/{id}`. |
| **Rolling volatility** | Std deviation of log-reach across recent posts | Same — auto-computed alongside the median. |
| **Posting frequency** | Number of posts in the last 14 days | Count of rows in the `Post` table for this account. |
| **Cluster entropy** | Shannon entropy of which topics the account posts about | Computed from per-post topic distribution stored in the `FeatureStore`. |
| **Cluster ID** | Which of the 20 topic clusters this post belongs to | Inferred from caption/hashtags at post creation time using the topic model. |
| **Cluster tier** | Whether this niche historically performs well (strong/medium/weak) | Precomputed lookup by cluster — set during account onboarding, invisible to user. |
| **Content quality** | 0–1 score for hook, caption, hashtag quality | Map from a simple 1–5 star rating the user gives before posting. |
| **Hour of day** | Posting hour | Read from the post timestamp — or ask "what time will you post?" with a time-picker. |
| **Prediction ID / Stage-1 prior** | Internal ML state passed between stages | Returned by the backend on Stage-1 call, stored in local state — never shown to the user. |
| **Likes / Comments at 1h** | Raw engagement 60 minutes after posting | The **only two numbers** a user should ever type. Everything else is automatic. |

### The correct end-user flow

**Onboarding (one-time, ~2 minutes):**
1. "What's your Instagram handle?" → `POST /accounts`
2. "What kind of content do you post?" (pick a niche from a human-readable list) → sets `cluster_tier` and initial `cluster_id`
3. "Paste your last few post reach numbers so we can calibrate" → `POST /accounts/{id}/posts` with `reach_24h` → backend auto-builds rolling features

**Before posting:**
1. "Describe this post" or "paste your caption" → backend infers `cluster_id`
2. "What time will you post?" → `hour_of_day` from time-picker
3. "How good is this one?" (1–5 stars) → maps to `content_quality`
4. → **Predict** button → backend calls `GET /accounts/{id}` for rolling features, then `POST /predict/stage1` → show result

**After posting (~1 hour in):**
1. "How many likes and comments do you have right now?" → two number inputs
2. → **Update prediction** → backend calls `POST /predict/stage2` → show updated result

This maps cleanly onto the backend endpoints that already exist: `POST /accounts`, `POST /accounts/{id}/posts`, `PATCH /posts/{id}/velocity`.

---

## Relationship to Backend

The frontend talks **only** to the backend API — no direct ML engine access, no model files. All prediction logic stays server-side.

For production, set the environment variable:

```
VITE_API_BASE=https://your-backend-domain.com
```

Without it, requests go to `/api/*` which Vite proxies to `localhost:8000` during development.
