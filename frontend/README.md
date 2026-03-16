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

## Relationship to Backend

The frontend talks **only** to the backend API — no direct ML engine access, no model files. All prediction logic stays server-side.

For production, set the environment variable:

```
VITE_API_BASE=https://your-backend-domain.com
```

Without it, requests go to `/api/*` which Vite proxies to `localhost:8000` during development.
