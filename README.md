# CreatorDynamics

A predictive intelligence system for Instagram creators — estimates whether a Reel will outperform an account's recent baseline, and updates that prediction in real time as early engagement arrives.

---

## What This Is

Most creator analytics tools tell you what happened. CreatorDynamics tells you what's going to happen — before you post, and again within the first few hours after.

The system is built in two prediction stages:

- **Stage 1 (pre-post):** Given an account's history and content attributes, predict whether this post will beat the account's rolling performance baseline.
- **Stage 2 (post-live):** Once early engagement velocity is observed (likes/comments in first 1–3 hours), update the Stage-1 prediction with real evidence.

---

## Project Structure

```
creatorDynamics/
│
├── ml_engine/        — ML core: simulation, training, evaluation
├── backend/          — API server (planned)
├── frontend/         — Creator dashboard UI (planned)
└── README.md
```

---

## System Architecture

```
ml_engine (Phase 1 ✓ · Phase 2 in progress)
      ↓
backend  (REST API — serves predictions)
      ↓
frontend (Creator dashboard — displays predictions)
```

Each layer is independently deployable. The ML engine has no dependency on the backend or frontend.

---

## Current Status

| Layer      | Status             | Description                                      |
|------------|--------------------|--------------------------------------------------|
| ml_engine  | Phase 1 complete   | Synthetic simulation + Stage-1 survival model    |
| ml_engine  | Phase 2 in progress| Stage-2 velocity correction model               |
| backend    | Not started        | FastAPI server to serve model predictions        |
| frontend   | Not started        | React dashboard for creators                     |

---

## Quickstart

Each sub-project has its own README with setup instructions:

- [`ml_engine/README.md`](ml_engine/README.md) — run the ML engine locally
- [`backend/README.md`](backend/README.md) — run the API server
- [`frontend/README.md`](frontend/README.md) — run the dashboard

---

## Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| ML engine  | Python 3.14, LightGBM, pandas, scikit-learn |
| Backend    | Python / FastAPI (planned)        |
| Frontend   | React / TypeScript (planned)      |

---

## Key Concepts

**Survival** — a post "survives" if its 24-hour reach exceeds the account's recent rolling weighted median. This is the binary label the model predicts.

**Rolling weighted median** — the account's performance baseline, computed with exponential decay weighting (`weight = exp(-λ × age)`) over past posts. More recent posts count more.

**Stage-1 → Stage-2 flow** — Stage 1 outputs a prior probability before the post goes live. Stage 2 corrects it using observed early engagement velocity.
