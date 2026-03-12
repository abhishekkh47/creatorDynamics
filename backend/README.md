# Backend

REST API server for CreatorDynamics. Serves ML model predictions to the frontend and handles any future integrations (Instagram API, webhooks, etc.).

---

## Status

**Not yet implemented.** This folder is a placeholder.

The backend will be built after the ML engine reaches Phase 3 (real data ingestion). At that point, the API needs to serve live predictions rather than synthetic ones.

---

## Planned Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI (Python) |
| Model serving | LightGBM booster loaded from `ml_engine/outputs/model.txt` |
| Database | TBD |
| Auth | TBD |

---

## Planned Endpoints

```
POST /predict/pre-post
  Input:  account history features + content attributes
  Output: Stage-1 survival probability

POST /predict/post-live
  Input:  Stage-1 prior + early engagement velocity (1h, 3h, 6h)
  Output: Stage-2 corrected survival probability

GET  /account/:id/baseline
  Output: rolling weighted median + volatility for an account
```

---

## Setup (once implemented)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Relationship to ML Engine

The backend does not retrain models. It loads the pre-trained model artifact from `ml_engine/outputs/model.txt` and uses it for inference. Retraining is done by running `ml_engine/main.py` and copying the updated `model.txt`.
