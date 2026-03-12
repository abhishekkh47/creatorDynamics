# Frontend

Creator-facing dashboard for CreatorDynamix. Displays pre-post survival predictions, real-time post tracking, and account performance trends.

---

## Status

**Not yet implemented.** This folder is a placeholder.

The frontend will be built after the backend API is live and serving real predictions.

---

## Planned Stack

| Component | Technology |
|-----------|------------|
| Framework | React + TypeScript |
| Styling | TBD |
| State management | TBD |
| Charts | TBD |

---

## Planned Views

**Pre-post prediction panel**
- Input: content category, quality estimate, posting time
- Output: predicted survival probability + confidence band
- Context: account's recent baseline + volatility

**Post-live tracker**
- Shows Stage-2 corrected probability as early engagement arrives
- Updates at 1h, 3h, 6h after posting
- Alert if post is underperforming early velocity expectations

**Account dashboard**
- Rolling baseline chart over time
- Cluster performance breakdown (which topics perform best for this account)
- Historical survival rate by time of day, day of week

---

## Setup (once implemented)

```bash
cd frontend
npm install
npm run dev
```

---

## Relationship to Backend

The frontend only talks to the backend API. It has no direct access to the ML engine or model files. All prediction logic stays server-side.
