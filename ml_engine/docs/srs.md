# Instagram Reach Intelligence Engine
## Software Requirements Specification

**Version:** 1.0
**Phase:** Synthetic + Stage-1 Modeling

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for Phase 1 of the Instagram Reach Intelligence Engine — a predictive system designed to estimate whether a Reel will outperform an account's recent baseline performance.

Phase 1 focuses on synthetic simulation and survival prediction modeling.

### 1.2 Scope

**The system will:**

- Simulate Instagram account behavior
- Generate time-progressive synthetic dataset
- Compute rolling performance baseline
- Train a survival classification model
- Evaluate predictive capability using time-based validation

**The system will NOT:**

- Connect to Instagram APIs
- Use real user data
- Deploy production endpoints
- Implement velocity correction
- Simulate regime shifts

---

## 2. Overall Description

### 2.1 Product Perspective

The system is the ML core of a larger product vision that will eventually:

- Predict reel performance before posting
- Update prediction after early engagement
- Detect performance drift
- Offer content strategy insights

Phase 1 validates modeling feasibility using synthetic data.

### 2.2 Product Functions

The system shall:

- Generate synthetic accounts (200 total)
- Generate topic clusters (20 total)
- Simulate ~10,000 posts over time
- Compute reach using causal modeling
- Compute rolling weighted median baseline
- Label posts as survival (above baseline) or not
- Extract modeling features
- Train binary survival classifier
- Evaluate performance using Log Loss and ROC-AUC

### 2.3 User Characteristics

**Primary users:**

- ML engineer
- Backend engineer
- Data scientist

**Secondary future users:**

- Creator platform system
- Analytics dashboard backend

---

## 3. Functional Requirements

### FR-1: Synthetic Account Generation

The system shall:

- Generate 200 accounts
- Assign static follower count (Phase 1)
- Assign baseline reach ratio
- Assign cluster distribution
- Assign volatility parameters
- Apply slow baseline drift

### FR-2: Cluster Simulation

The system shall:

- Generate 20 topic clusters
- Assign performance multipliers
- Assign engagement multipliers
- Simulate strong, medium, and weak clusters

### FR-3: Post Simulation

The system shall:

- Simulate posts sequentially in time
- Respect account cadence patterns
- Compute reach before engagement
- Generate engagement curves

### FR-4: Baseline Computation

The system shall:

- Use exponential decay weighting
- Use global λ
- Compute weighted rolling median
- Avoid future leakage

### FR-5: Survival Labeling

The system shall define survival as:

```
reach_24h > rolling_weighted_median
```

### FR-6: Stage-1 Modeling

The system shall:

- Train binary classifier (LightGBM)
- Use time-based split
- Optimize for log loss
- Report ROC-AUC

---

## 4. Non-Functional Requirements

- Deterministic via global random seed
- Modular and extensible architecture
- No data leakage
- Clean separation of simulation and modeling
- Reproducible experiments

---

## 5. Constraints

- No external API usage
- No distributed infrastructure
- No hyperparameter tuning in Phase 1
- No follower growth simulation

---

## 6. Future Enhancements

- Velocity-based correction model
- Follower growth dynamics
- Regime shift simulation
- Walk-forward validation
- Real API ingestion
- Deployment as microservice
