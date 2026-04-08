# Phishing URL Detector v2

A production-shaped ML service that scores URLs as phishing or benign. Two models (gradient-boosted trees on engineered features, character-level CNN on raw URLs) sit behind a FastAPI service with SHAP explainability, PSI drift monitoring, SQLite prediction logging, Prometheus metrics, rate limiting, and a multi-stage container build.

## Why two models

| Property | XGBoost (engineered features) | CharCNN (raw URL) |
|---|---|---|
| Accuracy on standard benchmarks | ~97% | ~96% |
| Inference latency (CPU, single URL) | <1 ms | ~5 ms |
| Interpretability | High (SHAP per-feature) | Low (saliency only) |
| Adapts to new attack patterns | Needs new features | Learns automatically |
| Cold start without training data | Hard | Hard |

The XGBoost model is the default for production. The CharCNN is the comparison baseline that proves you considered the alternatives, which is the question that gets asked in every ML system design interview.

## Architecture

```
                      ┌──────────────────────────┐
   client ──────────► │      FastAPI service     │
                      │                          │
                      │  ┌─────────────────────┐ │
                      │  │  /predict           │ │ ──► XGBoost ──► SHAP
                      │  │  /predict/batch     │ │      OR
                      │  │  /feedback          │ │ ──► CharCNN
                      │  │  /drift             │ │
                      │  │  /metrics  /health  │ │
                      │  └─────────────────────┘ │
                      └────┬──────────┬──────────┘
                           │          │
                  ┌────────▼────┐  ┌──▼────────────┐
                  │ SQLite      │  │ Prometheus    │
                  │ predictions │  │ metrics       │
                  └────┬────────┘  └───────────────┘
                       │
                       ▼
                  ┌─────────────┐
                  │ PSI drift   │
                  │ monitor     │
                  └─────────────┘
```

## Project layout

```
phishing-detector/
├── app/                   FastAPI service
│   ├── main.py            Routes, middleware, lifecycle
│   ├── config.py          Pydantic settings (env-driven)
│   ├── models.py          Request/response schemas
│   ├── inference.py       Loads and runs both models
│   ├── explainer.py       SHAP TreeExplainer wrapper
│   ├── drift.py           PSI computation
│   ├── storage.py         SQLite prediction store
│   └── metrics.py         Prometheus counters/histograms/gauges
├── ml/                    Model training
│   ├── features.py        42 engineered URL features
│   ├── data_utils.py      Dataset loading + stratified split
│   ├── train_xgb.py       XGBoost trainer with threshold tuning
│   ├── train_cnn.py       CharCNN PyTorch trainer
│   └── evaluate.py        Side-by-side model comparison
├── tests/
│   ├── test_features.py   Feature extraction edge cases
│   ├── test_drift.py      PSI math
│   ├── test_storage.py    SQLite round-trips
│   └── test_api.py        FastAPI integration (auto-skips w/o model)
├── scripts/
│   ├── fetch_data.sh      Pulls OpenPhish + URLhaus + Tranco
│   └── benchmark.py       Async load test (p50/p95/p99)
├── Dockerfile             Multi-stage build, non-root runtime
├── docker-compose.yml     API + Prometheus + Grafana
├── prometheus.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .github/workflows/ci.yml
```

## Quick start

```bash
git clone <this-repo>
cd phishing-detector

# 1. Get a dataset
make install
make data        # ~3 min, fetches ~80k labeled URLs

# 2. Train both models
make train       # XGBoost, ~30s on CPU
make train-cnn   # CharCNN, ~5 min on CPU, faster on GPU

# 3. Compare them
make evaluate    # writes docs/comparison.md

# 4. Run the service
make serve       # local, with reload
# or
make up          # docker compose: api + prometheus + grafana
```

## API examples

```bash
# Single URL with explanation
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "url": "http://paypa1-secure-verify.tk/wp-login.php",
      "explain": true
    }'
```

Response:
```json
{
  "url": "http://paypa1-secure-verify.tk/wp-login.php",
  "model": "xgb",
  "phish_probability": 0.9847,
  "is_phish": true,
  "threshold": 0.5,
  "latency_ms": 0.92,
  "request_id": "f8e7c1a2-...",
  "explanation": [
    {"feature": "is_suspicious_tld", "value": 1.0, "shap_value": 1.42},
    {"feature": "brand_in_subdomain", "value": 1.0, "shap_value": 0.88},
    {"feature": "has_login_hint", "value": 1.0, "shap_value": 0.71},
    {"feature": "has_homoglyph", "value": 1.0, "shap_value": 0.43},
    {"feature": "url_entropy", "value": 4.21, "shap_value": 0.31}
  ]
}
```

```bash
# Batch
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"urls": ["https://google.com", "http://login-update.cf/auth"]}'

# Feedback (use the request_id from a prior /predict response)
curl -X POST http://localhost:8000/feedback \
    -H "Content-Type: application/json" \
    -d '{"request_id": "f8e7c1a2-...", "actual_label": 1}'

# Drift report
curl http://localhost:8000/drift

# Prometheus metrics
curl http://localhost:8000/metrics
```

Interactive Swagger docs: http://localhost:8000/docs

## What's new in v2

| Capability | v1 | v2 |
|---|---|---|
| Models | XGBoost only | XGBoost + CharCNN, side-by-side eval |
| Features | 29 | 42 with brand/homoglyph/punycode detection |
| Framework | Flask | FastAPI (async, auto-docs, type validation) |
| Explainability | None | SHAP TreeExplainer per request |
| Logging | stdout only | SQLite predictions table with feedback |
| Drift detection | None | PSI per feature, configurable threshold |
| Metrics | None | Prometheus: counters, histograms, drift gauges |
| Rate limiting | None | slowapi, configurable per minute |
| Tests | None | 25+ pytest cases across 4 test files |
| Dockerfile | Single stage | Multi-stage, non-root, cached wheels |
| CI | None | Lint + type-check + test + image build |

## Drift monitoring

PSI (Population Stability Index) is the standard metric for detecting that production traffic has shifted away from training distribution. The convention is:

| PSI | Status | Action |
|---|---|---|
| < 0.1 | Stable | Nothing |
| 0.1 to 0.2 | Warning | Investigate flagged features |
| > 0.2 | Drifted | Retrain candidate |

The `/drift` endpoint computes PSI for every feature against the reference statistics saved at training time, exposes the result as JSON, and updates Prometheus gauges so you can alert on it.

## Feedback loop

Every prediction returns a `request_id`. Downstream consumers (an analyst clicking "this was actually phishing", a customer reporting a false positive) can `POST /feedback` with that ID and the true label. Those rows accumulate in the predictions table and become a labeled set you can periodically extract to retrain the model on real production traffic.

## Skills demonstrated

Applied ML for security, feature engineering, gradient boosting, deep learning baselines (PyTorch), model explainability (SHAP), drift detection, async API design, observability with Prometheus, container hardening, multi-stage builds, ML system testing patterns, MLOps (model versioning, model cards, reference statistics, feedback loops).

## Skills mapped to job postings

- **"ML for cybersecurity"** — XGBoost + CharCNN on phishing classification
- **"Production ML pipelines"** — train/eval/serve/monitor with proper artifacts
- **"Model explainability"** — SHAP per-prediction
- **"Model monitoring"** — PSI drift + Prometheus
- **"MLOps"** — model cards, versioning, feedback ingestion
- **"Python backend"** — FastAPI, Pydantic, async, rate limiting
- **"Container deployment"** — multi-stage Docker, non-root, healthchecks
- **"Testing"** — pytest with 4 test modules covering unit + integration

## Production hardening notes

The current setup gets you 80% of the way to production. To finish:

1. Replace SQLite with PostgreSQL behind a real connection pool
2. Move feature extraction onto a Redis cache for hot URLs
3. Add OAuth or API key authentication on top of the rate limiter
4. Wire feedback into a scheduled retraining job (Airflow, Prefect, Argo)
5. Push the Docker image to a registry and deploy via Helm
6. Add a model registry like MLflow so model promotion is auditable
7. Build a small Grafana dashboard JSON to track the metrics this exposes

## Extension ideas

- Active learning: route predictions in 0.4 to 0.6 to a human review queue
- Adversarial robustness: test against URL perturbations from your thesis work
- Ensemble: average XGBoost and CharCNN probabilities, calibrate, see if it beats either alone
- ONNX export so the model runs in a browser extension
- Threat intel enrichment: pre-lookup against URLhaus before scoring
