"""
app/main.py
FastAPI inference service.

Endpoints:
    GET  /health             liveness + model + DB status
    GET  /metrics            Prometheus exposition
    GET  /stats              prediction store summary
    POST /predict            single URL, optional SHAP explanation
    POST /predict/batch      up to 1000 URLs
    POST /feedback           label correction for a past prediction
    GET  /drift              PSI report on recent traffic vs training stats
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import metrics
from app.config import settings
from app.drift import DriftMonitor
from app.explainer import XGBExplainer
from app.inference import InferenceEngine
from app.models import (BatchPredictRequest, BatchPredictResponse,
                         BatchResult, DriftReport, FeedbackRequest,
                         HealthResponse, PredictRequest, PredictResponse)
from app.storage import PredictionStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("phish-api")

# ----------------------------------------------------------------- init

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title=settings.api_title, version=settings.api_version)
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

engine = InferenceEngine()
store = PredictionStore(settings.db_path)
drift_monitor = DriftMonitor(
    f"{settings.xgb_meta_path}/reference_stats.json",
    threshold=settings.drift_psi_threshold,
)

explainer: XGBExplainer | None = None
if engine.xgb_model is not None:
    explainer = XGBExplainer(engine.xgb_model)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response("Rate limit exceeded", status_code=429)


# ----------------------------------------------------------------- routes

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=settings.api_version,
        models_loaded=engine.loaded,
        db_status=store.health(),
    )


@app.get("/metrics")
async def prom_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
async def stats():
    return store.stats()


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def predict(request: Request, body: PredictRequest):
    if body.model not in engine.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{body.model}' not loaded. "
                   f"Available: {engine.loaded}")

    t0 = time.perf_counter()
    proba_arr, X = engine.predict([body.url], body.model)
    proba = float(proba_arr[0])
    is_phish = proba >= settings.threshold
    latency_ms = (time.perf_counter() - t0) * 1000

    explanation = None
    if body.explain and body.model == "xgb" and explainer is not None:
        explanation = explainer.explain(X, top_k=8)[0]

    features_dict = X.iloc[0].to_dict() if X is not None else None
    request_id = store.log_prediction(
        body.url, body.model, proba, is_phish,
        settings.threshold, latency_ms, features_dict)

    # Metrics
    metrics.predictions_total.labels(
        model=body.model,
        outcome="phish" if is_phish else "benign").inc()
    metrics.prediction_latency.labels(model=body.model).observe(latency_ms / 1000)
    metrics.phish_probability.labels(model=body.model).observe(proba)

    return PredictResponse(
        url=body.url,
        model=body.model,
        phish_probability=round(proba, 4),
        is_phish=bool(is_phish),
        threshold=settings.threshold,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
        explanation=explanation,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def predict_batch(request: Request, body: BatchPredictRequest):
    if len(body.urls) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large (max {settings.max_batch_size})")
    if body.model not in engine.loaded:
        raise HTTPException(status_code=503,
                            detail=f"Model '{body.model}' not loaded")

    t0 = time.perf_counter()
    proba_arr, X = engine.predict(body.urls, body.model)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = []
    for i, url in enumerate(body.urls):
        proba = float(proba_arr[i])
        is_phish = proba >= settings.threshold
        feats = X.iloc[i].to_dict() if X is not None else None
        store.log_prediction(
            url, body.model, proba, is_phish,
            settings.threshold, latency_ms / len(body.urls), feats)
        results.append(BatchResult(
            url=url,
            phish_probability=round(proba, 4),
            is_phish=bool(is_phish)))
        metrics.predictions_total.labels(
            model=body.model,
            outcome="phish" if is_phish else "benign").inc()
        metrics.phish_probability.labels(model=body.model).observe(proba)

    metrics.batch_size.observe(len(body.urls))
    metrics.prediction_latency.labels(model=body.model).observe(
        latency_ms / 1000)

    return BatchPredictResponse(
        count=len(results),
        model=body.model,
        threshold=settings.threshold,
        latency_ms=round(latency_ms, 2),
        results=results,
    )


@app.post("/feedback")
@limiter.limit("60/minute")
async def feedback(request: Request, body: FeedbackRequest):
    ok = store.record_feedback(body.request_id, body.actual_label)
    if not ok:
        raise HTTPException(404, "request_id not found")
    return {"status": "recorded", "request_id": body.request_id}


@app.get("/drift", response_model=DriftReport)
async def drift():
    recent = store.recent_features(settings.drift_window_size)
    if recent.empty:
        return DriftReport(
            n_recent=0,
            n_reference=0,
            drifted_features=[],
            psi_scores={},
            overall_status="ok",
        )
    scores, drifted, status = drift_monitor.compute_drift(recent)
    for feat, score in scores.items():
        metrics.drift_score.labels(feature=feat).set(score)
    metrics.drifted_features_count.set(len(drifted))
    return DriftReport(
        n_recent=len(recent),
        n_reference=settings.drift_window_size,
        drifted_features=drifted,
        psi_scores=scores,
        overall_status=status,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
