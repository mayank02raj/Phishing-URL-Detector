"""
app/models.py
Pydantic request/response schemas.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=4, max_length=2048,
                     examples=["http://paypa1-secure-verify.tk/login.php"])
    model: Literal["xgb", "cnn"] = "xgb"
    explain: bool = False


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class PredictResponse(BaseModel):
    url: str
    model: str
    phish_probability: float
    is_phish: bool
    threshold: float
    latency_ms: float
    request_id: str
    explanation: Optional[list[FeatureContribution]] = None


class BatchPredictRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=1000)
    model: Literal["xgb", "cnn"] = "xgb"


class BatchResult(BaseModel):
    url: str
    phish_probability: float
    is_phish: bool


class BatchPredictResponse(BaseModel):
    count: int
    model: str
    threshold: float
    latency_ms: float
    results: list[BatchResult]


class DriftReport(BaseModel):
    n_recent: int
    n_reference: int
    drifted_features: list[str]
    psi_scores: dict[str, float]
    overall_status: Literal["ok", "warning", "drifted"]


class FeedbackRequest(BaseModel):
    request_id: str
    actual_label: int = Field(..., ge=0, le=1)
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    db_status: str
