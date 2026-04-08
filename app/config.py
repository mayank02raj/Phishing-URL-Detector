"""
app/config.py
Pydantic settings, sourced from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Models
    xgb_model_path: str = "models/xgb/v1/model.json"
    xgb_meta_path: str = "models/xgb/v1"
    cnn_model_path: str = "models/cnn/v1/model.pt"
    default_model: str = "xgb"

    # Inference
    threshold: float = 0.5
    max_batch_size: int = 1000

    # Storage
    db_path: str = "data/predictions.db"

    # Drift
    drift_window_size: int = 1000
    drift_psi_threshold: float = 0.2

    # API
    api_title: str = "Phishing URL Detector"
    api_version: str = "2.0.0"
    rate_limit_per_minute: int = 600

    class Config:
        env_file = ".env"
        env_prefix = "PHISH_"


settings = Settings()
