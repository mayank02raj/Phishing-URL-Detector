"""
app/metrics.py
Prometheus metrics for the phishing API.
"""

from prometheus_client import Counter, Histogram, Gauge

predictions_total = Counter(
    "phish_predictions_total",
    "Total number of predictions served",
    ["model", "outcome"],
)

prediction_latency = Histogram(
    "phish_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

batch_size = Histogram(
    "phish_batch_size",
    "Batch sizes seen at the batch endpoint",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

phish_probability = Histogram(
    "phish_probability_distribution",
    "Distribution of predicted phishing probabilities",
    ["model"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

drift_score = Gauge(
    "phish_drift_psi",
    "Most recent PSI score per feature",
    ["feature"],
)

drifted_features_count = Gauge(
    "phish_drifted_features_count",
    "Number of features currently exceeding the PSI threshold",
)
