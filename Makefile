.PHONY: help install data train train-cnn evaluate test serve docker up down bench clean

help:
	@echo "Phishing Detector v2 - common operations"
	@echo ""
	@echo "  make install     Install runtime + dev requirements"
	@echo "  make data        Download and assemble training dataset"
	@echo "  make train       Train XGBoost model"
	@echo "  make train-cnn   Train CharCNN baseline"
	@echo "  make evaluate    Side-by-side model comparison"
	@echo "  make test        Run pytest suite"
	@echo "  make serve       Run FastAPI locally with reload"
	@echo "  make docker      Build the production container"
	@echo "  make up          docker compose up (api + prom + grafana)"
	@echo "  make down        docker compose down"
	@echo "  make bench       Run load benchmark against localhost:8000"
	@echo "  make clean       Remove caches and data artifacts"

install:
	pip install -r requirements-dev.txt

data:
	bash scripts/fetch_data.sh

train:
	python -m ml.train_xgb --data data/urls.csv --out models/xgb/v1

train-cnn:
	python -m ml.train_cnn --data data/urls.csv --out models/cnn/v1 --epochs 8

evaluate:
	python -m ml.evaluate --data data/urls.csv \
		--xgb models/xgb/v1 --cnn models/cnn/v1 \
		--out docs/comparison.md

test:
	pytest tests/ -v --cov=app --cov=ml --cov-report=term-missing

serve:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t phish-api:latest .

up:
	docker compose up -d
	@echo ""
	@echo "API:        http://localhost:8000/docs"
	@echo "Metrics:    http://localhost:8000/metrics"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000  (admin/admin)"

down:
	docker compose down

bench:
	python scripts/benchmark.py --url http://localhost:8000 --workers 8 --duration 30

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	rm -rf .coverage htmlcov
	rm -f data/predictions.db
