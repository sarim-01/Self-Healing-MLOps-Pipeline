# End-to-End MLOps Pipeline
### FAST NUCES — Machine Learning Operations, Spring 2026

**Student:** Sarim Rasheed  
**Course:** Machine Learning Operations (MLOps)  
**Semester:** Spring 2026

---

## Project Overview

A production-grade MLOps system that:
- Ingests live evolving data from an HTTP API
- Trains and auto-retrains an ML model based on drift and accuracy
- Deploys the model as a REST API on AWS EC2
- Monitors everything with Prometheus + Grafana
- Alerts via Slack when something goes wrong
- Automates build, test, and deploy with GitHub Actions CI/CD

---

## System Architecture

```
Live API (data source)
        ↓
Ingestion Script (polls every 45s)
        ↓ schema change / drift detected
Auto-Retraining Pipeline
        ↓ new model
AWS EC2 (Docker container)
  ├── POST /predict
  ├── GET  /health
  └── GET  /metrics
        ↓ scraped every 15s
Prometheus → Grafana Dashboard
        ↓ alert rules
Alertmanager → Slack #mlops-alerts
        ↓ every git push
GitHub Actions CI/CD (lint → build → deploy)
```

---

## Project Structure

```
mlops-project/
├── exporter/
│   └── metrics.py              # All 8 Prometheus metrics
├── ingestion/
│   ├── ingestion.py            # Data polling, schema monitoring
│   └── drift_detector.py       # Z-score drift detection
├── model/
│   ├── train.py                # RandomForest training script
│   ├── retrain_trigger.py      # Auto-retraining orchestration
│   └── model_v{N}.pkl          # Versioned model artifacts
├── serving/
│   ├── app.py                  # FastAPI inference API
│   └── Dockerfile              # Container definition
├── deploy/
│   └── deploy.sh               # Automated deployment script
├── prometheus/
│   ├── prometheus.yml          # Prometheus configuration
│   └── alert_rules.yml         # 7 alert rules
├── alertmanager/
│   └── alertmanager.yml        # Slack routing config
├── grafana/dashboards/
│   └── mlops_dashboard.json    # 6-panel Grafana dashboard
├── tests/
│   ├── test_schema.py          # Schema change detection tests
│   ├── test_drift.py           # Drift detection tests
│   └── test_predict.py         # /predict endpoint tests
├── .github/workflows/
│   └── mlops-ci.yml            # CI/CD pipeline
├── docker-compose.yml          # Observability stack
├── requirements.txt
├── .env.example
└── README.md
```

---

## AWS EC2

**Public IP:** `98.93.249.239`  
**Instance Type:** t3.micro (Free Tier)  
**Region:** us-east-1  

### Live Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `http://98.93.249.239:8000/health` | GET | Health check |
| `http://98.93.249.239:8000/predict` | POST | Run ML inference |
| `http://98.93.249.239:8000/metrics` | GET | Prometheus metrics |

### Test /health
```bash
curl http://98.93.249.239:8000/health
# Expected: {"status":"ok"}
```

### Test /predict
```bash
curl -X POST http://98.93.249.239:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.0, "feature2": 2.0}}'
# Expected: {"prediction": 0, "confidence": 0.85, "model_version": 1}
```

### Test /metrics
```bash
curl http://98.93.249.239:8000/metrics
# Expected: Prometheus text format with all 8 metrics
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- Docker Desktop
- Git

### 1. Clone the repository
```bash
git clone https://github.com/NUCES-ISB/course-project-sarimrasheed.git
cd course-project-sarimrasheed
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your values
```

### 4. Start observability stack
```bash
docker-compose up -d
```

### 5. Access services
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Alertmanager: http://localhost:9093

### 6. Run ingestion
```bash
python ingestion/ingestion.py
```

### 7. Train model
```bash
python model/train.py
```

---

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `model_accuracy` | Gauge | Current model validation accuracy |
| `records_processed_total` | Counter | Total records ingested |
| `retrain_count_total` | Counter | Total retraining runs |
| `distribution_drift_detected` | Gauge | 1=drift detected, 0=clean |
| `feature_added` | Counter | Schema features added |
| `feature_removed` | Counter | Schema features removed |
| `datalake_unavailable` | Counter | API 503 error count |
| `response_delay_seconds` | Histogram | /predict latency |

---

## Slack Alerts

**Channel:** `#mlops-alerts`  
**Workspace:** Monitoring

### Configure Slack Webhook
Add to your `.env` file:
```
SLACK_WEBHOOK_URL=your_webhook_url_here
```

### Alert Rules

| Alert | Trigger | Message |
|---|---|---|
| DataLakeUnavailable | API returns 503 | Data source returned 503 |
| FeatureAdded | New column in schema | New feature detected |
| FeatureRemoved | Column removed | Feature dropped from schema |
| DistributionDrift | drift == 1 | Distribution drift detected |
| FeatureDriftDetected | drift > 0 | Feature-level drift flagged |
| HighResponseLatency | P95 > 1.0s | P95 latency exceeded 1 second |
| LowModelAccuracy | accuracy < 0.80 | Accuracy dropped below threshold |

### Trigger alerts manually for testing
```powershell
Invoke-WebRequest -Uri "http://localhost:9093/api/v2/alerts" `
  -Method POST `
  -ContentType "application/json" `
  -Body '[{"labels":{"alertname":"LowModelAccuracy"},"annotations":{"description":"Model accuracy dropped below threshold."}}]'
```

---

## CI/CD Pipeline

**File:** `.github/workflows/mlops-ci.yml`  
**Trigger:** Every push to `main` branch

### Jobs

| Job | Steps | Description |
|---|---|---|
| Lint and Test | flake8 + pytest | Code quality + unit tests |
| Build and Push | docker build + push | Push image to Docker Hub |
| Deploy to EC2 | SSH + docker run | Pull and restart container |

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |
| `EC2_HOST` | EC2 public IP |
| `EC2_USER` | SSH user (ubuntu) |
| `EC2_SSH_KEY` | PEM key file contents |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL |

---

## Running Unit Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_drift.py::test_compute_stats PASSED
tests/test_drift.py::test_drift_detected PASSED
tests/test_drift.py::test_no_drift PASSED
tests/test_drift.py::test_empty_baseline_no_drift PASSED
tests/test_predict.py::test_health_endpoint PASSED
tests/test_predict.py::test_predict_returns_prediction_key PASSED
tests/test_predict.py::test_predict_returns_confidence_key PASSED
tests/test_predict.py::test_predict_returns_model_version PASSED
tests/test_predict.py::test_predict_confidence_between_0_and_1 PASSED
tests/test_predict.py::test_metrics_endpoint PASSED
tests/test_schema.py::test_feature_added PASSED
tests/test_schema.py::test_feature_removed PASSED
tests/test_schema.py::test_no_schema_change PASSED
tests/test_schema.py::test_multiple_changes PASSED
14 passed
```

---

## Auto-Retraining Logic

Model retrains automatically when:
1. Validation accuracy drops below 0.80
2. Distribution drift detected in incoming data
3. Schema change (feature added or removed)

When retraining fires:
- Increments `retrain_count_total` counter
- Logs the reason
- Retrains RandomForest classifier
- Redeploys updated model to AWS EC2
- Sends Slack notification with reason and new accuracy

---

## Docker Hub

**Image:** `sarimrasheed/mlops-app:latest`

```bash
docker pull sarimrasheed/mlops-app:latest
docker run -d -p 8000:8000 sarimrasheed/mlops-app:latest
```

---

## Screenshots

All screenshots are in the `screenshots/` folder:

| File | Description |
|---|---|
| `ec2_running.png` | EC2 instance running on AWS |
| `docker_ec2.png` | Docker installed on EC2 |
| `health_endpoint.png` | /health returning {"status":"ok"} |
| `metrics_output.png` | Raw Prometheus metrics output |
| `grafana_dashboard.png` | Grafana dashboard with 6 panels |
| `all_alerts_slack.png` | All 7 Slack alerts firing |
| `cicd_green.png` | GitHub Actions all 3 jobs green |