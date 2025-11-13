# MLOps-Phase3

## Student Performance Prediction with Complete MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Published-blue)](https://hub.docker.com/r/a01566204/ml-service)
[![Tests](https://img.shields.io/badge/Tests-115%2F128%20Passing-brightgreen)](./tests/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5)](./k8s/)

---

## Table of Contents

- [Overview](#overview)
- [Requirements Completed](#requirements-completed)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Data Drift Detection](#data-drift-detection)
- [Results](#results)
- [Technologies](#technologies-used)
- [Project Structure](#project-structure)

---

## Overview

Production-ready MLOps system for predicting student academic performance with complete testing, monitoring, and deployment infrastructure.

### Key-Features
- **Comprehensive Testing:** 115/128 tests passing (90% success rate)
- **Restful API:** FastAPI with 6 production endpoints
- **Docker:** Published to DockerHub for easy deployment
- **Kubernetes:** Production orchestration with auto-scaling
- **Drift Detection:** Real-time monitoring with Evidently
- **Ful Reproducibility:** Consistent results across environments

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 41.8% |
| Precision | 95.7% |
| F1-Score | 27.4% |
| Recall | 16.0% |

---

## Requirementes Completed

### 1, Comprehensive Testing Framework

**Test Coverage:** 115/128 test passing (90% pass rate, 0 failures)
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

**Test Categories:**
- **Unit Tests:** Metrics, preprocessing, inference, drift detection
- **Integration Tests:** API endpoints, DVC pipeline, end-to-end
- **Edge Cases:** Empty data, invalid inputs, error handling

**Test Results:**

```
115 passed
13 skipped (expected - optional dependencies)
0 failed

```
---

### 2. Model Serving with FastAPI

**6 Production Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check with model status |
| `/predict` | POST | Model predictions with probabilities |
| `/model-info` | GET | Model metadata and features |
| `/detect-drift` | POST | Data drift detection |
| `/monitoring/stats` | GET | Monitoring metrics |

**Features:**
- Input validation with Pydantic
- Error handling (422, 503 status codes)
- OpenAPI/Swagger documentation
- Health checks with model verification

**Start the API:**

```bash
uvicorn app.main:app --reload --port 8000
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---
### 3. Reproducibility Across Environments

**Fixed Dependencies:**
- `requirements-prod.txt` - Production dependencies for Docker
- `reqirements-dev.txt` - Development and testing tools
- All versions pinned (e.g., `scikit-learn==1.7.2`)

**Random Seeds Fixed:**
- `random_state=42` in all models and splits
- Configured consistently in `params.yaml`

**DVC Versioning:**
- 16 trained models tracked
- Raw data versioned
- Complete pipeline reproducibility

**Verification:**
```bash
# Same sklearn version in Docker and training
docker run a01566204/ml-service:1.0.0 python -c "import sklearn; print(sklearn.__version__)"
# Output: 1.7.2
```

---

### 4. Docker Containerization

**Published Image:** [`a01566204/ml-service:1.0.0`](https://hub.docker.com/r/a01566204/ml-service)

**Features:**
- Optimized Python 3.10-slim base image
- Multi-layer caching for fast builds
- Health checks built-in
- Resource limits configured
- Production-ready configuration

**Quick Start:**
```bas
# Pull from DockerHub
docker pull a01566204/ml-service:1.0.0

# Run container
docker run -d -p 8000:8000 --name ml-api a01566204/ml-service:1.0.0

# Test
curl http://localhost:8000/health

# View logs
docker logs -f ml-api
```

**Build from Source:**
```bash
docker build -t ml-service:latest .
docker run -d -p 8000:8000 ml-service:latest
```

---

### 5. Data Drift Detection & Monitoring

**Drift Simulation Results**
| Drift Type | Drifted Columns | F1 Change | Performance Impact | Recommended Action |
|------------|-----------------|-----------|-------------------|-------------------|
| **Mean Shift** | 4/9 (44%) | -22% | 🔴 Critical | Immediate retraining |
| **Variance Change** | 4/9 (44%) | -33% | 🟢 Stable | Continue monitoring |
| **Distribution Shift** | 4/9 (44%) | -2% | 🟢 Stable | Continue monitoring |

**Run Drift Simulation:**
```bash
python scripts/simulate_drift.py
```

**Generated outputs:**
- `reports/drift/drift_analysis_*.png` - Distrubution visualizations
- `reports/drift/drift_report.json` - Comprehensive metrics
- `data/monitoring/drifted_data_*.csv` - Simulated datasets

**Alert Thresholds:**
- 🔴 **CRITICAL:** Performance degradation >10% → Immediate retraining required
- 🟡 **WARNING:** Performance degradation >5% → Schedule retraining
- 🟢 **MONITOR:** Drift detected but stable → Continue monitoring
- ✅ **OK:** No significant drift → Normal operations

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User / Client                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          Kubernetes Ingress / Load Balancer             │
│                  (External Access)                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│               Kubernetes Service                        │
│          (LoadBalancer / ClusterIP)                     │
└──────┬──────────────┬──────────────┬───────────────────┘
       │              │              │
       ▼              ▼              ▼
  ┌────────┐    ┌────────┐    ┌────────┐
  │ Pod 1  │    │ Pod 2  │    │ Pod 3  │
  │FastAPI │    │FastAPI │    │FastAPI │
  │ + ML   │    │ + ML   │    │ + ML   │
  │ Model  │    │ Model  │    │ Model  │
  └────────┘    └────────┘    └────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │       HPA        │
            │  (2-10 replicas) │
            │ CPU/Memory based │
            └──────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)
- kubectl (optional, for Kubernetes)

### Installation
```bash
# 1. Clone Repository
git clone https://github.com/Lia1566/MLOps-Phase3.git
cd MLOps-Phase3

# 2. Install dependencies
pip install -r requirements-prod.txt
pip install -r requirements-dev.txt

# 3. Verify installation
python -c "from app.inference import get_model; print('Installation successful!')"
```

### Run API Locally
```bash
# Start the API
uvicorn app.main:app --reload --port 8000

# Test in another terminal
curl http://localhost:8000/health
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=app --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Testing

### Test Results Summary

```
115 passed
13 skipped (expected)
0 failed
Test coverage: 85%+
Execution time: ~4 seconds
```

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_metrics.py      # Metrics calculation
│   ├── test_model_inference.py  # Model predictions
│   ├── test_preprocessing.py    # Data preprocessing
│   ├── test_feature_engineering.py  # Feature engineering
│   ├── test_drift.py        # Drift detection
│   └── test_config.py       # Configuration
├── integration/             # Integration tests
│   ├── test_api.py          # API endpoints
│   ├── test_dvc_stages.py   # DVC pipeline
│   └── test_pipeline_e2e.py # End-to-end pipeline
└── conftest.py             # Shared fixtures

```

### Running Specific Tests

```bash

# Test API Endpoints
pytest tests/integration/test_api.py::test_health_check -v

# Test model inference
pytest tests/unit/test_model_inference.py -v

# Test drift detection
pytest tests/unit/test_drift.py -v

# Test with markers
pytest -m unit  # Only unit tests
pytest -m integration  # Only integration tests
```

---

## API Documentation

### POST /predict - Make Predictions

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Class_X_Percentage": 85.5,
    "Class_XII_Percentage": 78.0,
    "Study_Hours": 5.0,
    "Gender": "Male",
    "Caste": "General",
    "Coaching": "Yes",
    "Medium": "English"
  }'
```

**Response:**
```.jason
{
  "prediction": 1,
  "prediction_label": "High Performance",
  "probability": 0.7834,
  "probabilities": {
    "Low Performance": 0.2166,
    "High Performance": 0.7834
  },
  "timestamp": "2025-11-12T10:30:00"
}
```

### GET /health - Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2025-11-12T10:30:00"
}
```

### GET /model-info - Model Metadata
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_name": "pipeline_baseline.pkl",
  "model_type": "Pipeline",
  "version": "1.0.0",
  "features": ["Class_X_Percentage", "Class_XII_Percentage", ...],
  "target_classes": {
    "0": "Low Performance",
    "1": "High Performance"
  }
}
```

### Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAI JSON:** http://localhost:8000/openapi.json

---

## Docker Deployment

### Quick Deploy
```bash
# Pull and run from DockerHub
docker pull a01566204/ml-service:1.0.0
docker run -d -p 8000:8000 --name ml-api a01566204/ml-service:1.0.0

# Test the deployment
curl http://localhost:8000/health

# View logs
docker logs -f ml-api

# Stop and remove
docker stop ml-api && docker rm ml-api
```

### Build from Source
```bash
# Build image
docker build -t ml-service:latest .

# Run container
docker run -d -p 8000:8000 --name ml-api ml-service:latest

# Or use docker-compose
docker-compose up -d

# View all services
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Docker Image Details

- **Base Image:** `python:3.10-slim`
- **Size:** ~1GB (optimized)
- **Layers:** Multi-stage build with caching
- **Health Check:** Built-in at `/health` endpoint
- **Port:** 8000

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or minikube)
- `kubectl` configured and connected

### Deploy to Kubernetes
```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get all -l app=ml-service

# Check pods
kubectl get pods -l app=ml-service

# Check service
kubectl get svc ml-service

# Check auto-scaler
kubectl get hpa ml-service-hpa
```

### Access the API

```bash
# Port forward for local access
kubectl port-forward svc/ml-service 8000:80

# Test the API
curl http://localhost:8000/health

# Or get LoadBalancer IP (if using cloud provider)
kubectl get svc ml-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### Kubernetes Features


| Feature | Configuration | Benefit |
|---------|--------------|---------|
| **Replicas** | 3 pods | High availability |
| **Auto-Scaling** | 2-10 pods | Cost optimization |
| **Health Probes** | Liveness & Readiness | Automatic recovery |
| **Resource Limits** | CPU: 500m, Memory: 1Gi | Predictable performance |
| **Rolling Updates** | Zero-downtime | Seamless deployments |
| **Load Balancing** | Service LoadBalancer | Traffic distribution |

### Scaling Examples
```bash
# Manual scaling
kubectl scale deployment ml-service --replicas=5

# Watch auto-scaling in action
kubectl get hpa ml-service-hpa --watch

# View current resource usage
kubectl top pods -l app=ml-service
```

### Monitoring
```bash
# View deployment status
kubectl describe deployment ml-service

# View pod logs
kubectl logs -f deployment/ml-service

# View recent events
kubectl get events --sort-by='.metadata.creationTimestamp'
```

### Complete Guide

See [k8s/README.md](./k8s/README.md) for detailed Kubernetes deployment guide.

---

## Data Drift Detection

### Overview

Real-time monitoring of data distribution changes using **Evidently** to detect when model retraining is needed.

### Drift Scenarios

We tested 3 drift scenarios:

1. **Mean Shift** - Performance indicators decreased by 10 points
2. **Variance Change** - Increased variability (σ=15) in percentages
3. **Distribution Shift** - Changed categorical distributions (less coaching, less English medium)

### Running Drift Simulation
```bash
# Run complete drift simulation
python scripts/simulate_drift.py

# Output:
# 3 drift visualizations (PNG)
# Comprehensive drift report (JSON)
# 3 drifted datasets (CSV)
```

### Drift Detection Results

| Scenario | Columns Drifted | Accuracy Change | F1 Change | Action Required |
|----------|----------------|-----------------|-----------|-----------------|
| Mean Shift | 4/9 (44%) | +29.6% | -22.1% | 🔴 CRITICAL: Retrain immediately |
| Variance Change | 4/9 (44%) | +22.8% | +33.1% | 🟢 MONITOR: Performance improved |
| Distribution Shift | 4/9 (44%) | +3.0% | -2.2% | 🟢 MONITOR: Stable performance |

### Generated Artifacts

**Visualizations:**
- `reports/drift/drift_analysis_mean_shift.png`
- `reports/drift/drift_analysis_variance_change.png`
- `reports/drift/drift_analysis_distribution_shift.png`

**Report:**
- `reports/drift/drift_report.json` - Complete metrics and recommendations

**Datasets:**
- `data/monitoring/drifted_data_mean_shift.csv`
- `data/monitoring/drifted_data_variance_change.csv`
- `data/monitoring/drifted_data_distribution_shift.csv`

### API Endpoint
```bash
# Detect drift via API
curl -X POST http://localhost:8000/detect-drift \
  -H "Content-Type: application/json" \
  -F "file=@data/monitoring/drifted_data_mean_shift.csv"
```

---

## Results

### Model Performance Comparison

| Metric | Baseline | Mean Shift | Variance Change | Distribution Shift |
|--------|----------|------------|-----------------|-------------------|
| **Accuracy** | 41.8% | 71.4% (+29.6%) | 64.6% (+22.8%) | 44.8% (+3.0%) |
| **Precision** | 95.7% | 100.0% (+4.3%) | 99.3% (+3.6%) | 90.6% (-5.1%) |
| **Recall** | 16.0% | 2.7% (-13.3%) | 43.6% (+27.6%) | 17.7% (+1.7%) |
| **F1-Score** | 27.4% | 5.3% (-22.1%) | 60.6% (+33.2%) | 29.6% (+2.2%) |

### Key Findings

1. **Mean Shift Drift:** Critical performance degradation despite higher accuracy
   - F1-score dropped by 22.1%
   - Model became too conservative (100% precision, 2.7% recall)
   - **Action:** Immediate retraining required
  
2. **Variance Change:** Performance actually improved
  - F1-score increased by 33.2%
  - Better balance between precision and recall
  - **Action:** Monitor but no immediate retraining

3. **Distribution Shift:** Minimal Impact
   - F1-score changed by only 2.2%
   - Model robust to categorical distribution changes
   - **Action:** Continue monitoring
  
### Test Coverage Results

```
Total Tests: 128
Passed: 115 (90%)
Skipped: 13 (10% - expected)
Failed: 0 (0%)

Test Coverage: 85%+
Execution Time: ~4 seconds
```

---

## Technologies Used

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **ML Framework** | scikit-learn | 1.7.2 | Model training & inference |
| **API Framework** | FastAPI | 0.115.0 | REST API |
| **Web Server** | Uvicorn | 0.25.0 | ASGI server |
| **Data Processing** | pandas | 2.1.4 | Data manipulation |
| **Numerical** | numpy | 1.26.2 | Numerical operations |
| **Validation** | Pydantic | 2.10.3 | Data validation |

### MLOps Tools

| Tool | Purpose |
|------|---------|
| **Evidently** | Data drift detection |
| **DVC** | Data & model versioning |
| **pytest** | Testing framework |
| **Docker** | Containerization |
| **Kubernetes** | Orchestration |
| **GitHub Actions** | CI/CD |

### Infrastructure

- **Container Registry:** DockerHub
- **Orchestration:** Kubernetes
- **Cloud-Ready:** GKE, EKS, AKS compatible
- **Monitoring:** Custom metrics + Evidently

---

## Project Structure

```
MLOps-Phase3/
├── app/                      # FastAPI application
│   ├── main.py              # API endpoints & routes
│   ├── models.py            # Pydantic models
│   ├── inference.py         # Model inference logic
│   ├── drift_detection.py   # Drift monitoring
│   └── config.py            # Configuration management
│
├── src/                      # Source code modules
│   ├── models/              # ML models & pipelines
│   ├── preprocessing/       # Data preprocessing
│   └── utils/               # Utility functions
│
├── tests/                    # Comprehensive test suite
│   ├── unit/                # Unit tests
│   │   ├── test_metrics.py
│   │   ├── test_model_inference.py
│   │   ├── test_preprocessing.py
│   │   ├── test_drift.py
│   │   └── test_config.py
│   ├── integration/         # Integration tests
│   │   ├── test_api.py
│   │   ├── test_dvc_stages.py
│   │   └── test_pipeline_e2e.py
│   └── conftest.py          # Shared fixtures
│
├── k8s/                      # Kubernetes manifests
│   ├── deployment.yaml      # Deployment configuration
│   ├── service.yaml         # Service & load balancing
│   ├── hpa.yaml             # Horizontal Pod Autoscaler
│   ├── configmap.yaml       # Configuration
│   ├── ingress.yaml         # Ingress rules
│   └── README.md            # Kubernetes guide
│
├── scripts/                  # Utility scripts
│   └── simulate_drift.py    # Drift simulation & detection
│
├── models/                   # Trained models (DVC tracked)
│   └── pipeline_baseline.pkl
│
├── data/                     # Data directories (DVC tracked)
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   ├── reference/           # Reference data for drift
│   └── monitoring/          # Monitoring data
│
├── reports/                  # Reports & visualizations
│   ├── drift/               # Drift detection reports
│   │   ├── drift_analysis_*.png
│   │   └── drift_report.json
│   └── figures/             # Other visualizations
│
├── config/                   # Configuration files
│   └── config.yaml
│
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose config
├── requirements-prod.txt    # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── pytest.ini               # Pytest configuration
├── params.yaml              # DVC parameters
├── dvc.yaml                 # DVC pipeline definition
└── README.md                # This file
```

---

## Model Information

### Model Details


| Property | Value |
|----------|-------|
| **Model Path** | `models/pipeline_baseline.pkl` |
| **Model Type** | Scikit-learn Pipeline |
| **Algorithm** | Logistic Regression |
| **Preprocessing** | StandardScaler |
| **Version** | 1.0.0 |
| **DVC Tracked** | Yes |

### Model Components
```python
Pipeline(steps=[
    ('standardscaler', StandardScaler()),
    ('logisticregression', LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])
```

### Features

**Input Features (9):**
- `Class_X_Percentage` - Class 10 percentage (0-100)
- `Class_XII_Percentage` - Class 12 percentage (0-100)
- `Study_Hours` - Daily study hours (0-12)
- `Gender_Male` - Gender indicator (binary)
- `Caste_General` - General caste category (binary)
- `Caste_OBC` - OBC category (binary)
- `Caste_SC` - SC category (binary)
- `Coaching_Yes` - Coaching attendance (binary)
- `Medium_English` - English medium (binary)

**Target Variable:**
- `0` - Low Performance
- `1` - High Performance

---

## Academic Context

**Course:** MLOps - Machine Learning Operations  
**Institution:** Tecnológico de Monterrey  
**Phase:** 3 - Production ML System  
**Trimester:** Sept-Nov 2025

### Requirements Met

**Requirement 1:** Comprehensive Testing Framework (115/128 passing)  
**Requirement 2:** Model Serving with FastAPI (6 endpoints)  
**Requirement 3:** Reproducibility Across Environments  
**Requirement 4:** Docker Containerization (Published to DockerHub)  
**Requirement 5:** Data Drift Detection & Monitoring  
**Bonus:** Kubernetes Orchestration with Auto-Scaling

---

## Links

- **GitHub Repository:** https://github.com/Lia1566/MLOps-Phase3
- **DockerHub Image:** https://hub.docker.com/r/a01566204/ml-service
- **API Documentation:** http://localhost:8000/docs (when running)
- **Kubernetes Guide:** [k8s/README.md](./k8s/README.md)

---

## Deployment Options

### Option 1: Local Development
```bash
uvicorn app.main:app --reload --port 8000
```

### Option 2: Docker
```bash
docker run -d -p 8000:8000 a01566204/ml-service:1.0.0
```

### Option 3: Kubernetes
```bash
kubectl apply -f k8s/
kubectl port-forward svc/ml-service 8000:80
```

---

## License

This project is for academic purposed as part of MLOps coursework at Tecnológico de Monterrey.

---

## Conclusion

This project demonstrates a **complete production-ready MLOps system** suitable for real-world deployment:

### Achievements
- **90% test pass rate** with comprehensive coverage
- **6 production API endpoints** with full documentation
- **Docker image published** to DockerHub for easy deployment
- **Kubernetes manifests** with auto-scaling (2-10 pods)
- **100% reproducibility** across environments (sklearn 1.7.2 matching)
- **Real-time drift detection** with 3 scenario simulations
- **Professional documentation** with examples and guides
















