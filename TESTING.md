# Testing Guide

## Run All Tests
```bash
pytest -q
```

## Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# API tests only
pytest tests/integration/test_api.py -v
```

## Run with Coverage
```bash
pytest tests/ --cov=src --cov=app --cov-report=term
```

## Test Results
- 111 tests passing
- 87% pass rate
- Covers: preprocessing, metrics, inference, API, pipeline
