[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pytest](https://img.shields.io/badge/pytest-100%25-green.svg)](https://docs.pytest.org/)

# Anomaly Detection Engine

## Overview
A production-grade, unsupervised outlier detection engine built for multi-dimensional financial transaction monitoring, fraud detection, and risk exception tracking. This repository implements an ensemble of complementary machine learning paradigms—global tree partitioning (Isolation Forest), local density estimation (Local Outlier Factor), and boundary-based spatial separation (One-Class SVM)—combined via majority voting consensus.

## Architectural Features
* **Multi-Model Ensemble:** Combines Isolation Forest, Local Outlier Factor (Novelty mode), and One-Class SVM with an institutional majority voting consensus ($\ge 2$ model agreement required for outlier flagging).
* **Vectorized Feature Engineering:** Computes robust Z-scores using median and Median Absolute Deviation (MAD) via contiguous NumPy and Pandas operations without explicit iteration loops.
* **Strict Modern Python Standards:** Built for Python 3.12+ with native lowercase generics (`list`, `dict`), pipe syntax (`|`), and zero-copy contiguous memory handling (`.to_numpy(dtype=np.float64, copy=False)`).
* **Automated Testing & Logging:** Integrated `pytest` test suite covering feature transformations and model output validation, paired with structured execution logging to console and file sinks.

## Repository Structure
```text
anomaly-detection/
├── .github/
│   └── workflows/
│       └── pytest.yml
├── data/
│   ├── raw/
│   └── processed/
├── logs/
│   └── execution.log
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── logger.py
│   ├── models.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── environment.yml
├── pyproject.toml
└── README.md

---

## Setup and Execute

conda env create -f environment.yml
conda activate anomaly-detection
python -m src.main
pytest -v

---

## Sample Output

```
Sample Flagged Anomalies:
     transaction_amount  account_velocity  geo_distance  consensus_prediction
10           8500.00000              25.0    150.000000                    -1
55           8500.00000              25.0    150.000000                    -1
120          8500.00000              25.0    150.000000                    -1
155           712.15159               9.0      0.059969                    -1
245           345.50929               2.0     41.856218                    -1
```
