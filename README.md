# Machine Learning & Fraud/Anomaly Detection

[![CI Test Suite](https://github.com/bealljamesp/anomaly-detection-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/bealljamesp/anomaly-detection-engine/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Overview

The **Anomaly Detection Engine** is a modular, high-throughput Python framework designed to ingest, process, and evaluate high-dimensional transactional ledgers for latent anomalies, system logic failures, and compliance drift.

Built with an emphasis on **rigorous MLOps governance**, this project demonstrates end-to-end data pipeline vectorization using **Polars**, unsupervised machine learning via **Isolation Forests**, automated unit testing with **Pytest**, continuous integration through **GitHub Actions**, cloud-native file storage ingestion (`pyarrow`/`s3fs`), containerization via **Docker**, and model interpretability using **SHAP (SHapley Additive exPlanations)**.

This architecture reflects production oversight analytics applied in high-stakes auditing, risk scoring, and federal data governance environments (e.g., DHS Office of Inspector General).

---

## 🛠️ Key Features & Technical Stack

* **High-Throughput Vectorized Ingestion:** Leverages `Polars` for $O(1)$ memory allocation profiles and ultra-fast DataFrame transformations over transactional datasets.
* **Unsupervised Anomaly Detection:** Implements an `IsolationForest` model to detect multivariate outliers, risk score transactional patterns, and flag statistical anomalies.
* **Model Explainability & Auditability:** Integrates SHAP game-theoretic feature attribution to decompose anomaly scores into transparent, human-readable explanations.
* **Production MLOps Governance:**
  * Automated testing suite using `pytest` for pipeline validation and contamination assertions.
  * Continuous Integration (CI) pipeline powered by `GitHub Actions` running Python 3.12.
  * Modern Python packaging structured around `pyproject.toml` standards.

---

## 🛠️ Repository Architecture

```text
anomaly-detection-engine/
├── .github/
│   └── workflows/
│       └── tests.yml           # Automated CI/CD pipeline running pytest on Python 3.12
├── src/
│   └── anomaly_engine/
│       ├── __init__.py         # Package entry point
│       ├── pipeline.py        # Polars-based data generation & feature preprocessing
│       ├── model.py           # Isolation Forest anomaly detection engine
│       └── explainability.py  # SHAP game-theoretic model interpretability module
├── tests/
│   └── test_pipeline.py       # Pytest suite for unit & integration testing
├── main.py                    # Pipeline execution script
├── pyproject.toml             # Package dependencies and dev configurations
└── README.md                  # Technical documentation
```
---

## 🔍 Sample Output

```
🚀 Running Anomaly Detection Pipeline...
Ingested 5000 records successfully.
⚠️ Detected 125 anomalous transactions.
shape: (5, 6)
┌────────────────┬─────────────┬─────────────────┬────────────────┬──────────────┬───────────────┐
│ transaction_id ┆ amount      ┆ daily_txn_count ┆ raw_risk_score ┆ anomaly_flag ┆ anomaly_score │
│ ---            ┆ ---         ┆ ---             ┆ ---            ┆ ---          ┆ ---           │
│ str            ┆ f64         ┆ i32             ┆ f64            ┆ bool         ┆ f64           │
╞════════════════╪═════════════╪═════════════════╪════════════════╪══════════════╪═══════════════╡
│ TXN-000018     ┆ 333.981371  ┆ 26              ┆ 76.257703      ┆ true         ┆ -0.005938     │
│ TXN-000056     ┆ 57.952765   ┆ 48              ┆ 93.922475      ┆ true         ┆ -0.076843     │
│ TXN-000088     ┆ 1498.813633 ┆ 39              ┆ 98.653787      ┆ true         ┆ -0.069733     │
│ TXN-000104     ┆ 1634.895048 ┆ 28              ┆ 82.129535      ┆ true         ┆ -0.051309     │
│ TXN-000155     ┆ 381.8864    ┆ 34              ┆ 72.890922      ┆ true         ┆ -0.02428      │
└────────────────┴─────────────┴─────────────────┴────────────────┴──────────────┴───────────────┘

--- BASELINE (NORMAL) TRANSACTIONS ---
shape: (1, 3)
┌───────────┬─────────────────┬────────────────┐
│ amount    ┆ daily_txn_count ┆ raw_risk_score │
│ ---       ┆ ---             ┆ ---            │
│ f64       ┆ f64             ┆ f64            │
╞═══════════╪═════════════════╪════════════════╡
│ 98.942182 ┆ 5.018051        ┆ 49.842573      │
└───────────┴─────────────────┴────────────────┘

--- FLAGGED ANOMALOUS TRANSACTIONS ---
shape: (1, 3)
┌────────────┬─────────────────┬────────────────┐
│ amount     ┆ daily_txn_count ┆ raw_risk_score │
│ ---        ┆ ---             ┆ ---            │
│ f64        ┆ f64             ┆ f64            │
╞════════════╪═════════════════╪════════════════╡
│ 1071.61075 ┆ 38.616          ┆ 88.519353      │
└────────────┴─────────────────┴────────────────┘
```
