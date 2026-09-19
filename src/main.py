from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import compute_robust_zscores
from src.logger import setup_logger
from src.models import ComprehensiveAnomalyEnsemble

logger = setup_logger("anomaly_main")


def generate_synthetic_transactions(
    n_samples: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic financial transaction features with injected outliers."""
    rng = np.random.default_rng(seed)

    transaction_amount = rng.normal(loc=500.0, scale=120.0, size=n_samples)
    account_velocity = rng.poisson(lam=3.0, size=n_samples).astype(np.float64)
    geo_distance = rng.exponential(scale=5.0, size=n_samples)

    df = pd.DataFrame(
        {
            "transaction_amount": transaction_amount,
            "account_velocity": account_velocity,
            "geo_distance": geo_distance,
        }
    )

    anomaly_indices = [10, 55, 120, 250, 500]
    df.loc[anomaly_indices, "transaction_amount"] = 8500.0
    df.loc[anomaly_indices, "account_velocity"] = 25.0
    df.loc[anomaly_indices, "geo_distance"] = 150.0

    return df


def main() -> None:
    logger.info("Initializing synthetic financial transaction dataset generation...")
    df = generate_synthetic_transactions(n_samples=1000)

    logger.info("Computing robust Z-scores via vectorized operations...")
    features = ["transaction_amount", "account_velocity", "geo_distance"]
    z_scores = compute_robust_zscores(df, features)

    x_matrix = z_scores.to_numpy(dtype=np.float64, copy=False)

    logger.info(
        "Fitting Comprehensive Anomaly Ensemble (Isolation Forest, LOF, One-Class SVM)..."
    )
    ensemble = ComprehensiveAnomalyEnsemble(contamination=0.01)
    ensemble.fit(x_matrix)

    logger.info("Generating predictions and majority voting consensus...")
    predictions = ensemble.predict(x_matrix)

    df["consensus_prediction"] = predictions["consensus"]
    anomalies_detected = int((df["consensus_prediction"] == -1).sum())

    logger.info(
        f"Execution complete. Evaluated: {len(df)} transactions | Outliers Flagged: {anomalies_detected}"
    )
    print("\nSample Flagged Anomalies:")
    print(df[df["consensus_prediction"] == -1].head())


if __name__ == "__main__":
    main()
