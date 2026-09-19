from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class ComprehensiveAnomalyEnsemble:
    """
    Multi-model anomaly detection engine incorporating Isolation Forest,
    Local Outlier Factor, and OneClass SVM for institutional financial fraud detection.
    """

    def __init__(self, contamination: float = 0.01) -> None:
        self.contamination = contamination
        self.scaler = StandardScaler()

        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.oc_svm = OneClassSVM(
            nu=self.contamination,
            kernel="rbf",
            gamma="scale",
        )

    def fit(self, x: npt.NDArray[np.float64]) -> ComprehensiveAnomalyEnsemble:
        """Fit scaler and all unsupervised outlier detection estimators."""
        x_scaled = self.scaler.fit_transform(x)
        self.iso_forest.fit(x_scaled)
        self.lof.fit(x_scaled)
        self.oc_svm.fit(x_scaled)
        return self

    def predict(self, x: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.int64]]:
        """
        Generate binary predictions (-1 for anomaly, 1 for normal) across all models
        and compute a majority voting consensus.
        """
        x_scaled = self.scaler.transform(x)
        p_iso = self.iso_forest.predict(x_scaled)
        p_lof = self.lof.predict(x_scaled)
        p_svm = self.oc_svm.predict(x_scaled)

        stacked = np.column_stack((p_iso, p_lof, p_svm))
        consensus = np.where(np.sum(stacked == -1, axis=1) >= 2, -1, 1)

        return {
            "isolation_forest": p_iso,
            "local_outlier_factor": p_lof,
            "one_class_svm": p_svm,
            "consensus": consensus,
        }
