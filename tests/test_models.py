from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import compute_robust_zscores
from src.models import ComprehensiveAnomalyEnsemble


def test_robust_zscores() -> None:
    """Verify vectorized robust Z-score calculation without explicit loops."""
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 100.0],
            "col2": [10.0, 11.0, 12.0, 13.0, 200.0],
        }
    )
    z_scores = compute_robust_zscores(df, ["col1", "col2"])

    assert isinstance(z_scores, pd.DataFrame)
    assert z_scores.shape == (5, 2)
    # Check that extreme outlier yields large magnitude robust z-score
    assert z_scores.loc[4, "col1"] > 3.0


def test_comprehensive_ensemble_pipeline() -> None:
    """Verify model fitting, output shapes, and consensus classification bounds."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=(200, 3))
    x[0:3, :] += 15.0  # Inject explicit outliers

    ensemble = ComprehensiveAnomalyEnsemble(contamination=0.02)
    ensemble.fit(x)
    preds = ensemble.predict(x)

    assert "isolation_forest" in preds
    assert "local_outlier_factor" in preds
    assert "one_class_svm" in preds
    assert "consensus" in preds

    for model_name, arr in preds.items():
        assert arr.shape == (200,)
        assert set(np.unique(arr)).issubset({-1, 1})
