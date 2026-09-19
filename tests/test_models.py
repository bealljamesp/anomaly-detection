import numpy as np

from src.models import ComprehensiveAnomalyEnsemble


def test_anomaly_ensemble_shape() -> None:
    """Verify model fitting and prediction output dimensions and value ranges."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=(500, 5))
    x[0:5, :] += 10.0  # Inject synthetic outliers

    model = ComprehensiveAnomalyEnsemble(contamination=0.02)
    model.fit(x)
    predictions = model.predict(x)

    assert "consensus" in predictions
    assert predictions["consensus"].shape == (500,)
    assert set(np.unique(predictions["consensus"])).issubset({-1, 1})
