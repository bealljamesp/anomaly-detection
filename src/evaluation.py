import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_curve, roc_auc_score


def evaluate_detector(
    y_true: npt.NDArray[np.int64],
    y_scores: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Compute ROC AUC and optimal F1 score metrics for anomaly ranking."""
    auc = roc_auc_score(y_true, -y_scores)
    precision, recall, _ = precision_recall_curve(y_true, -y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1 = float(np.max(f1_scores))

    return {
        "roc_auc": float(auc),
        "best_f1": best_f1,
    }
