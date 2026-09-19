import numpy as np
import pandas as pd


def compute_robust_zscores(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute robust z-scores using median and median absolute deviation (MAD)

    via vectorized pandas operations without explicit loops.
    """
    sub = data.loc[:, columns]
    medians = sub.median()
    mads = (sub - medians).abs().median()
    scales = np.where(mads > 0, 1.4826 * mads, 1.0)
    return (sub - medians) / scales
