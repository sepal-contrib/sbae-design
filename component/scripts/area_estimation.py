"""Stratified area estimation (Olofsson et al. 2014) and SRS comparator.

Vendored and de-bugged from openforis/accuracy-assessment backend/compute.py:
- weights are name-keyed (bug #1);
- the single error-adjusted estimate IS the column sum * A_total, Eq. 8 (bug #2);
- SRS uses reference-class column sums over total sample area (bug #3 + risk R1);
- divide-by-zero coerces to 0; the dead se2_terms block is removed.
Areas carry the input's native unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratum_weights(df_area: pd.DataFrame) -> tuple[pd.Series, float]:
    """W_i = A_i / A_total, keyed by map_code (never by row position)."""
    area_by_class = df_area.groupby("map_code", as_index=True)["map_area"].sum()
    a_total = float(area_by_class.sum())
    if a_total <= 0:
        raise ValueError("Total map area must be > 0")
    return area_by_class / a_total, a_total


def pij_matrix(matrix_area: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """p_ij = w_i * (n_ij / n_i.), weights aligned to matrix rows BY LABEL."""
    row_sums = matrix_area.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = matrix_area.div(row_sums.replace({0.0: np.nan}), axis=0).fillna(0.0)
    return frac.mul(w.reindex(matrix_area.index).fillna(0.0), axis=0)


def stratified_area_estimates(
    pij: pd.DataFrame, matrix_area: pd.DataFrame, A_total: float, z: float
) -> pd.DataFrame:
    """Error-adjusted area per reference class (Eq. 8) + SE (Eq. 10) + CI.

    area_estimate_j = (sum_i p_ij) * A_total
    SE(p_j) = sqrt( sum_i w_i^2 * [f_ij (1 - f_ij) / (n_i - 1)] ),
      f_ij = n_ij / n_i. (row fraction); n_i. is the row area-sum (== count when area=1).
    """
    p_hat = pij.sum(axis=0)          # column sum over map classes i -> per ref class j
    w = pij.sum(axis=1)              # equals stratum weight W_i
    row_sums = matrix_area.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_frac = matrix_area.div(row_sums.replace({0.0: np.nan}), axis=0).fillna(0.0)

    se2 = []
    for j in pij.columns:
        term = (w.pow(2)) * (
            row_frac[j] * (1.0 - row_frac[j]) / (row_sums.clip(lower=1.0) - 1.0)
        )
        term = term.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        se2.append(term.sum())
    se_prop = pd.Series(se2, index=pij.columns).clip(lower=0.0).pow(0.5)

    area_est = p_hat * A_total
    se_area = se_prop * A_total
    return pd.DataFrame(
        {
            "area_estimate": area_est,
            "standard_error": se_area,
            "confidence_interval": se_area * z,
        }
    )


def srs_estimates(matrix_area: pd.DataFrame, A_total: float, z: float) -> pd.DataFrame:
    """Simple-random comparator per reference class.

    p_j = (sum_i n_ij) / (sum_ij n_ij)  [reference-class column sum / total sample area]
    SE  = sqrt(p_j (1 - p_j) / N),  N = total sample area (== sample count when area=1).
    """
    col_sums = matrix_area.sum(axis=0)      # per reference class j
    N = float(matrix_area.values.sum())
    denom = N if N else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = col_sums / denom
    se_prop = np.sqrt((1.0 - weight) * weight / denom)
    return pd.DataFrame(
        {
            "srs_weight": weight,
            "srs_area_estimate": weight * A_total,
            "srs_standard_error": se_prop,
            "srs_confidence_interval": se_prop * z * A_total,
        }
    )
