"""Accuracy-assessment analysis: standardization, confusion matrix, accuracies.

Olofsson et al. 2014. Ported from the openforis/accuracy-assessment backend
(io.py, compute.py, accuracy.py) with two deliberate corrections:
- the confusion matrix counts samples (n_ij) rather than summing the per-sample
  `area` column, so n_i, SE/CI and the reported sample counts stay valid;
- accuracies use the map-union-reference class set with full margins, keeping
  reference-only classes (the reference squares on map classes only).
See docs/analysis_port_report.md. Areas carry the input's native unit.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def standardize_reference(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename user columns to canonical map_code/ref_code/area (+ optional x/y).

    Other columns are preserved (e.g. a filter column). If no sample-area column
    is mapped, every sample gets area = 1.0.
    """
    out = df.copy()
    renames = {}
    if mapping.get("map"):
        renames[mapping["map"]] = "map_code"
    if mapping.get("ref"):
        renames[mapping["ref"]] = "ref_code"
    if mapping.get("x") and mapping["x"] in out.columns:
        renames[mapping["x"]] = "x"
    if mapping.get("y") and mapping["y"] in out.columns:
        renames[mapping["y"]] = "y"
    if mapping.get("sample_area") and mapping["sample_area"] in out.columns:
        renames[mapping["sample_area"]] = "area"
    out = out.rename(columns=renames)
    if "area" not in out.columns:
        out["area"] = 1.0
    out["area"] = pd.to_numeric(out["area"], errors="coerce").fillna(0.0)
    return out


def standardize_area(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename the area/strata file to canonical map_code/map_area."""
    out = df.rename(
        columns={
            mapping.get("area_class", "map_code"): "map_code",
            mapping.get("area_value", "map_area"): "map_area",
        }
    ).copy()
    out["map_area"] = pd.to_numeric(out["map_area"], errors="coerce")
    return out


def apply_filter(df: pd.DataFrame, filter_spec: Optional[dict]) -> pd.DataFrame:
    """Subset rows where filter column value is in include_values.

    None spec -> passthrough. Unknown column -> passthrough (caller may warn).
    """
    if not filter_spec:
        return df
    col = filter_spec.get("column")
    if not col or col not in df.columns:
        return df
    vals = filter_spec.get("include_values", [])
    if not vals:
        return df
    # Compare as strings so a numeric column matches string values from the UI.
    return df[df[col].astype(str).isin([str(v) for v in vals])].copy()


def legends(df_ref: pd.DataFrame) -> Tuple[list, list]:
    """Return (map_legend, ref_legend): sorted unique class codes."""
    map_legend = sorted(pd.Series(df_ref["map_code"].dropna().unique()).tolist())
    ref_legend = sorted(pd.Series(df_ref["ref_code"].dropna().unique()).tolist())
    return map_legend, ref_legend


def confusion_matrix_counts(
    df_ref: pd.DataFrame, map_legend: Sequence, ref_legend: Sequence
) -> pd.DataFrame:
    """Sample COUNTS n_ij; rows = map classes, cols = reference classes.

    Cells are the number of reference samples (not summed plot areas): the
    Olofsson estimator, its variance denominator ``n_i - 1`` and the reported
    sample counts must all be true sample counts. A per-sample plot-area column
    does not weight the estimator (that would produce invalid SE/CI/counts).

    Reindexed on BOTH full legends (fill 0) so reference-only classes are kept.
    """
    return (
        df_ref.pivot_table(
            index="map_code",
            columns="ref_code",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(index=list(map_legend), columns=list(ref_legend), fill_value=0)
        .astype(float)
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide, coercing division by zero (0/0 or x/0) to 0.0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (numerator / denominator.replace({0.0: np.nan})).fillna(0.0)


def accuracies_from_matrices(
    matrix_area: pd.DataFrame, pij: pd.DataFrame
) -> pd.DataFrame:
    """Producer's, User's, Weighted-Producer's accuracy per class (0..1).

    Uses FULL row/column margins so reference-only or map-only classes are not
    dropped: User's accuracy denominator is the full row sum over ALL reference
    classes; Producer's is the full column sum over ALL map classes. 0/0 -> 0.0.
    """
    all_classes = sorted(set(matrix_area.index) | set(matrix_area.columns))

    def _diag(m: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                c: float(m.loc[c, c]) if (c in m.index and c in m.columns) else 0.0
                for c in all_classes
            }
        )

    row_sums = matrix_area.sum(axis=1).reindex(all_classes).fillna(0.0)
    col_sums = matrix_area.sum(axis=0).reindex(all_classes).fillna(0.0)
    diag = _diag(matrix_area)
    users = _safe_divide(diag, row_sums)
    producers = _safe_divide(diag, col_sums)

    col_sums_w = pij.sum(axis=0).reindex(all_classes).fillna(0.0)
    diag_w = _diag(pij)
    weighted_prod = _safe_divide(diag_w, col_sums_w)

    return pd.DataFrame(
        {
            "producers_accuracy": producers,
            "users_accuracy": users,
            "weighted_producers_accuracy": weighted_prod,
        }
    )


def overall_accuracy(pij: pd.DataFrame) -> float:
    """Overall accuracy = sum of the diagonal of the area-proportion matrix."""
    classes = sorted(set(pij.index) | set(pij.columns))
    m = pij.reindex(index=classes, columns=classes, fill_value=0.0)
    return float(np.trace(m.values))


def convert_area(value: float, unit: str) -> float:
    """Convert a native (m2) area for display. 'ha' -> /10000; else identity."""
    if unit == "ha":
        return value / 10000.0
    return value
