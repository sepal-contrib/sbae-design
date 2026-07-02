"""Pure functions for accuracy-assessment analysis: standardization, confusion
matrix, and accuracy metrics (Olofsson et al. 2014).

Vendored and de-bugged from openforis/accuracy-assessment backend
(io.py, compute.py, accuracy.py). Areas carry the input's native unit.
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


def confusion_matrix_area(
    df_ref: pd.DataFrame, map_legend: Sequence, ref_legend: Sequence
) -> pd.DataFrame:
    """Summed per-sample areas; rows = map classes, cols = reference classes.

    Reindexed on BOTH full legends (fill 0) so reference-only classes are kept.
    """
    return (
        df_ref.pivot_table(
            index="map_code",
            columns="ref_code",
            values="area",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=list(map_legend), columns=list(ref_legend), fill_value=0.0)
        .astype(float)
    )


def accuracies_from_matrices(
    matrix_area: pd.DataFrame, pij: pd.DataFrame
) -> pd.DataFrame:
    """Producer's, User's, Weighted-Producer's accuracy per class (0..1).

    0/0 (class absent from a margin) -> 0.0, never NaN.
    """
    classes = list(matrix_area.index)
    m = matrix_area.reindex(index=classes, columns=classes, fill_value=0.0)
    diag = pd.Series(np.diag(m.values), index=classes)
    col_sums = m.sum(axis=0)
    row_sums = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        producers = (diag / col_sums.replace({0.0: np.nan})).fillna(0.0)
        users = (diag / row_sums.replace({0.0: np.nan})).fillna(0.0)

    pij_sq = pij.reindex(index=classes, columns=classes, fill_value=0.0)
    diag_w = pd.Series(np.diag(pij_sq.values), index=classes)
    col_sums_w = pij_sq.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_prod = (diag_w / col_sums_w.replace({0.0: np.nan})).fillna(0.0)

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
