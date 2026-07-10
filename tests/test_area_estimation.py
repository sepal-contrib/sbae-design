"""Tests for stratified area estimation (Olofsson 2014) + SRS comparator.

Encodes the 4 verified R bugs as regression tests (see spec section 6/7).
"""

import numpy as np
import pandas as pd
import pytest

from component.scripts.accuracy import confusion_matrix_counts, standardize_reference
from component.scripts.area_estimation import (
    pij_matrix,
    srs_estimates,
    stratified_area_estimates,
    stratum_weights,
)


def _ref_2x2():
    rows = [(1, 1)] * 45 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 40
    return standardize_reference(
        pd.DataFrame(rows, columns=["m", "r"]), {"map": "m", "ref": "r"}
    )


def _matrix_2x2():
    return confusion_matrix_counts(_ref_2x2(), [1, 2], [1, 2])


def _area(order=(1, 2), values=(60.0, 40.0)):
    return pd.DataFrame({"map_code": list(order), "map_area": list(values)})


def test_stratum_weights_by_class_and_total():
    w, a_total = stratum_weights(_area())
    assert a_total == 100.0
    assert np.isclose(w.loc[1], 0.6) and np.isclose(w.loc[2], 0.4)


def test_stratum_weights_zero_total_raises():
    with pytest.raises(ValueError):
        stratum_weights(pd.DataFrame({"map_code": [1], "map_area": [0.0]}))


def test_pij_matrix_values():
    w, _ = stratum_weights(_area())
    pij = pij_matrix(_matrix_2x2(), w)
    assert np.isclose(pij.loc[1, 1], 0.54)
    assert np.isclose(pij.loc[1, 2], 0.06)
    assert np.isclose(pij.loc[2, 1], 0.08)
    assert np.isclose(pij.loc[2, 2], 0.32)


def test_pij_weights_name_matched_not_positional():
    """Bug #1: area file rows in reversed order must still weight by class."""
    w_sorted, _ = stratum_weights(_area(order=(1, 2), values=(60.0, 40.0)))
    w_rev, _ = stratum_weights(_area(order=(2, 1), values=(40.0, 60.0)))
    pij_sorted = pij_matrix(_matrix_2x2(), w_sorted)
    pij_rev = pij_matrix(_matrix_2x2(), w_rev)
    pd.testing.assert_frame_equal(pij_sorted, pij_rev)


def test_stratified_area_estimates_values():
    w, a_total = stratum_weights(_area())
    pij = pij_matrix(_matrix_2x2(), w)
    out = stratified_area_estimates(pij, _matrix_2x2(), a_total, z=1.96)
    assert np.isclose(out.loc[1, "area_estimate"], 62.0)
    assert np.isclose(out.loc[2, "area_estimate"], 38.0)
    assert np.isclose(out.loc[1, "standard_error"], 3.44045, atol=1e-4)
    assert np.isclose(out.loc[1, "confidence_interval"], 3.44045 * 1.96, atol=1e-3)


def test_error_adjusted_equals_column_sum():
    """Bug #2 resolution: single estimate == column sum * A_total (Eq. 8)."""
    w, a_total = stratum_weights(_area())
    pij = pij_matrix(_matrix_2x2(), w)
    out = stratified_area_estimates(pij, _matrix_2x2(), a_total, z=1.96)
    expected = pij.sum(axis=0) * a_total
    assert np.allclose(out["area_estimate"].values, expected.values)


def test_srs_uses_reference_column_sums():
    """Bug #3 + risk R1: SRS from reference-class column sums / total area."""
    out = srs_estimates(_matrix_2x2(), A_total=100.0, z=1.96)
    assert np.isclose(out.loc[1, "srs_weight"], 0.55)
    assert np.isclose(out.loc[2, "srs_weight"], 0.45)
    assert np.isclose(out.loc[1, "srs_area_estimate"], 55.0)
    assert np.isclose(
        out.loc[1, "srs_standard_error"], np.sqrt(0.45 * 0.55 / 100.0) * 100.0
    )


def test_srs_weights_sum_to_one():
    """SRS weights sum to 1 and SE is finite; a per-plot area column is ignored."""
    rows = [(1, 1, 10.0)] * 4 + [(1, 2, 30.0)] + [(2, 1, 5.0)] + [(2, 2, 15.0)] * 3
    df = standardize_reference(
        pd.DataFrame(rows, columns=["m", "r", "plot"]),
        {"map": "m", "ref": "r", "sample_area": "plot"},
    )
    m = confusion_matrix_counts(df, [1, 2], [1, 2])
    out = srs_estimates(m, A_total=100.0, z=1.96)
    assert np.isclose(out["srs_weight"].sum(), 1.0)
    assert out["srs_standard_error"].notna().all()


def test_matrix_is_sample_counts_not_summed_plot_area():
    """Blocker: a per-sample area column must not inflate n_i / sample counts.

    Two samples in stratum 1, each with plot area 50, must count as n_i = 2, not
    100 (summed area), otherwise SE/CI and the reported sample counts are wrong.
    """
    rows = [(1, 1, 50.0), (1, 1, 50.0), (1, 2, 50.0)]
    df = standardize_reference(
        pd.DataFrame(rows, columns=["m", "r", "plot"]),
        {"map": "m", "ref": "r", "sample_area": "plot"},
    )
    m = confusion_matrix_counts(df, [1, 2], [1, 2])
    assert m.loc[1, 1] == 2.0 and m.loc[1, 2] == 1.0
    assert m.loc[1].sum() == 3.0  # stratum-1 count, not 150
