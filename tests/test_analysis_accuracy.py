"""Tests for accuracy-assessment analysis math (confusion matrix + accuracies)."""

import numpy as np
import pandas as pd

from component.scripts.accuracy import (
    accuracies_from_matrices,
    apply_filter,
    confusion_matrix_area,
    convert_area,
    legends,
    overall_accuracy,
    standardize_area,
    standardize_reference,
)


def _ref_2x2():
    """Reference rows reproducing the shared 2x2 fixture (area=1)."""
    rows = [(1, 1)] * 45 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 40
    return pd.DataFrame(rows, columns=["mapclass", "refclass"])


def _area_2x2():
    return pd.DataFrame({"code": [1, 2], "areapx": [60.0, 40.0]})


def test_standardize_reference_renames_and_defaults_area():
    df = _ref_2x2()
    out = standardize_reference(df, {"map": "mapclass", "ref": "refclass"})
    assert {"map_code", "ref_code", "area"}.issubset(out.columns)
    assert (out["area"] == 1.0).all()


def test_standardize_reference_uses_sample_area_col():
    df = _ref_2x2().assign(plot=2.0)
    out = standardize_reference(
        df, {"map": "mapclass", "ref": "refclass", "sample_area": "plot"}
    )
    assert (out["area"] == 2.0).all()


def test_standardize_area_renames():
    out = standardize_area(_area_2x2(), {"area_class": "code", "area_value": "areapx"})
    assert list(out.columns[:2]) == ["map_code", "map_area"] or {
        "map_code",
        "map_area",
    }.issubset(out.columns)
    assert out["map_area"].tolist() == [60.0, 40.0]


def test_apply_filter_subsets_and_none_passthrough():
    df = _ref_2x2().assign(strata="a")
    df.loc[:9, "strata"] = "b"
    assert len(apply_filter(df, None)) == len(df)
    only_a = apply_filter(df, {"column": "strata", "include_values": ["a"]})
    assert (only_a["strata"] == "a").all()
    # unknown column -> passthrough (no crash)
    assert len(apply_filter(df, {"column": "nope", "include_values": ["x"]})) == len(df)


def test_legends_sorted_unique():
    std = standardize_reference(_ref_2x2(), {"map": "mapclass", "ref": "refclass"})
    map_legend, ref_legend = legends(std)
    assert map_legend == [1, 2]
    assert ref_legend == [1, 2]


def test_confusion_matrix_area_counts():
    std = standardize_reference(_ref_2x2(), {"map": "mapclass", "ref": "refclass"})
    m = confusion_matrix_area(std, [1, 2], [1, 2])
    assert m.loc[1, 1] == 45 and m.loc[1, 2] == 5
    assert m.loc[2, 1] == 10 and m.loc[2, 2] == 40


def test_accuracies_match_hand_computation():
    std = standardize_reference(_ref_2x2(), {"map": "mapclass", "ref": "refclass"})
    m = confusion_matrix_area(std, [1, 2], [1, 2])
    pij = pd.DataFrame(
        [[0.54, 0.06], [0.08, 0.32]], index=[1, 2], columns=[1, 2]
    )
    acc = accuracies_from_matrices(m, pij)
    assert np.isclose(acc.loc[1, "users_accuracy"], 0.9)
    assert np.isclose(acc.loc[2, "users_accuracy"], 0.8)
    assert np.isclose(acc.loc[1, "producers_accuracy"], 45 / 55)
    assert np.isclose(acc.loc[2, "producers_accuracy"], 40 / 45)
    assert np.isclose(acc.loc[1, "weighted_producers_accuracy"], 0.54 / 0.62)
    assert np.isclose(acc.loc[2, "weighted_producers_accuracy"], 0.32 / 0.38)


def test_producers_accuracy_zero_not_nan_for_absent_reference_class():
    # map class 2 never appears as a reference label -> col sum 0 -> PA=0 (not NaN)
    m = pd.DataFrame([[10.0, 0.0], [3.0, 0.0]], index=[1, 2], columns=[1, 2])
    pij = pd.DataFrame([[0.5, 0.0], [0.3, 0.0]], index=[1, 2], columns=[1, 2])
    acc = accuracies_from_matrices(m, pij)
    assert acc.loc[2, "producers_accuracy"] == 0.0
    assert acc.loc[2, "weighted_producers_accuracy"] == 0.0


def test_overall_accuracy_is_diagonal_sum():
    pij = pd.DataFrame([[0.54, 0.06], [0.08, 0.32]], index=[1, 2], columns=[1, 2])
    assert np.isclose(overall_accuracy(pij), 0.86)


def test_convert_area_hectares():
    assert convert_area(10000.0, "ha") == 1.0
    assert convert_area(1234.0, "m2") == 1234.0
