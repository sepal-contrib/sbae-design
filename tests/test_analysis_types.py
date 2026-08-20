"""Tests for analysis typed contracts."""

import pandas as pd

from component.analysis.types import (
    AccuracyRow,
    AnalysisInputs,
    AnalysisMethod,
    AnalysisResults,
    ClassEstimate,
)


def test_method_from_string():
    assert AnalysisMethod.from_string("stratified_estimation") is (
        AnalysisMethod.STRATIFIED_ESTIMATION
    )


def test_inputs_confidence_decimal():
    inp = AnalysisInputs(
        method=AnalysisMethod.STRATIFIED_ESTIMATION,
        reference_df=pd.DataFrame(),
        area_data=pd.DataFrame(),
        confidence_level=95.0,
        column_mapping={},
    )
    assert inp.confidence_level_decimal == 0.95


def test_error_result():
    r = AnalysisResults.error(AnalysisMethod.STRATIFIED_ESTIMATION, "boom")
    assert r.success is False and r.error_message == "boom"
    d = r.to_dict()
    assert d["success"] is False and d["error_message"] == "boom"


def test_to_dict_serializes_matrix_and_rows():
    m = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=[1, 2], columns=[1, 2])
    r = AnalysisResults(
        method=AnalysisMethod.STRATIFIED_ESTIMATION,
        confusion_matrix=m,
        class_estimates=[
            ClassEstimate(
                map_code=1,
                class_name="Forest",
                number_samples=50.0,
                map_pixel_count=60.0,
                area_estimate=62.0,
                standard_error=3.44,
                confidence_interval=6.74,
                srs_area_estimate=55.0,
                srs_standard_error=0.05,
                srs_confidence_interval=9.8,
            )
        ],
        accuracy_rows=[
            AccuracyRow(
                map_code=1,
                class_name="Forest",
                users_accuracy=0.9,
                producers_accuracy=0.81,
                weighted_producers_accuracy=0.87,
            )
        ],
        overall_accuracy=0.86,
        total_area=100.0,
        confidence_level=95.0,
        z=1.96,
        map_legend=[1, 2],
        ref_legend=[1, 2],
    )
    d = r.to_dict()
    assert d["confusion_matrix"] == {
        "index": [1, 2],
        "columns": [1, 2],
        "data": [[1.0, 2.0], [3.0, 4.0]],
    }
    assert d["class_estimates"][0]["area_estimate"] == 62.0
    assert d["accuracy_rows"][0]["users_accuracy"] == 0.9
    assert d["overall_accuracy"] == 0.86
