# tests/test_analysis_state.py
"""Tests for analysis app-state reactives, setters and CSV exports."""

import pandas as pd

from component.model.state_manager import AppState


def _results_dict():
    return {
        "method": "stratified_estimation",
        "success": True,
        "confusion_matrix": {
            "index": [1, 2],
            "columns": [1, 2],
            "data": [[45.0, 5.0], [10.0, 40.0]],
        },
        "class_estimates": [
            {
                "map_code": 1,
                "class_name": "Forest",
                "number_samples": 50.0,
                "map_pixel_count": 60.0,
                "area_estimate": 62.0,
                "standard_error": 3.44,
                "confidence_interval": 6.74,
                "srs_area_estimate": 55.0,
                "srs_standard_error": 0.05,
                "srs_confidence_interval": 9.8,
            }
        ],
        "accuracy_rows": [
            {
                "map_code": 1,
                "class_name": "Forest",
                "users_accuracy": 0.9,
                "producers_accuracy": 0.81,
                "weighted_producers_accuracy": 0.87,
            }
        ],
        "overall_accuracy": 0.86,
        "total_area": 100.0,
        "area_unit": "ha",
        "confidence_level": 95.0,
        "z": 1.96,
        "map_legend": [1, 2],
        "ref_legend": [1, 2],
    }


def test_defaults_present():
    st = AppState()
    assert st.analysis_reference_df.value.empty
    assert st.analysis_area_source.value == "design"
    assert st.analysis_confidence_level.value == 95.0
    assert st.analysis_area_unit.value == "ha"
    assert st.analysis_results.value is None


def test_set_and_clear_analysis_results():
    st = AppState()
    st.set_analysis_results(_results_dict())
    assert st.analysis_results.value["overall_accuracy"] == 0.86
    st.clear_analysis_data()
    assert st.analysis_results.value is None
    assert st.analysis_reference_df.value.empty


def test_export_confusion_matrix_csv():
    st = AppState()
    st.set_analysis_results(_results_dict())
    csv = st.export_confusion_matrix_csv()
    assert "45" in csv and "40" in csv


def test_export_area_estimates_csv_has_headers():
    st = AppState()
    st.set_analysis_results(_results_dict())
    csv = st.export_area_estimates_csv()
    assert "area_estimate" in csv and "Forest" in csv


def test_exports_empty_when_no_results():
    st = AppState()
    assert st.export_confusion_matrix_csv() == ""
    assert st.export_area_estimates_csv() == ""
    assert st.export_accuracy_csv() == ""
    assert st.export_reference_csv() == ""


def test_export_reference_csv():
    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame({"a": [1], "b": [2]})
    csv = st.export_reference_csv()
    assert "a" in csv and "b" in csv
