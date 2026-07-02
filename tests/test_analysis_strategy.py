"""Tests for the analysis strategy (base validation + stratified analyze)."""

import numpy as np
import pandas as pd

from component.analysis.stratified_estimation import StratifiedEstimationStrategy
from component.analysis.types import AnalysisInputs, AnalysisMethod


def _inputs(**over):
    rows = [(1, 1)] * 45 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 40
    ref = pd.DataFrame(rows, columns=["map_code", "ref_code"])
    ref["area"] = 1.0
    base = dict(
        method=AnalysisMethod.STRATIFIED_ESTIMATION,
        reference_df=ref,
        area_data=pd.DataFrame({"map_code": [1, 2], "map_area": [60.0, 40.0]}),
        confidence_level=95.0,
        column_mapping={"map": "map_code", "ref": "ref_code"},
        class_names={1: "Forest", 2: "NonForest"},
    )
    base.update(over)
    return AnalysisInputs(**base)


def test_common_validation_flags_empty_reference():
    s = StratifiedEstimationStrategy()
    errs = s.validate_inputs(_inputs(reference_df=pd.DataFrame()))
    assert any("reference" in e.lower() for e in errs)


def test_common_validation_flags_bad_confidence():
    s = StratifiedEstimationStrategy()
    errs = s.validate_inputs(_inputs(confidence_level=97.0))
    assert any("confidence" in e.lower() for e in errs)


def test_is_ready_true_for_valid_inputs():
    assert StratifiedEstimationStrategy().is_ready(_inputs()) is True


def test_analyze_produces_expected_estimates_and_accuracy():
    s = StratifiedEstimationStrategy()
    res = s.analyze(_inputs())
    assert res.success is True
    by_code = {c.map_code: c for c in res.class_estimates}
    assert np.isclose(by_code[1].area_estimate, 62.0)
    assert np.isclose(by_code[2].area_estimate, 38.0)
    assert np.isclose(by_code[1].standard_error, 3.44045, atol=1e-4)
    assert np.isclose(by_code[1].srs_area_estimate, 55.0)
    assert np.isclose(res.overall_accuracy, 0.86)
    acc = {a.map_code: a for a in res.accuracy_rows}
    assert np.isclose(acc[1].users_accuracy, 0.9)
    assert by_code[1].class_name == "Forest"


def test_analyze_errors_when_map_class_missing_from_area():
    s = StratifiedEstimationStrategy()
    res = s.analyze(
        _inputs(area_data=pd.DataFrame({"map_code": [1], "map_area": [60.0]}))
    )
    assert res.success is False
    assert "area" in (res.error_message or "").lower()


def test_analyze_applies_filter():
    s = StratifiedEstimationStrategy()
    rows = (
        [(1, 1, "keep")] * 45
        + [(1, 2, "keep")] * 5
        + [(2, 1, "drop")] * 10
        + [(2, 2, "keep")] * 40
    )
    ref = pd.DataFrame(rows, columns=["map_code", "ref_code", "grp"])
    ref["area"] = 1.0
    res = s.analyze(
        _inputs(
            reference_df=ref, filter_spec={"column": "grp", "include_values": ["keep"]}
        )
    )
    # after filtering out the (2,1) cell, class 2 has only correct samples
    acc = {a.map_code: a for a in res.accuracy_rows}
    assert np.isclose(acc[2].users_accuracy, 1.0)
