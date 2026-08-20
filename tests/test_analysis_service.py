"""Tests for the analysis service (state -> inputs -> results)."""

import types

import numpy as np
import pandas as pd

from component.analysis.service import AnalysisService


class _R:
    def __init__(self, value):
        self.value = value


def _fake_state():
    rows = [(1, 1)] * 45 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 40
    ref = pd.DataFrame(rows, columns=["mapc", "refc"])
    st = types.SimpleNamespace(
        analysis_reference_df=_R(ref),
        analysis_column_mapping=_R({"map": "mapc", "ref": "refc"}),
        analysis_confidence_level=_R(95.0),
        analysis_filter=_R(None),
        analysis_area_unit=_R("ha"),
        analysis_area_source=_R("design"),
        analysis_area_df=_R(pd.DataFrame()),
        area_data=_R(pd.DataFrame({"map_code": [1, 2], "map_area": [60.0, 40.0]})),
    )
    st.get_class_lookup = lambda: {1: "Forest", 2: "NonForest"}
    return st


def test_is_ready_and_analyze_from_state():
    st = _fake_state()
    assert AnalysisService.is_ready(st) is True
    res = AnalysisService.analyze_from_state(st)
    assert res.success is True
    by_code = {c.map_code: c for c in res.class_estimates}
    assert np.isclose(by_code[1].area_estimate, 62.0)
    assert by_code[1].class_name == "Forest"


def test_not_ready_when_no_reference():
    st = _fake_state()
    st.analysis_reference_df = _R(pd.DataFrame())
    assert AnalysisService.is_ready(st) is False
    assert AnalysisService.get_validation_errors(st)


def test_create_inputs_map_source_uses_area_df():
    from component.model.state_manager import AppState

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"map_code": [1, 2], "ref_code": [1, 1]}
    )
    st.analysis_column_mapping.value = {"map": "map_code", "ref": "ref_code"}
    st.analysis_area_df.value = pd.DataFrame(
        {"map_code": [1, 2], "map_area": [100.0, 50.0]}
    )
    st.analysis_area_source.value = "map"
    inputs = AnalysisService.create_inputs_from_state(st)
    assert list(inputs.area_data["map_area"]) == [100.0, 50.0]
