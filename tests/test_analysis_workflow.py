"""Import/smoke + end-to-end wiring tests for the analysis UI and workflow."""

import numpy as np
import pandas as pd

from component.analysis.service import AnalysisService
from component.model.state_manager import AppState
from component.widget.analysis_tab import (
    AnalysisPanel,
    guess_column_mapping,
    load_example_analysis_data,
)


def test_guess_column_mapping_matches_common_headers():
    cols = ["id", "PredictedClass", "ReferenceClass", "location_x", "location_y"]
    m = guess_column_mapping(cols)
    assert m["map"] == "PredictedClass"
    assert m["ref"] == "ReferenceClass"
    assert m["x"] == "location_x"
    assert m["y"] == "location_y"


def test_analysis_panel_is_component():
    assert callable(AnalysisPanel)


def test_design_area_feeds_analysis_end_to_end():
    st = AppState()
    # design side output (map_code / map_area in m²) + class lookup
    st.area_data.value = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [600000.0, 400000.0],
            "map_edited_class": ["Forest", "NonForest"],
        }
    )
    rows = [(1, 1)] * 45 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 40
    st.analysis_reference_df.value = pd.DataFrame(rows, columns=["mapc", "refc"])
    st.analysis_column_mapping.value = {"map": "mapc", "ref": "refc"}
    st.analysis_area_source.value = "design"
    st.analysis_confidence_level.value = 95.0

    assert AnalysisService.is_ready(st) is True
    res = AnalysisService.analyze_from_state(st)
    assert res.success is True
    total = sum(c.area_estimate for c in res.class_estimates)
    assert np.isclose(total, 1000000.0)  # estimates sum to A_total (native m²)
    names = {c.map_code: c.class_name for c in res.class_estimates}
    assert names[1] == "Forest"  # class lookup from area_data


def test_load_example_analysis_data_runs_analysis():
    st = AppState()
    load_example_analysis_data(st)

    # self-contained: uses the bundled reference + strata via the upload source
    assert st.analysis_area_source.value == "upload"
    assert not st.analysis_reference_df.value.empty
    assert not st.analysis_area_df.value.empty

    assert AnalysisService.is_ready(st) is True
    res = AnalysisService.analyze_from_state(st)
    assert res.success is True
    assert res.confusion_matrix is not None and not res.confusion_matrix.empty
    # 9 classes in the bundled example (2,4,11,12,13,31,32,33,34)
    assert len(res.class_estimates) == 9
    assert 0.0 <= res.overall_accuracy <= 1.0
    # area estimates sum to A_total (native units of the strata file)
    total_area = sum(c.area_estimate for c in res.class_estimates)
    assert np.isclose(total_area, res.total_area)
