"""Import/smoke + end-to-end wiring tests for the analysis UI and workflow."""

from component.widget.analysis_tab import AnalysisPanel, guess_column_mapping


def test_guess_column_mapping_matches_common_headers():
    cols = ["id", "PredictedClass", "ReferenceClass", "location_x", "location_y"]
    m = guess_column_mapping(cols)
    assert m["map"] == "PredictedClass"
    assert m["ref"] == "ReferenceClass"
    assert m["x"] == "location_x"
    assert m["y"] == "location_y"


def test_analysis_panel_is_component():
    assert callable(AnalysisPanel)
