"""Tests for sample design workflow state."""

import ipyvuetify as v
import pandas as pd
import pytest
import solara
from ipyvuetify import VuetifyTemplate

from component.model import app_state
from component.model.state_manager import AppState
from component.tile.class_editor import class_editor_table
from component.widget.sample_configuration import (
    AA_DESIGN_INTRO,
    AccuracyDesignControls,
    SampleDesignWorkflowSelector,
    apply_sample_design_workflow,
)


def test_sample_design_workflow_defaults_to_aa_design():
    state = AppState()

    assert state.sample_design_workflow.value == "aa_design"


def test_sample_design_workflow_accepts_only_known_modes():
    state = AppState()

    state.set_sample_design_workflow("advanced")
    assert state.sample_design_workflow.value == "advanced"

    with pytest.raises(ValueError, match="sample_design_workflow"):
        state.set_sample_design_workflow("experimental")


def test_aa_design_workflow_forces_openforis_parity_sampling():
    state = AppState()
    state.set_sampling_parameters(
        target_error=5.0,
        confidence_level=95.0,
        sampling_method="systematic",
        stratified_allocation_method="balanced",
    )

    apply_sample_design_workflow(state, "aa_design")

    assert state.sample_design_workflow.value == "aa_design"
    assert state.sampling_method.value == "stratified"
    assert state.stratified_allocation_method.value == "proportional"


def test_advanced_workflow_moves_away_from_stratified_default():
    state = AppState()

    apply_sample_design_workflow(state, "advanced")

    assert state.sample_design_workflow.value == "advanced"
    assert state.sampling_method.value == "simple"


def test_aa_design_controls_keep_class_editor_in_dialog():
    app_state.area_data.value = pd.DataFrame(
        {
            "map_code": [1],
            "map_area": [1000],
            "map_edited_class": ["Forest"],
        }
    )
    app_state.expected_user_accuracies.value = {1: 0.9}
    app_state.eua_modes.value = {1: "high"}

    _, rc = solara.render(AccuracyDesignControls(), handle_error=False)

    edit_button = rc.find(v.Btn, block=True)
    edit_button.assert_single()
    assert "Edit classes & EUA" in edit_button.widget.children
    rc.find(v.TextField, label="Target Standard Error (%)").assert_single()
    rc.find(v.Html, children=["Code 1"]).assert_empty()


def test_aa_design_description_mentions_olofsson():
    assert "Olofsson" in AA_DESIGN_INTRO


def test_aa_design_description_visible_before_data():
    app_state.area_data.value = None
    app_state.sample_design_workflow.value = "aa_design"

    _, rc = solara.render(SampleDesignWorkflowSelector(), handle_error=False)

    templates = rc.find(VuetifyTemplate).widgets
    assert any("Olofsson" in (getattr(w, "template", "") or "") for w in templates)


def test_allocation_preview_shows_only_samples():
    app_state.area_data.value = pd.DataFrame(
        {"map_code": [1], "map_area": [1000], "map_edited_class": ["Forest"]}
    )
    app_state.expected_user_accuracies.value = {1: 0.9}
    app_state.eua_modes.value = {1: "high"}
    app_state.sample_results.value = {
        "allocation_dict": {1: 30},
        "samples_per_class": [
            {
                "map_code": 1,
                "equal_samples": 100,
                "proportional_samples": 7,
                "adjusted_samples": 30,
            }
        ],
    }

    _, rc = solara.render(
        class_editor_table(show_sample_controls=True), handle_error=False
    )

    texts = " ".join(
        str(c) for w in rc.find(v.Html).widgets for c in (w.children or [])
    )
    assert "Equal:" not in texts
    assert "Prop.:" not in texts
    assert "Adjusted:" not in texts
    rc.find(v.TextField, label="Samples").assert_single()
