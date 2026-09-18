"""Sample configuration widget using the new sampling architecture.

This module provides the UI for configuring sampling parameters
and delegates calculations to the sampling service.
"""

import logging

import solara
import solara.lab

from component.message import msg
from component.model import app_state
from component.sampling import SamplingService
from component.tile.class_editor import class_editor_table
from component.tile.export import Export
from component.widget.aoi_upload_selector import AoiUploadSelector
from component.widget.custom_widgets import Section
from component.widget.point_generation import (
    PointGeneration,
    PointGenerationView,
    use_point_generation_task,
)
from component.widget.summary import Summary

logger = logging.getLogger("sbae.sample_configuration")

AA_DESIGN_WORKFLOW = "aa_design"
ADVANCED_WORKFLOW = "advanced"


def apply_sample_design_workflow(state, workflow: str):
    """Apply workflow-specific sampling defaults."""
    state.set_sample_design_workflow(workflow)

    if workflow == AA_DESIGN_WORKFLOW:
        state.set_sampling_parameters(
            state.target_error.value,
            state.confidence_level.value,
            state.min_samples_per_class.value,
            state.expected_accuracy.value,
            sampling_method="stratified",
            simple_total_samples=state.simple_total_samples.value,
            # aa_design is the Olofsson per-class-EUA design -> neyman path
            # (per-class EUA is only active for neyman; AGENTS.md).
            stratified_allocation_method="neyman",
        )
    elif workflow == ADVANCED_WORKFLOW and state.sampling_method.value == "stratified":
        state.set_sampling_parameters(
            state.target_error.value,
            state.confidence_level.value,
            state.min_samples_per_class.value,
            state.expected_accuracy.value,
            sampling_method="simple",
            simple_total_samples=state.simple_total_samples.value,
        )
    else:
        state.set_sample_design_workflow(workflow)


@solara.component
def SampleConfiguration(sbae_map=None, theme_state=None):
    """Sample configuration widget for the right panel."""
    # Use use_ref to persist value across renders without re-initializing
    prev_method_ref = solara.use_ref(app_state.sampling_method.value)
    current_method = app_state.sampling_method.value

    # Check if method changed - use_ref.current persists between renders
    if prev_method_ref.current != current_method:
        logger.info(
            f">>> METHOD CHANGED from {prev_method_ref.current} to {current_method}, CLEARING ALL DATA"
        )

        # Clean up map layers
        if sbae_map:
            sbae_map.remove_layer("clas", none_ok=True)
            if sbae_map.sample_points_layer:
                try:
                    sbae_map.remove_layer(sbae_map.sample_points_layer)
                except Exception:
                    pass
                sbae_map.sample_points_layer = None

        # Clear all sampling data (both file and AOI)
        app_state.clear_all_sampling_data()

        # Update tracked method AFTER clearing
        prev_method_ref.current = current_method

    def auto_calculate():
        """Auto-calculate when ready and parameters change."""
        if SamplingService.is_ready(app_state):
            run_calculation()

    solara.use_effect(
        auto_calculate,
        [
            app_state.target_error.value,
            app_state.confidence_level.value,
            app_state.min_samples_per_class.value,
            app_state.expected_accuracy.value,
            app_state.sampling_method.value,
            app_state.stratified_allocation_method.value,
            app_state.simple_total_samples.value,
            app_state.area_data.value,
            app_state.aoi_gdf.value,
            app_state.expected_user_accuracies.value,
            app_state.high_eua.value,
            app_state.low_eua.value,
            app_state.eua_modes.value,
            app_state.sample_design_workflow.value,
        ],
    )

    def run_calculation():
        """Execute the sampling calculation."""
        try:
            results = SamplingService.calculate_from_state(app_state)
            if results.success:
                app_state.set_sample_results(results.to_dict())
            else:
                app_state.add_error(
                    results.error_message or msg("design.error.calculation_failed")
                )
        except Exception as e:
            app_state.add_error(msg("design.error.calculating", error=e))

    active_tab = solara.use_reactive(0)
    point_generation_controller = use_point_generation_task(sbae_map)

    with solara.Column():
        with solara.lab.Tabs(value=active_tab):
            solara.lab.Tab(msg("design.tab"))
            solara.lab.Tab(msg("analysis.tab"))

        if active_tab.value == 0:
            DesignTab(
                sbae_map,
                theme_state=theme_state,
                point_generation_controller=point_generation_controller,
            )
        else:
            AnalysisTab(sbae_map=sbae_map, theme_state=theme_state)


@solara.component
def MethodologyHelpButton(title=None, content=None):
    """Question-mark icon that opens the methodology explanation in a dialog.

    Defaults to the design-step methodology; the analysis tab passes its own.
    """
    show, set_show = solara.use_state(False)
    title = title if title is not None else msg("design.help.title")
    content = content if content is not None else msg("design.help.body")

    solara.Button(
        icon_name="mdi-help-circle-outline",
        on_click=lambda: set_show(True),
        icon=True,
        small=True,
    )

    with solara.v.Dialog(
        v_model=show,
        on_v_model=set_show,
        max_width=600,
        eager=True,
    ):
        with solara.v.Card():
            solara.v.CardTitle(children=[title])
            with solara.v.CardText(style="max-height: 70vh; overflow-y: auto;"):
                solara.Markdown(content)
            with solara.v.CardActions():
                solara.v.Spacer()
                solara.Button(
                    msg("common.close"), text=True, on_click=lambda: set_show(False)
                )


@solara.component
def DesignTab(sbae_map=None, theme_state=None, point_generation_controller=None):
    """Olofsson accuracy-assessment sample design."""
    with solara.Row(style="align-items: center; gap: 4px;"):
        with solara.Column(style="flex: 1;"):
            solara.Markdown(msg("design.intro"))
        MethodologyHelpButton()

    AoiUploadSelector(sbae_map)

    # Check if data source is available for current method
    sampling_method = app_state.sampling_method.value
    has_valid_data = False
    if sampling_method in ("simple", "systematic"):
        has_valid_data = app_state.aoi_gdf.value is not None
    elif sampling_method == "stratified":
        has_valid_data = (
            app_state.area_data.value is not None
            and not app_state.area_data.value.empty
        )

    if has_valid_data:
        if sampling_method in ("simple", "systematic"):
            SimpleSystematicParameters()
        elif sampling_method == "stratified":
            AccuracyDesignControls()

    # Design outputs (summary / generate points / export) live inside the
    # Design tab so they no longer leak onto the Analysis tab.
    DesignOutputs(
        sbae_map,
        theme_state=theme_state,
        point_generation_controller=point_generation_controller,
    )


@solara.component
def DesignOutputs(sbae_map=None, theme_state=None, point_generation_controller=None):
    """Design-phase outputs, relocated from standalone right-panel sections.

    Renders the sample-design summary, point generation and export blocks using
    the shared theme-aware ``Section`` header (matching pysepal's right-panel
    section look). Kept standalone so it scopes cleanly to the Design tab (and
    renders without a map).
    """
    Section(msg("design.outputs.summary"), "mdi-progress-check")
    Summary(theme_state=theme_state)

    Section(
        msg("design.outputs.generate_points"),
        "mdi-map-marker-multiple",
        msg("design.outputs.generate_points_description"),
    )
    if point_generation_controller is None:
        PointGeneration(sbae_map)
    else:
        PointGenerationView(sbae_map, point_generation_controller)

    Section(
        msg("design.outputs.export"),
        "mdi-download",
        msg("design.outputs.export_description"),
    )
    Export()


@solara.component
def AnalysisTab(sbae_map=None, theme_state=None):
    """Accuracy-assessment analysis (area estimation + accuracies)."""
    from component.widget.analysis_tab import AnalysisPanel

    AnalysisPanel(sbae_map=sbae_map, theme_state=theme_state)


@solara.component
def SampleDesignWorkflowSelector():
    """Toggle between Olofsson accuracy-assessment design and sampling."""

    def update_workflow(value):
        if value is not None:
            try:
                apply_sample_design_workflow(app_state, value)
            except (ValueError, TypeError) as e:
                app_state.add_error(msg("design.workflow.invalid", error=e))

    active_workflow = app_state.sample_design_workflow.value

    solara.Text(msg("design.workflow.title"), style="font-weight: bold;")
    with solara.Row(gap="4px", style="margin-bottom: 8px;"):
        solara.Button(
            label=msg("design.workflow.aa_design"),
            on_click=lambda: update_workflow(AA_DESIGN_WORKFLOW),
            icon_name="mdi-bullseye-arrow",
            color="primary" if active_workflow == AA_DESIGN_WORKFLOW else None,
            outlined=active_workflow != AA_DESIGN_WORKFLOW,
            small=True,
            text=True,
        )
        solara.Button(
            label=msg("design.workflow.advanced"),
            on_click=lambda: update_workflow(ADVANCED_WORKFLOW),
            icon_name="mdi-flask-outline",
            color="primary" if active_workflow == ADVANCED_WORKFLOW else None,
            outlined=active_workflow != ADVANCED_WORKFLOW,
            small=True,
            text=True,
        )

    if active_workflow == AA_DESIGN_WORKFLOW:
        solara.Markdown(msg("design.intro"))

    if active_workflow == ADVANCED_WORKFLOW:
        SamplingMethodSelector(values=["simple", "systematic"])


@solara.component
def AccuracyDesignControls():
    """Compact controls for the Olofsson accuracy-assessment design."""
    ClassEditorDialogButton(
        dialog_title=msg("design.class_editor.dialog_title_with_samples"),
        show_sample_controls=True,
    )
    StratifiedParameters()


@solara.component
def ClassEditorDialogButton(
    button_label=None,
    dialog_title=None,
    show_sample_controls=False,
):
    """Open the class/EUA editor in a dialog instead of rendering it inline."""
    show_editor_dialog, set_show_editor_dialog = solara.use_state(False)
    button_label = button_label or msg("design.class_editor.button")
    dialog_title = dialog_title or msg("design.class_editor.dialog_title")

    solara.Button(
        button_label,
        icon_name="mdi-pencil",
        on_click=lambda: set_show_editor_dialog(True),
        color="primary",
        block=True,
        style="margin-bottom: 12px;",
        small=True,
    )

    with solara.v.Dialog(
        v_model=show_editor_dialog,
        on_v_model=set_show_editor_dialog,
        max_width=900,
        eager=True,
    ):
        with solara.v.Card():
            solara.v.CardTitle(children=[dialog_title])
            with solara.v.CardText(style="max-height: 70vh; overflow-y: auto;"):
                if show_editor_dialog:
                    class_editor_table(show_sample_controls=show_sample_controls)


@solara.component
def SamplingMethodSelector(values=None):
    """Dropdown for selecting sampling method."""
    if values is None:
        values = ["stratified", "simple", "systematic"]

    def update_method(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value,
                    app_state.confidence_level.value,
                    app_state.min_samples_per_class.value,
                    app_state.expected_accuracy.value,
                    value,
                    app_state.simple_total_samples.value,
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(msg("design.error.invalid_method", error=e))

    solara.Select(
        label=msg("design.parameters.sampling_method"),
        value=app_state.sampling_method.value,
        values=values,
        on_value=update_method,
    )


@solara.component
def SimpleSystematicParameters():
    """Parameters for simple and systematic sampling."""

    def update_total_samples(value):
        if value is not None and value != "":
            try:
                int_value = int(float(value))
                if int_value > 0:
                    app_state.set_sampling_parameters(
                        app_state.target_error.value,
                        app_state.confidence_level.value,
                        app_state.min_samples_per_class.value,
                        app_state.expected_accuracy.value,
                        app_state.sampling_method.value,
                        int_value,
                    )
            except (ValueError, TypeError) as e:
                app_state.add_error(msg("design.error.invalid_total", error=e))

    def update_confidence(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value, float(value)
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(msg("design.error.invalid_confidence", error=e))

    def update_expected_accuracy(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value,
                    app_state.confidence_level.value,
                    app_state.min_samples_per_class.value,
                    float(value),
                    app_state.sampling_method.value,
                    app_state.simple_total_samples.value,
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(
                    msg("design.error.invalid_expected_accuracy", error=e)
                )

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label=msg("design.parameters.total_sample_size"),
                v_model=app_state.simple_total_samples.value,
                on_v_model=update_total_samples,
                type="number",
            )
        with solara.Column(style="flex: 1;"):
            solara.Select(
                label=msg("design.parameters.confidence_level"),
                value=app_state.confidence_level.value,
                values=[90.0, 95.0, 99.0],
                on_value=update_confidence,
            )

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.SliderFloat(
                msg("design.parameters.expected_accuracy"),
                value=app_state.expected_accuracy.value,
                min=50.0,
                max=99.0,
                step=1.0,
                on_value=update_expected_accuracy,
            )


@solara.component
def StratifiedParameters():
    """Parameters for stratified sampling."""

    def update_target_error(value):
        if value is not None and value != "":
            try:
                float_value = float(value)
                if float_value > 0:
                    app_state.set_sampling_parameters(
                        float_value, app_state.confidence_level.value
                    )
            except (ValueError, TypeError) as e:
                app_state.add_error(msg("design.error.invalid_target_error", error=e))

    def update_min_samples(value):
        if value is not None and value != "":
            try:
                int_value = int(float(value))
                if int_value > 0:
                    app_state.set_sampling_parameters(
                        app_state.target_error.value,
                        app_state.confidence_level.value,
                        int_value,
                    )
            except (ValueError, TypeError) as e:
                app_state.add_error(msg("design.error.invalid_min_samples", error=e))

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label=msg("design.parameters.target_error"),
                v_model=app_state.target_error.value,
                on_v_model=update_target_error,
                type="number",
                hint=msg("design.parameters.target_error_hint"),
            )

        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label=msg("design.parameters.min_samples"),
                v_model=app_state.min_samples_per_class.value,
                on_v_model=update_min_samples,
                type="number",
                hint=msg("design.parameters.min_samples_hint"),
            )
