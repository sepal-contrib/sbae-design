"""Sample configuration widget using the new sampling architecture.

This module provides the UI for configuring sampling parameters
and delegates calculations to the sampling service.
"""

import logging

import solara
import solara.lab

from component.model import app_state
from component.sampling import SamplingService
from component.tile.class_editor import class_editor_table
from component.widget.aoi_upload_selector import AoiUploadSelector

logger = logging.getLogger("sbae.sample_configuration")

AA_DESIGN_WORKFLOW = "aa_design"
ADVANCED_WORKFLOW = "advanced"

# Short intro shown inline in the Design tab.
AA_DESIGN_INTRO = (
    "**Accuracy assessment design** — stratified sample size and allocation "
    "following Olofsson et al. (2014) good practices."
)

# Detailed methodology, shown in the "What's this?" help popup.
AA_DESIGN_HELP = (
    "Sample size and allocation follow **Olofsson et al. (2014)**, "
    "*Good practices for estimating area and assessing accuracy of land "
    "change*.\n\n"
    "**Inputs:** Expected User's Accuracy (EUA) per class (High / Low "
    "confidence), target standard error, and minimum samples per class.\n\n"
    "**Total sample size:** `n = ( sum(Wi*Si) / SE )^2`, with "
    "`Si = sqrt(EUAi*(1-EUAi))` and `Wi` the area share.\n\n"
    "**Allocation (adjusted proportional):** start from area-proportional "
    "samples, raise any class below the minimum up to it, then redistribute "
    "the rest proportionally. Edit any class's sample count to override."
)


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
            stratified_allocation_method="proportional",
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
def SampleConfiguration(sbae_map=None):
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
                app_state.add_error(results.error_message or "Calculation failed")
        except Exception as e:
            app_state.add_error(f"Error calculating samples: {e!s}")

    active_tab = solara.use_reactive(0)

    with solara.Column():
        with solara.lab.Tabs(value=active_tab):
            solara.lab.Tab("Design")
            solara.lab.Tab("Analysis")

        if active_tab.value == 0:
            DesignTab(sbae_map)
        else:
            AnalysisTab()


@solara.component
def MethodologyHelpButton(
    title="Olofsson AA design — methodology",
    content=AA_DESIGN_HELP,
):
    """Question-mark icon that opens the methodology explanation in a dialog."""
    show, set_show = solara.use_state(False)

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
                solara.Button("Close", text=True, on_click=lambda: set_show(False))


@solara.component
def DesignTab(sbae_map=None):
    """Olofsson accuracy-assessment sample design."""
    with solara.Row(style="align-items: center; gap: 4px;"):
        with solara.Column(style="flex: 1;"):
            solara.Markdown(AA_DESIGN_INTRO)
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

    if not has_valid_data:
        return

    if sampling_method in ("simple", "systematic"):
        SimpleSystematicParameters()
    elif sampling_method == "stratified":
        AccuracyDesignControls()


@solara.component
def AnalysisTab():
    """Accuracy-assessment analysis (area estimation + accuracies)."""
    from component.widget.analysis_tab import AnalysisPanel

    AnalysisPanel()


@solara.component
def SampleDesignWorkflowSelector():
    """Toggle between Olofsson accuracy-assessment design and sampling."""

    def update_workflow(value):
        if value is not None:
            try:
                apply_sample_design_workflow(app_state, value)
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid sample design workflow: {e!s}")

    active_workflow = app_state.sample_design_workflow.value

    solara.Text("Sample Design Workflow", style="font-weight: bold;")
    with solara.Row(gap="4px", style="margin-bottom: 8px;"):
        solara.Button(
            label="Olofsson AA Design",
            on_click=lambda: update_workflow(AA_DESIGN_WORKFLOW),
            icon_name="mdi-bullseye-arrow",
            color="primary" if active_workflow == AA_DESIGN_WORKFLOW else None,
            outlined=active_workflow != AA_DESIGN_WORKFLOW,
            small=True,
            text=True,
        )
        solara.Button(
            label="Sampling",
            on_click=lambda: update_workflow(ADVANCED_WORKFLOW),
            icon_name="mdi-flask-outline",
            color="primary" if active_workflow == ADVANCED_WORKFLOW else None,
            outlined=active_workflow != ADVANCED_WORKFLOW,
            small=True,
            text=True,
        )

    if active_workflow == AA_DESIGN_WORKFLOW:
        solara.Markdown(AA_DESIGN_INTRO)

    if active_workflow == ADVANCED_WORKFLOW:
        SamplingMethodSelector(values=["simple", "systematic"])


@solara.component
def AccuracyDesignControls():
    """Compact controls for the Olofsson accuracy-assessment design."""
    ClassEditorDialogButton(
        button_label="Edit classes & EUA",
        dialog_title="Edit Classes, EUA & Samples",
        show_sample_controls=True,
    )
    StratifiedParameters()


@solara.component
def ClassEditorDialogButton(
    button_label="Edit classes & EUA",
    dialog_title="Edit Classes and Expected User's Accuracy",
    show_sample_controls=False,
):
    """Open the class/EUA editor in a dialog instead of rendering it inline."""
    show_editor_dialog, set_show_editor_dialog = solara.use_state(False)

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
                app_state.add_error(f"Invalid sampling method: {e!s}")

    solara.Select(
        label="Sampling Method",
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
                app_state.add_error(f"Invalid sample total: {e!s}")

    def update_confidence(value):
        if value is not None:
            try:
                app_state.set_sampling_parameters(
                    app_state.target_error.value, float(value)
                )
            except (ValueError, TypeError) as e:
                app_state.add_error(f"Invalid confidence level: {e!s}")

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
                app_state.add_error(f"Invalid expected accuracy: {e!s}")

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label="Total Sample Size",
                v_model=app_state.simple_total_samples.value,
                on_v_model=update_total_samples,
                type="number",
            )
        with solara.Column(style="flex: 1;"):
            solara.Select(
                label="Confidence Level",
                value=app_state.confidence_level.value,
                values=[90.0, 95.0, 99.0],
                on_value=update_confidence,
            )

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.SliderFloat(
                "Expected Overall Accuracy (%)",
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
                app_state.add_error(f"Invalid target error: {e!s}")

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
                app_state.add_error(f"Invalid minimum samples: {e!s}")

    with solara.Row(gap="8px", style="margin-bottom: 8px;"):
        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label="Target Standard Error (%)",
                v_model=app_state.target_error.value,
                on_v_model=update_target_error,
                type="number",
                hint="Standard error of expected overall accuracy",
            )

        with solara.Column(style="flex: 1;"):
            solara.v.TextField(
                label="Minimum Samples per Class",
                v_model=app_state.min_samples_per_class.value,
                on_v_model=update_min_samples,
                type="number",
                hint="Safety minimum for small/rare classes",
            )
