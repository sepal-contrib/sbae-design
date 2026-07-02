"""Accuracy-assessment analysis UI: upload -> mapping -> compute -> results."""

import logging

import pandas as pd
import solara
from sepal_ui.sepalwidgets.file_input import FileInputComponent

from component.analysis.service import AnalysisService
from component.model import app_state
from component.widget.analysis_results import AnalysisResultsView  # Task 9/10

logger = logging.getLogger("sbae.analysis.ui")

_ROLE_HINTS = {
    "map": ("map", "predicted", "class_map", "mapclass", "map_code", "pred"),
    "ref": ("ref", "reference", "validated", "truth", "ref_code", "actual"),
    "x": ("x", "long", "lon", "location_x", "xcoord"),
    "y": ("y", "lat", "location_y", "ycoord"),
    "sample_area": ("area", "plot_size", "weight"),
}


def guess_column_mapping(columns: list) -> dict:
    """Best-effort default mapping from CSV header names."""
    mapping = {}
    lowered = {c.lower(): c for c in columns}
    for role, hints in _ROLE_HINTS.items():
        for hint in hints:
            match = next((orig for low, orig in lowered.items() if hint in low), None)
            if match:
                mapping[role] = match
                break
    return mapping


@solara.component
def AnalysisPanel():
    """Full analysis panel filling the Analysis tab."""
    reading = solara.use_reactive(False)
    ref_path = solara.use_reactive(None)

    def read_reference_worker():
        path = ref_path.value
        if not path:
            return None
        return pd.read_csv(path)

    read_result = solara.use_thread(
        read_reference_worker, dependencies=[ref_path.value], intrusive_cancel=False
    )

    def handle_read_result():
        if read_result.state == solara.ResultState.RUNNING:
            reading.value = True
            app_state.analysis_status.value = "Reading reference file..."
        elif read_result.state == solara.ResultState.ERROR:
            reading.value = False
            app_state.analysis_status.value = ""
            app_state.add_error(f"Could not read reference file: {read_result.error}")
        elif read_result.state == solara.ResultState.FINISHED and (
            read_result.value is not None
        ):
            df = read_result.value
            app_state.analysis_reference_df.value = df
            app_state.analysis_column_mapping.value = guess_column_mapping(
                list(df.columns)
            )
            app_state.analysis_status.value = ""
            reading.value = False

    solara.use_effect(handle_read_result, [read_result.state])

    # ---- auto-recalc engine (mirrors sample_configuration) ----
    def recalc():
        if AnalysisService.is_ready(app_state):
            results = AnalysisService.analyze_from_state(app_state)
            if results.success:
                app_state.set_analysis_results(results.to_dict())
                if set(results.map_legend) != set(results.ref_legend):
                    note = (
                        "Note: map and reference legends differ; classes present in "
                        "only one are shown with zero-filled values."
                    )
                    if note not in app_state.error_messages.value:
                        app_state.add_error(note)
            else:
                app_state.set_analysis_results(None)
                app_state.add_error(results.error_message or "Analysis failed")
        else:
            app_state.set_analysis_results(None)

    solara.use_effect(
        recalc,
        [
            app_state.analysis_reference_df.value,
            app_state.analysis_column_mapping.value,
            app_state.area_data.value,
            app_state.analysis_area_source.value,
            app_state.analysis_area_df.value,
            app_state.analysis_confidence_level.value,
            app_state.analysis_filter.value,
        ],
    )

    with solara.Column(style="padding: 8px 4px; gap: 12px;"):
        solara.Markdown(
            "### Accuracy assessment\n"
            "Upload the collected **reference/validation** table (CSV). Areas come "
            "from your classification (design step) or a separate area file."
        )
        with solara.Card("1 · Reference data"):
            FileInputComponent(extensions=[".csv"], on_value=lambda p: ref_path.set(p))
            if app_state.analysis_status.value:
                solara.Info(app_state.analysis_status.value)

        ref_df = app_state.analysis_reference_df.value
        if ref_df is not None and not ref_df.empty:
            _ColumnMappingCard(list(ref_df.columns))
            _AnalysisControls()
            _FilterCard(list(ref_df.columns))

        if app_state.analysis_results.value is not None:
            AnalysisResultsView()


@solara.component
def _ColumnMappingCard(columns: list):
    """Dropdowns mapping CSV columns to analysis roles."""
    mapping = app_state.analysis_column_mapping.value
    options = [None, *list(columns)]

    def make_setter(role):
        def _set(value):
            updated = dict(app_state.analysis_column_mapping.value)
            updated[role] = value
            app_state.analysis_column_mapping.value = updated

        return _set

    labels = {
        "map": "Map / predicted class *",
        "ref": "Reference class *",
        "x": "X / longitude",
        "y": "Y / latitude",
        "sample_area": "Per-sample area (optional)",
    }
    with solara.Card("2 · Column mapping"):
        with solara.Column(gap="4px"):
            for role, label in labels.items():
                solara.Select(
                    label=label,
                    value=mapping.get(role),
                    values=options,
                    on_value=make_setter(role),
                )


@solara.component
def _AnalysisControls():
    """Area source, confidence level, unit, and optional filter controls."""
    with solara.Card("3 · Options"):
        with solara.Column(gap="8px"):
            has_design_area = (
                app_state.area_data.value is not None
                and not app_state.area_data.value.empty
            )
            solara.Select(
                label="Area / strata source",
                value=app_state.analysis_area_source.value,
                values=["design", "upload"],
                on_value=lambda v: app_state.analysis_area_source.set(v),
            )
            if app_state.analysis_area_source.value == "design" and not has_design_area:
                solara.Warning(
                    "No classification area is loaded from the design step. "
                    "Switch to 'upload' to provide an area CSV."
                )
            if app_state.analysis_area_source.value == "upload":
                _AreaUpload()
            solara.Select(
                label="Confidence level (%)",
                value=app_state.analysis_confidence_level.value,
                values=[90.0, 95.0, 99.0],
                on_value=lambda v: app_state.analysis_confidence_level.set(float(v)),
            )
            solara.Select(
                label="Display unit",
                value=app_state.analysis_area_unit.value,
                values=["ha", "m2"],
                on_value=lambda v: app_state.analysis_area_unit.set(v),
            )


@solara.component
def _FilterCard(columns: list):
    """Optional row filter: keep only reference rows whose column is in selected values."""
    current = app_state.analysis_filter.value or {}
    col = current.get("column")

    def set_column(value):
        if not value:
            app_state.analysis_filter.value = None
        else:
            app_state.analysis_filter.value = {"column": value, "include_values": []}

    def set_values(values):
        if col:
            app_state.analysis_filter.value = {"column": col, "include_values": values}

    with solara.Card("4 · Filter (optional)"):
        solara.Select(
            label="Filter column",
            value=col,
            values=[None, *list(columns)],
            on_value=set_column,
        )
        if col and col in app_state.analysis_reference_df.value.columns:
            all_values = sorted(
                app_state.analysis_reference_df.value[col].dropna().unique().tolist(),
                key=str,
            )
            solara.SelectMultiple(
                label="Keep values",
                values=current.get("include_values", []),
                all_values=[str(v) for v in all_values],
                on_value=set_values,
            )


@solara.component
def _AreaUpload():
    """Upload + read a separate area/strata CSV (standalone mode)."""
    area_path = solara.use_reactive(None)

    def read_area_worker():
        return pd.read_csv(area_path.value) if area_path.value else None

    result = solara.use_thread(
        read_area_worker, dependencies=[area_path.value], intrusive_cancel=False
    )

    def handle():
        if result.state == solara.ResultState.FINISHED and result.value is not None:
            app_state.analysis_area_df.value = result.value
        elif result.state == solara.ResultState.ERROR:
            app_state.add_error(f"Could not read area file: {result.error}")

    solara.use_effect(handle, [result.state])
    FileInputComponent(extensions=[".csv"], on_value=lambda p: area_path.set(p))
    # map/value columns are chosen via the same mapping keys area_class/area_value
    cols = (
        list(app_state.analysis_area_df.value.columns)
        if not app_state.analysis_area_df.value.empty
        else []
    )
    if cols:
        mapping = app_state.analysis_column_mapping.value

        def set_area_role(role):
            def _set(v):
                updated = dict(app_state.analysis_column_mapping.value)
                updated[role] = v
                app_state.analysis_column_mapping.value = updated

            return _set

        solara.Select(
            label="Area file: class column",
            value=mapping.get("area_class"),
            values=[None, *cols],
            on_value=set_area_role("area_class"),
        )
        solara.Select(
            label="Area file: area column",
            value=mapping.get("area_value"),
            values=[None, *cols],
            on_value=set_area_role("area_value"),
        )
