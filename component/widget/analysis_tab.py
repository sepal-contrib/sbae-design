"""Accuracy-assessment analysis UI: upload -> mapping -> compute -> results."""

import logging
import os
from pathlib import Path

import pandas as pd
import solara
from sepal_ui.sepalwidgets.file_input import FileInputComponent

from component.analysis.service import AnalysisService
from component.model import app_state
from component.widget.analysis_results import AnalysisResultsView  # Task 9/10
from component.widget.custom_widgets import Section
from component.widget.sample_configuration import MethodologyHelpButton

logger = logging.getLogger("sbae.analysis.ui")

# Short intro shown inline in the Analysis tab (mirrors AA_DESIGN_INTRO).
AA_ANALYSIS_INTRO = (
    "**Accuracy assessment analysis** — error-adjusted area estimates and class "
    "accuracies from your reference sample, following Olofsson et al. (2014)."
)

# Detailed methodology, shown in the "?" help popup.
AA_ANALYSIS_HELP = (
    "Estimates **error-adjusted areas** and accuracies from a collected "
    "**reference / validation sample** using the **Olofsson et al. (2014)** "
    "good-practice estimators.\n\n"
    "**Inputs:** the reference table (mapped class vs. reference class per "
    "sample) and the mapped class areas — taken from your classification "
    "(design step) or a separate area / strata CSV.\n\n"
    "**Outputs:** the error matrix, error-adjusted area per class with "
    "confidence intervals, and user's / producer's / overall accuracies."
)

# Short inline help shown under each section header (kept to 2-3 lines).
_COLUMN_MAPPING_HELP = (
    "Match each CSV column to its role. Every reference point needs its "
    "**mapped class** (`map_code` -- what the classification assigned) and its "
    "**reference class** (`ref_code` -- the interpreted truth); those two build "
    "the error matrix. X/Y are required only when the class is read from an "
    "uploaded map. Fields marked \\* are required."
)
_OPTIONS_HELP = (
    "Where the per-class **areas** (the stratum weights) come from -- the design "
    "step's classification, a separate area/strata CSV, or a classification map "
    "you upload here -- plus the **confidence level** for the error-adjusted "
    "intervals and the **display unit**."
)
_FILTER_HELP = (
    "Optional: restrict the assessment to a subset of reference points -- pick a "
    "column and keep only the values you choose (e.g. one region, tile, or "
    "campaign). Leave empty to use every row."
)

# Friendly labels for the area/strata source. The internal keys stay
# "design" / "map" / "upload"; only the dropdown display changes.
_AREA_SOURCE_LABELS = {
    "design": "From the design map",
    "map": "Upload a classification map",
    "upload": "From an area / strata CSV",
}
_AREA_SOURCE_ORDER = ["design", "map", "upload"]
_AREA_SOURCE_BY_LABEL = {label: key for key, label in _AREA_SOURCE_LABELS.items()}

# Bundled example dataset (collected reference + strata for the aa_test_congo map).
_EXAMPLE_DIR = (
    Path(__file__).parent.parent.parent / "tests" / "data" / "analysis_example"
)
_EXAMPLE_REFERENCE_CSV = _EXAMPLE_DIR / "reference_example.csv"
_EXAMPLE_AREA_CSV = _EXAMPLE_DIR / "area_example.csv"
_EXAMPLE_COLUMN_MAPPING = {
    "map": "map_code",
    "ref": "ref_code",
    "x": "location_x",
    "y": "location_y",
    "area_class": "map_code",
    "area_value": "map_area",
}


def load_example_analysis_data(state) -> None:
    """Load the bundled example reference + strata into the analysis state.

    Self-contained: uses the "upload" area source so it never overwrites the
    design tab's ``area_data``. Loads inputs only -- the dashboard stays blank
    until the user presses Calculate (results are only computed on that action).
    """
    if not _EXAMPLE_REFERENCE_CSV.exists() or not _EXAMPLE_AREA_CSV.exists():
        state.add_error(f"Example analysis data not found in {_EXAMPLE_DIR}")
        return
    try:
        ref_df = pd.read_csv(_EXAMPLE_REFERENCE_CSV)
        area_df = pd.read_csv(_EXAMPLE_AREA_CSV)
    except Exception as e:  # pragma: no cover - defensive
        state.add_error(f"Could not load example analysis data: {e!s}")
        return
    state.analysis_reference_df.value = ref_df
    state.analysis_reference_name.value = _EXAMPLE_REFERENCE_CSV.name
    state.analysis_area_df.value = area_df
    state.analysis_area_name.value = _EXAMPLE_AREA_CSV.name
    state.analysis_area_source.value = "upload"
    state.analysis_column_mapping.value = dict(_EXAMPLE_COLUMN_MAPPING)
    state.analysis_confidence_level.value = 95.0
    state.analysis_status.value = ""
    state.set_analysis_results(None)  # explicit Calculate reveals the result


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


def derive_map_source(state, sbae_map=None):
    """Sample the classification raster for the "map" area source.

    Fills ``map_code`` on the reference table by sampling the raster at each
    point, computes the per-class area table, derives class colors when the
    design step didn't (standalone mode), and renders the raster on
    ``sbae_map``. Mutates ``analysis_reference_df`` / ``analysis_area_df`` /
    ``analysis_column_mapping`` in place. Errors are surfaced via
    ``state.add_error`` and never raised. Returns the dropped-point count, or
    ``None`` when there is nothing to derive.

    Called from the Calculate action (``run_calculation``), off the UI thread,
    so it may block on raster I/O and call the map methods directly.
    """
    raster = state.analysis_classification_path.value
    ref = state.analysis_reference_df.value
    if not raster or ref is None or ref.empty:
        return None
    from component.scripts.accuracy import derive_from_classification

    try:
        ref_out, area_df, dropped = derive_from_classification(
            ref, state.analysis_column_mapping.value or {}, raster
        )
    except Exception as e:  # surface, don't crash the Calculate task
        state.add_error(f"Classification-map analysis failed: {e}")
        return None

    mapping = dict(state.analysis_column_mapping.value or {})
    mapping["map"] = "map_code"
    state.analysis_column_mapping.value = mapping
    state.analysis_area_df.value = area_df
    state.analysis_reference_df.value = ref_out

    # Standalone mode never runs the design-step upload that populates
    # class_colors, so it's empty here -- without this, add_class_raster falls
    # back to a continuous colormap and the map renders near-black. Derive it
    # from the raster; guarded so a real design-step palette is kept.
    if not state.class_colors.value:
        from component.scripts.geospatial import get_color_palette

        state.class_colors.value = get_color_palette(
            raster, sorted(int(c) for c in area_df["map_code"].tolist())
        )

    if sbae_map is not None:
        try:
            sbae_map.add_class_raster(
                raster,
                state.class_colors.value or {},
                "Classification (analysis)",
                "clas_an",
            )
            # Reference points are drawn by the panel's render thread (from
            # analysis_reference_df, for every source) on their own layer.
        except Exception as e:  # results are valid; only the map layer failed
            state.add_error(
                "Analysis ran, but the classification map could not be "
                f"rendered on the map: {e}"
            )
    return dropped


def run_calculation(state, sbae_map=None):
    """Run the explicit Calculate action for the analysis tab.

    Derives the map source (if selected), runs the accuracy assessment, and
    stores the results with their inputs signature.
    Never raises; user-correctable problems are surfaced via
    ``state.add_error`` and leave the dashboard blank. Runs off the UI thread
    (the map source samples a raster), so callers wrap it in ``to_thread``.
    """
    if state.analysis_area_source.value == "map":
        derive_map_source(state, sbae_map)

    if not AnalysisService.is_ready(state):
        state.set_analysis_results(None)
        errors = AnalysisService.get_validation_errors(state)
        state.add_error(
            "; ".join(errors)
            if errors
            else "Provide the required inputs before calculating."
        )
        return

    results = AnalysisService.analyze_from_state(state)
    if results.success:
        state.set_analysis_results(
            results.to_dict(), signature=AnalysisService.inputs_signature(state)
        )
        if set(results.map_legend) != set(results.ref_legend):
            note = (
                "Note: map and reference legends differ; classes present in "
                "only one are shown with zero-filled values."
            )
            if note not in state.error_messages.value:
                state.add_error(note)
    else:
        state.set_analysis_results(None)
        state.add_error(results.error_message or "Analysis failed")


def _results_are_fresh(state) -> bool:
    """True when the stored results still match the current inputs.

    Calculate stamps the results with a signature of the inputs that produced
    them; any later edit changes the live signature, so the dashboard is hidden
    until the user recalculates.
    """
    return (
        state.analysis_results.value is not None
        and state.analysis_results_signature.value
        == AnalysisService.inputs_signature(state)
    )


@solara.component
def ExampleDataButton():
    """One-click loader for the bundled example analysis dataset."""
    solara.Button(
        "Use example data",
        on_click=lambda: load_example_analysis_data(app_state),
        color="default",
        text=True,
        small=True,
    )


@solara.component
def CurrentTableDisplay(title: str, df, name: str = "", on_clear=None):
    """Compact card showing a loaded analysis table.

    Mirrors ``CurrentFileDisplay`` from the design tab: once a CSV is loaded we
    show its name and shape with a clear button instead of the file picker.
    """
    if df is None or df.empty:
        return

    n_rows, n_cols = df.shape
    row_word = "row" if n_rows == 1 else "rows"
    col_word = "column" if n_cols == 1 else "columns"

    with solara.Card(classes=["mb-2"]):
        with solara.Row(justify="space-between", style={"align-items": "center"}):
            with solara.Column(gap="0px"):
                with solara.Row(gap="6px", style="align-items: baseline;"):
                    solara.Text(f"{title}:", style="font-weight: 600; font-size: 14px;")
                    solara.Text(name or "loaded", style="font-size: 14px;")
                solara.Text(
                    f"{n_rows:,} {row_word}, {n_cols} {col_word}",
                    style="font-size: 12px;",
                )
            if on_clear is not None:
                solara.Button(
                    label="",
                    icon_name="mdi-close",
                    on_click=on_clear,
                    color="error",
                    text=True,
                    icon=True,
                    small=True,
                )


@solara.component
def AnalysisPanel(sbae_map=None, theme_toggle=None):
    """Full analysis panel filling the Analysis tab."""
    reading = solara.use_reactive(False)
    ref_path = solara.use_reactive(None)
    show_ref_modal = solara.use_reactive(False)

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
            if ref_path.value:
                app_state.analysis_reference_name.value = Path(ref_path.value).name
            app_state.analysis_column_mapping.value = guess_column_mapping(
                list(df.columns)
            )
            app_state.analysis_status.value = ""
            reading.value = False

    solara.use_effect(handle_read_result, [read_result.state])

    def clear_reference():
        """Clear the loaded reference table (mirrors the design 'clear file')."""
        ref_path.set(None)
        app_state.analysis_reference_df.value = pd.DataFrame()
        app_state.analysis_reference_name.value = ""
        app_state.analysis_column_mapping.value = {}
        app_state.set_analysis_results(None)

    # ---- explicit Calculate action ----
    # No auto-recalc: results are computed only when the user presses Calculate,
    # and the dashboard is hidden again as soon as any input changes (see
    # _results_are_fresh). A trigger counter re-runs the worker on each click;
    # use_thread (not use_task) keeps it off the UI thread without needing a
    # running asyncio loop -- matching the panel's other workers.
    calc_trigger = solara.use_reactive(0)

    def _calc_worker():
        if calc_trigger.value == 0:
            return None  # don't compute on mount
        run_calculation(app_state, sbae_map)
        return calc_trigger.value

    calc_result = solara.use_thread(
        _calc_worker, dependencies=[calc_trigger.value], intrusive_cancel=False
    )

    # Render the analysis reference points on their own map layer (distinct color
    # + name from the design sample), for every source, whenever the reference
    # table and its x/y mapping are ready. Build + attach run off the UI thread
    # (tippecanoe), mirroring the derivation's raster render.
    def render_reference_points_worker():
        if sbae_map is None:
            return None
        df = app_state.analysis_reference_df.value
        mapping = app_state.analysis_column_mapping.value or {}
        x, y = mapping.get("x"), mapping.get("y")
        if (
            df is None
            or df.empty
            or not x
            or not y
            or x not in df.columns
            or y not in df.columns
        ):
            return None
        points = pd.DataFrame(
            {
                "longitude": df[x],
                "latitude": df[y],
                "map_code": df["map_code"] if "map_code" in df.columns else 0,
            }
        )
        sbae_map.add_reference_points(points)
        return len(points)

    _mapping_now = app_state.analysis_column_mapping.value or {}
    solara.use_thread(
        render_reference_points_worker,
        dependencies=[
            app_state.analysis_reference_df.value,
            _mapping_now.get("x"),
            _mapping_now.get("y"),
        ],
        intrusive_cancel=False,
    )

    ref_df = app_state.analysis_reference_df.value
    ref_loaded = ref_df is not None and not ref_df.empty

    def can_calculate() -> bool:
        """Enough inputs present to attempt a calculation (per-source file)."""
        if not ref_loaded:
            return False
        source = app_state.analysis_area_source.value
        if source == "design":
            area = app_state.area_data.value
            return area is not None and not area.empty
        if source == "upload":
            area = app_state.analysis_area_df.value
            return area is not None and not area.empty
        if source == "map":
            return bool(app_state.analysis_classification_path.value)
        return False

    def close_ref_modal_when_loaded():
        # Once a reference table is loaded (upload or example), close the modal
        # so the "current table" card + downstream controls take over.
        if ref_loaded:
            show_ref_modal.value = False

    solara.use_effect(close_ref_modal_when_loaded, [ref_loaded])

    with solara.Column(style="gap: 12px;"):
        with solara.Row(style="align-items: center; gap: 4px;"):
            with solara.Column(style="flex: 1;"):
                solara.Markdown(AA_ANALYSIS_INTRO)
            MethodologyHelpButton(
                title="Accuracy assessment — methodology",
                content=AA_ANALYSIS_HELP,
            )

        # Reference upload sits directly under the intro (no subtitle).
        if ref_loaded:
            CurrentTableDisplay(
                "Reference table",
                ref_df,
                name=app_state.analysis_reference_name.value,
                on_clear=clear_reference,
            )
        else:
            solara.Button(
                "Upload reference data",
                on_click=lambda: show_ref_modal.set(True),
                color="primary",
                block=True,
                small=True,
            )

        if ref_loaded:
            _ColumnMappingCard(
                list(ref_df.columns), app_state.analysis_area_source.value
            )
            _AnalysisControls()
            _FilterCard(list(ref_df.columns))

            calc_running = calc_result.state in (
                solara.ResultState.STARTING,
                solara.ResultState.RUNNING,
            )
            solara.Button(
                "Calculate accuracy assessment",
                on_click=lambda: calc_trigger.set(calc_trigger.value + 1),
                color="primary",
                block=True,
                small=True,
                loading=calc_running,
                disabled=calc_running or not can_calculate(),
            )

        # Results section, always present like the design's Summary: the worked
        # output once fresh, otherwise a hint on what to do next. Results are
        # stamped with an inputs signature at Calculate time; once any input
        # changes the dashboard is hidden until the user recalculates.
        if _results_are_fresh(app_state):
            AnalysisResultsView(theme_toggle=theme_toggle)
        elif not ref_loaded:
            solara.Info(
                "Upload a reference table (or load the example data) to run the "
                "accuracy assessment."
            )
        else:
            solara.Info(
                "Set the area source and column mapping, then press "
                "**Calculate** to run the accuracy assessment."
            )

    # Rendered outside the Column so state changes don't unmount it mid-flow.
    if show_ref_modal.value:
        with solara.v.Dialog(
            v_model=show_ref_modal.value,
            on_v_model=show_ref_modal.set,
            max_width=900,
            eager=True,
        ):
            _ReferenceUploadDialog(ref_path, on_close=lambda: show_ref_modal.set(False))


@solara.component
def _ReferenceUploadDialog(ref_path, on_close=None):
    """Modal card for uploading the reference CSV (mirrors the design upload modal).

    Same styling as ``UploadDialogCard``: a titled card with the file picker and
    example-data shortcut inside, and a right-aligned Close button.
    """
    with solara.v.Card():
        solara.v.CardTitle(children=["Upload reference data"])
        with solara.v.CardText(style="max-height: 70vh; overflow-y: auto;"):
            solara.Markdown(
                "Upload the collected **reference / validation** table (CSV)."
            )
            FileInputComponent(extensions=[".csv"], on_value=lambda p: ref_path.set(p))
            if app_state.analysis_status.value:
                solara.Info(app_state.analysis_status.value)
            with solara.Row(justify="center", classes=["mt-2"]):
                solara.Text("or")
                ExampleDataButton()
        with solara.v.CardActions():
            solara.v.Spacer()
            solara.Button("Close", text=True, small=True, on_click=on_close)


@solara.component
def _ColumnMappingCard(columns: list, area_source: str = "upload"):
    """Dropdowns mapping CSV columns to analysis roles."""
    mapping = app_state.analysis_column_mapping.value
    options = [None, *list(columns)]

    def make_setter(role):
        def _set(value):
            updated = dict(app_state.analysis_column_mapping.value)
            updated[role] = value
            app_state.analysis_column_mapping.value = updated

        return _set

    map_source = area_source == "map"
    labels = {
        "map": "Map / predicted class *",
        "ref": "Reference class *",
        "x": "X / longitude *" if map_source else "X / longitude",
        "y": "Y / latitude *" if map_source else "Y / latitude",
        "sample_area": "Per-sample area (optional)",
    }
    if map_source:
        del labels["map"]  # map_code is derived from the raster
    with solara.Column(gap="4px"):
        Section("Column mapping", "mdi-swap-horizontal")
        solara.Markdown(_COLUMN_MAPPING_HELP)
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
    with solara.Column(gap="8px"):
        Section("Options", "mdi-tune")
        solara.Markdown(_OPTIONS_HELP)
        source = app_state.analysis_area_source.value
        has_design_area = (
            app_state.area_data.value is not None
            and not app_state.area_data.value.empty
        )
        solara.Select(
            label="Area / strata source",
            value=_AREA_SOURCE_LABELS[source],
            values=[_AREA_SOURCE_LABELS[k] for k in _AREA_SOURCE_ORDER],
            on_value=lambda label: app_state.analysis_area_source.set(
                _AREA_SOURCE_BY_LABEL[label]
            ),
        )
        if source == "design":
            if has_design_area:
                DesignClassificationCard()
            else:
                solara.Warning(
                    "No classification is loaded from the design step. Switch to "
                    "'Upload a classification map' or 'From an area / strata CSV'."
                )
        elif source == "upload":
            _AreaUpload()
        elif source == "map":
            _ClassificationMapUpload()
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

    with solara.Column(gap="4px"):
        Section("Filter (optional)", "mdi-filter-variant")
        solara.Markdown(_FILTER_HELP)
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
            if area_path.value:
                app_state.analysis_area_name.value = Path(area_path.value).name
        elif result.state == solara.ResultState.ERROR:
            app_state.add_error(f"Could not read area file: {result.error}")

    solara.use_effect(handle, [result.state])

    def clear_area():
        """Clear the loaded area/strata table."""
        area_path.set(None)
        app_state.analysis_area_df.value = pd.DataFrame()
        app_state.analysis_area_name.value = ""

    area_df = app_state.analysis_area_df.value
    if area_df is not None and not area_df.empty:
        CurrentTableDisplay(
            "Area / strata table",
            area_df,
            name=app_state.analysis_area_name.value,
            on_clear=clear_area,
        )
    else:
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


@solara.component
def DesignClassificationCard():
    """Read-only card naming the design-step classification reused as the source.

    Mirrors the design tab's ``CurrentFileDisplay`` minus the clear button --
    the design step owns that file.
    """
    file_path = app_state.file_path.value
    area = app_state.area_data.value
    name = Path(file_path).name if file_path else "design classification"
    n_classes = 0 if area is None or area.empty else int(area["map_code"].nunique())
    class_word = "class" if n_classes == 1 else "classes"
    with solara.Card(classes=["mb-2"]):
        with solara.Column(gap="0px"):
            with solara.Row(gap="6px", style="align-items: baseline;"):
                solara.Text("Design map:", style="font-weight: 600; font-size: 14px;")
                solara.Text(name, style="font-size: 14px;")
            solara.Text(
                f"{n_classes} {class_word} from the design step",
                style="font-size: 12px;",
            )


@solara.component
def CurrentRasterDisplay(path: str, on_clear=None):
    """Design-style card for the uploaded classification raster (map source).

    Mirrors ``CurrentFileDisplay``: name + type/size and a clear button, instead
    of a bare filename row.
    """
    if not path:
        return
    name = Path(path).name
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    with solara.Card(classes=["mb-2"]):
        with solara.Row(justify="space-between", style={"align-items": "center"}):
            with solara.Column(gap="0px"):
                with solara.Row(gap="6px", style="align-items: baseline;"):
                    solara.Text(
                        "Classification map:",
                        style="font-weight: 600; font-size: 14px;",
                    )
                    solara.Text(name, style="font-size: 14px;")
                solara.Text(f"GeoTIFF | {size_mb:.1f} MB", style="font-size: 12px;")
            if on_clear is not None:
                solara.Button(
                    label="",
                    icon_name="mdi-close",
                    on_click=on_clear,
                    color="error",
                    text=True,
                    icon=True,
                    small=True,
                )


@solara.component
def _ClassificationMapUpload():
    """Pick a classification GeoTIFF for the "map" area source.

    Only selects the file. The raster is sampled (map_code + per-class areas)
    and rendered when the user presses Calculate -- see ``derive_map_source`` /
    ``run_calculation``. Clearing the path blanks the dashboard via the inputs
    signature (the path is part of it), so no explicit reset is needed here.
    """
    path = app_state.analysis_classification_path
    if path.value:
        CurrentRasterDisplay(path.value, on_clear=lambda: path.set(None))
    else:
        FileInputComponent(extensions=[".tif", ".tiff"], on_value=lambda p: path.set(p))
