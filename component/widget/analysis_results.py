"""Rendered analysis outputs: matrices, tables, overall accuracy, export."""

import solara

from component.message import msg
from component.model import app_state
from component.scripts.accuracy import convert_area
from component.widget.custom_widgets import DownloadMenu, Section


def _unit_label(unit: str) -> str:
    return "ha" if unit == "ha" else "m²"


@solara.component
def AnalysisResultsView(theme_state=None):
    results = app_state.analysis_results.value
    if not results:
        return
    from component.widget.analysis_dashboard import AnalysisSummaryCard

    AnalysisSummaryCard(theme_state=theme_state)


@solara.component
def _ConfusionMatrix(results):
    cm = results.get("confusion_matrix")
    if not cm:
        return
    with solara.Column(gap="4px"):
        Section(
            msg("analysis.tables.confusion_title"),
            "mdi-grid",
            msg("analysis.tables.confusion_description"),
        )
        with solara.GridFixed(columns=len(cm["columns"]) + 1):
            solara.Text(
                msg("analysis.tables.confusion_corner"), style="font-weight: bold;"
            )
            for c in cm["columns"]:
                solara.Text(str(c), style="font-weight: bold;")
            for code, row in zip(cm["index"], cm["data"]):
                solara.Text(str(code), style="font-weight: bold;")
                for v in row:
                    solara.Text(f"{v:g}")


@solara.component
def _AreaEstimates(results, unit):
    rows = results.get("class_estimates", [])
    if not rows:
        return
    u = _unit_label(unit)
    headers = [
        msg("analysis.tables.class_header"),
        msg("analysis.tables.samples_header"),
        msg("analysis.tables.map_area_header", unit=u),
        msg("analysis.tables.adjusted_area_header", unit=u),
        msg("analysis.tables.confidence_interval_header", unit=u),
        msg("analysis.tables.srs_area_header", unit=u),
    ]
    with solara.Column(gap="4px"):
        Section(
            msg("analysis.tables.area_title"),
            "mdi-chart-box-outline",
            msg("analysis.tables.area_description"),
        )
        with solara.GridFixed(columns=len(headers)):
            for h in headers:
                solara.Text(h, style="font-weight: bold;")
            for r in rows:
                solara.Text(str(r["class_name"]))
                solara.Text(f"{r['number_samples']:g}")
                solara.Text(f"{convert_area(r['map_pixel_count'], unit):,.2f}")
                solara.Text(f"{convert_area(r['area_estimate'], unit):,.2f}")
                solara.Text(f"{convert_area(r['confidence_interval'], unit):,.2f}")
                solara.Text(f"{convert_area(r['srs_area_estimate'], unit):,.2f}")


@solara.component
def _Accuracy(results):
    rows = results.get("accuracy_rows", [])
    if not rows:
        return
    headers = [
        msg("analysis.tables.class_header"),
        msg("analysis.tables.users_header"),
        msg("analysis.tables.producers_header"),
        msg("analysis.tables.weighted_producers_header"),
    ]
    with solara.Column(gap="4px"):
        Section(msg("analysis.tables.accuracy_title"), "mdi-target")
        with solara.GridFixed(columns=len(headers)):
            for h in headers:
                solara.Text(h, style="font-weight: bold;")
            for r in rows:
                solara.Text(str(r["class_name"]))
                solara.Text(f"{r['users_accuracy'] * 100:.1f}%")
                solara.Text(f"{r['producers_accuracy'] * 100:.1f}%")
                solara.Text(f"{r['weighted_producers_accuracy'] * 100:.1f}%")


@solara.component
def _Downloads():
    items = [
        (
            msg("analysis.downloads.confusion"),
            app_state.export_confusion_matrix_csv(),
            "confusion_matrix.csv",
        ),
        (
            msg("analysis.downloads.area"),
            app_state.export_area_estimates_csv(),
            "area_estimates.csv",
        ),
        (
            msg("analysis.downloads.accuracy"),
            app_state.export_accuracy_csv(),
            "accuracy_table.csv",
        ),
        (
            msg("analysis.downloads.reference"),
            app_state.export_reference_csv(),
            "reference_input.csv",
        ),
    ]
    with solara.Column(gap="4px"):
        Section(msg("analysis.downloads.title"), "mdi-download")
        DownloadMenu(items)
