"""Rendered analysis outputs: matrices, tables, overall accuracy, export."""

import solara

from component.message import use_translator
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
    ms = use_translator()
    cm = results.get("confusion_matrix")
    if not cm:
        return
    tables = ms.analysis.tables
    with solara.Column(gap="4px"):
        Section(tables.confusion_title, "mdi-grid", tables.confusion_description)
        with solara.GridFixed(columns=len(cm["columns"]) + 1):
            solara.Text(tables.confusion_corner, style="font-weight: bold;")
            for c in cm["columns"]:
                solara.Text(str(c), style="font-weight: bold;")
            for code, row in zip(cm["index"], cm["data"]):
                solara.Text(str(code), style="font-weight: bold;")
                for v in row:
                    solara.Text(f"{v:g}")


@solara.component
def _AreaEstimates(results, unit):
    ms = use_translator()
    rows = results.get("class_estimates", [])
    if not rows:
        return
    u = _unit_label(unit)
    tables = ms.analysis.tables
    headers = [
        tables.class_header,
        tables.samples_header,
        tables.map_area_header.format(u),
        tables.adjusted_area_header.format(u),
        tables.confidence_interval_header.format(u),
        tables.srs_area_header.format(u),
    ]
    with solara.Column(gap="4px"):
        Section(tables.area_title, "mdi-chart-box-outline", tables.area_description)
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
    ms = use_translator()
    rows = results.get("accuracy_rows", [])
    if not rows:
        return
    tables = ms.analysis.tables
    headers = [
        tables.class_header,
        tables.users_header,
        tables.producers_header,
        tables.weighted_producers_header,
    ]
    with solara.Column(gap="4px"):
        Section(tables.accuracy_title, "mdi-target")
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
    ms = use_translator()
    downloads = ms.analysis.downloads
    items = [
        (
            downloads.confusion,
            app_state.export_confusion_matrix_csv(),
            "confusion_matrix.csv",
        ),
        (
            downloads.area,
            app_state.export_area_estimates_csv(),
            "area_estimates.csv",
        ),
        (downloads.accuracy, app_state.export_accuracy_csv(), "accuracy_table.csv"),
        (downloads.reference, app_state.export_reference_csv(), "reference_input.csv"),
    ]
    with solara.Column(gap="4px"):
        Section(downloads.title, "mdi-download")
        DownloadMenu(items)
