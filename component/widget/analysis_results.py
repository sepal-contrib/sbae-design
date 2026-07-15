"""Rendered analysis outputs: matrices, tables, overall accuracy, export."""

import solara

from component.model import app_state
from component.scripts.accuracy import convert_area
from component.widget.custom_widgets import DownloadMenu, Section


def _unit_label(unit: str) -> str:
    return "ha" if unit == "ha" else "m²"


@solara.component
def AnalysisResultsView(theme_toggle=None):
    results = app_state.analysis_results.value
    if not results:
        return
    from component.widget.analysis_dashboard import AnalysisSummaryCard

    AnalysisSummaryCard(theme_toggle=theme_toggle)


@solara.component
def _OverallAccuracy(results):
    oa = results.get("overall_accuracy", 0.0) * 100
    ci = results.get("confidence_level", 95.0)
    with solara.Column(gap="4px"):
        Section("Overall accuracy", "mdi-bullseye-arrow")
        solara.Markdown(f"**{oa:.1f}%**  ·  confidence level {ci:.0f}%")


@solara.component
def _ConfusionMatrix(results):
    cm = results.get("confusion_matrix")
    if not cm:
        return
    with solara.Column(gap="4px"):
        Section("Error matrix", "mdi-grid", "rows = map, cols = reference")
        with solara.GridFixed(columns=len(cm["columns"]) + 1):
            solara.Text("map\\ref", style="font-weight: bold;")
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
        "Class",
        "Samples",
        f"Map area ({u})",
        f"Adj. area ({u})",
        f"± CI ({u})",
        f"SRS area ({u})",
    ]
    with solara.Column(gap="4px"):
        Section(
            "Area estimates", "mdi-chart-box-outline", "Error-adjusted (Olofsson 2014)"
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
    headers = ["Class", "User's", "Producer's", "Weighted PA"]
    with solara.Column(gap="4px"):
        Section("Accuracy by class", "mdi-target")
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
            "Error matrix",
            app_state.export_confusion_matrix_csv(),
            "confusion_matrix.csv",
        ),
        (
            "Area estimates",
            app_state.export_area_estimates_csv(),
            "area_estimates.csv",
        ),
        ("Accuracy table", app_state.export_accuracy_csv(), "accuracy_table.csv"),
        ("Reference input", app_state.export_reference_csv(), "reference_input.csv"),
    ]
    with solara.Column(gap="4px"):
        Section("Downloads", "mdi-download")
        DownloadMenu(items)
