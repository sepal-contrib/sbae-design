"""Results summary card (KPIs + button) and the analysis dashboard modal it opens."""

import solara

from component.model import app_state
from component.widget.analysis_chart import (
    AccuracyByClassChart,
    AreaEstimateChart,
    AreaProportionChart,
    ConfusionMatrixChart,
)
from component.widget.analysis_results import (
    _Accuracy,
    _AreaEstimates,
    _ConfusionMatrix,
    _Downloads,
)


def dashboard_kpis(results: dict) -> dict:
    """Scalar KPIs for the summary card / modal header."""
    rows = results.get("class_estimates", [])
    return {
        "overall_accuracy_pct": round(results.get("overall_accuracy", 0.0) * 100, 1),
        "confidence_level": results.get("confidence_level", 95.0),
        "n_samples": int(sum(r.get("number_samples", 0) for r in rows)),
        "n_classes": len(rows),
    }


@solara.component
def _DashboardHeader(results: dict):
    k = dashboard_kpis(results)
    with solara.Row(style="align-items: center; gap: 16px; flex-wrap: wrap;"):
        solara.Text(
            f"Overall {k['overall_accuracy_pct']:.1f}%",
            style="font-weight: 700; font-size: 18px;",
        )
        solara.Text(f"CL {k['confidence_level']:.0f}%")
        solara.Text(f"n={k['n_samples']} samples")
        solara.Text(f"{k['n_classes']} classes")
        solara.v.Spacer()
        solara.ToggleButtonsSingle(
            value=app_state.analysis_area_unit.value,
            values=["ha", "m2"],
            on_value=app_state.analysis_area_unit.set,
        )


@solara.component
def AnalysisDashboardModal(open, theme_toggle=None):
    results = app_state.analysis_results.value
    if not results:
        return
    unit = app_state.analysis_area_unit.value
    with solara.v.Dialog(
        v_model=open.value, on_v_model=open.set, max_width=1100, eager=True
    ):
        with solara.v.Card():
            solara.v.CardTitle(children=["Accuracy assessment results"])
            with solara.v.CardText(style="max-height: 80vh; overflow-y: auto;"):
                _DashboardHeader(results)
                with solara.ColumnsResponsive(6, small=12):
                    ConfusionMatrixChart(results, theme_toggle=theme_toggle)
                    AccuracyByClassChart(results, theme_toggle=theme_toggle)
                    AreaEstimateChart(results, unit, theme_toggle=theme_toggle)
                    AreaProportionChart(results, theme_toggle=theme_toggle)
                with solara.Details("Tables"):
                    _AreaEstimates(results, unit)
                    _Accuracy(results)
                    _ConfusionMatrix(results)
            with solara.v.CardActions():
                solara.v.Spacer()
                solara.Button("Close", text=True, on_click=lambda: open.set(False))


@solara.component
def AnalysisSummaryCard(theme_toggle=None):
    results = app_state.analysis_results.value
    # Hook must run unconditionally, before the early return, for hook-order
    # stability across renders (see solara's rules-of-hooks).
    open_modal = solara.use_reactive(False)
    if not results:
        return
    k = dashboard_kpis(results)
    with solara.Column(gap="8px"):
        solara.Text(
            f"{k['overall_accuracy_pct']:.1f}%",
            style="font-weight: 700; font-size: 22px;",
        )
        solara.Text(
            f"overall accuracy - CL {k['confidence_level']:.0f}% - "
            f"n={k['n_samples']} - {k['n_classes']} classes",
            style="opacity: 0.8;",
        )
        solara.Button(
            "Ver dashboard",
            icon_name="mdi-view-dashboard",
            color="primary",
            block=True,
            on_click=lambda: open_modal.set(True),
        )
        _Downloads()
    AnalysisDashboardModal(open_modal, theme_toggle=theme_toggle)
