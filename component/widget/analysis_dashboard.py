"""Results summary card (KPIs + button) and the analysis dashboard modal it opens."""

import ipyvuetify as ipv
import solara
from traitlets import Int, Unicode

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


class _DialogResizer(ipv.VuetifyTemplate):
    """Dispatch a window resize event whenever ``tick`` changes.

    ECharts measures zero width when first mounted inside a not-yet-visible
    dialog, so the charts render tiny. Bumping ``tick`` once the dialog opens
    forces the widgets to re-layout to their real column width. Mirrors the
    resize trick used by the sepal-gee-bundle dashboards.
    """

    tick = Int(0).tag(sync=True)
    template = Unicode("""
        <script class='sbae-dashboard-resize'>
        {
            watch: {
                tick() {
                    this.$nextTick(() => {
                        setTimeout(() => {
                            window.dispatchEvent(new Event("resize"));
                        }, 120);
                    });
                }
            }
        }
        </script>
        """).tag(sync=True)


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
def _KpiStat(icon: str, label: str, value: str, hint: str):
    """One compact KPI: a primary icon, a faded label, and the value.

    Mirrors the sepal-gee-bundle dashboard stat items (icon + caption label +
    body value in a dense list item, no surrounding card). ``hint`` rides along
    as a hover tooltip so each metric keeps its explanation without clutter.
    """
    with solara.Tooltip(hint):
        with solara.v.Col(cols="auto", class_="pa-0"):
            with solara.v.ListItem(dense=True, class_="pa-0 pr-4"):
                with solara.v.ListItemIcon(class_="mr-2 my-auto"):
                    solara.v.Icon(small=True, color="primary", children=[icon])
                with solara.v.ListItemContent(class_="py-1"):
                    solara.v.ListItemTitle(
                        class_="caption", style_="opacity: 0.6;", children=[label]
                    )
                    solara.v.ListItemSubtitle(class_="body-2", children=[value])


@solara.component
def _DashboardKpiCards(results: dict):
    """Overall accuracy / confidence / samples / classes as compact stat items."""
    k = dashboard_kpis(results)
    items = [
        (
            "mdi-target",
            "Overall accuracy",
            f"{k['overall_accuracy_pct']:.1f}%",
            "Share of reference points classified correctly.",
        ),
        (
            "mdi-percent-outline",
            "Confidence level",
            f"{k['confidence_level']:.0f}%",
            "Probability that the ± intervals contain the true value.",
        ),
        (
            "mdi-map-marker-multiple",
            "Reference samples",
            f"{k['n_samples']:,}",
            "Total reference points assessed.",
        ),
        (
            "mdi-shape-outline",
            "Classes",
            f"{k['n_classes']}",
            "Number of map classes evaluated.",
        ),
    ]
    with solara.v.Row(dense=True, align="center", justify="center", class_="mb-3"):
        for icon, label, value, hint in items:
            _KpiStat(icon, label, value, hint)


@solara.component
def AnalysisDashboardModal(open, theme_toggle=None):
    # Hooks must run unconditionally, before the early return, for hook-order
    # stability across renders (see solara's rules-of-hooks).
    #
    # Kick ECharts into a re-layout each time the dialog opens; otherwise the
    # charts, mounted eagerly while the dialog was hidden, stay tiny.
    resizer = solara.use_memo(_DialogResizer, [])

    def _resize_on_open():
        if open.value:
            resizer.tick += 1

    solara.use_effect(_resize_on_open, [open.value])

    results = app_state.analysis_results.value
    if not results:
        return
    unit = app_state.analysis_area_unit.value

    with solara.v.Dialog(
        v_model=open.value, on_v_model=open.set, max_width=1400, eager=True
    ):
        with solara.v.Card():
            solara.v.CardTitle(children=["Accuracy assessment results"])
            with solara.v.CardText(style="max-height: 80vh; overflow-y: auto;"):
                solara.v.Html(tag="div", children=[resizer], style_="display: none;")
                _DashboardKpiCards(results)
                with solara.ColumnsResponsive(6, small=12):
                    ConfusionMatrixChart(results, theme_toggle=theme_toggle)
                    AccuracyByClassChart(results, theme_toggle=theme_toggle)
                    AreaEstimateChart(results, unit, theme_toggle=theme_toggle)
                    AreaProportionChart(results, theme_toggle=theme_toggle)
                with solara.Details("Tables"):
                    # Space the three tables apart so they don't read as one block.
                    with solara.Column(style="gap: 28px; padding-top: 8px;"):
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
        # Compact stat chips + one graph, mirroring the design tab's summary
        # style so both tabs feel consistent.
        with solara.Row(gap="4px", justify="center", style="flex-wrap: wrap;"):
            for chip_text in (
                f"Overall {k['overall_accuracy_pct']:.1f}%",
                f"CL {k['confidence_level']:.0f}%",
                f"n={k['n_samples']:,}",
                f"{k['n_classes']} classes",
            ):
                solara.v.Chip(
                    small=True, label=True, outlined=True, children=[chip_text]
                )
        AreaProportionChart(
            results, theme_toggle=theme_toggle, legend_width=None, card=False
        )
        solara.Button(
            "View dashboard",
            color="primary",
            block=True,
            small=True,
            on_click=lambda: open_modal.set(True),
        )
        _Downloads()
    AnalysisDashboardModal(open_modal, theme_toggle=theme_toggle)
