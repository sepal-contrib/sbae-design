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


# Short (<= 3 lines) explanation shown under the dashboard KPI tiles.
_DASHBOARD_EXPLANATION = (
    "Error-adjusted area estimates and class accuracies from your reference "
    "sample, following Olofsson et al. (2014). Overall accuracy is the share of "
    "reference points classified correctly; the charts and tables below break it "
    "down per class with confidence intervals."
)


@solara.component
def _KpiCard(label: str, value: str):
    """A single dashboard stat tile: prominent value + muted label (no icon).

    A plain theme-aware ``solara.Card`` (no custom background) so it reads as a
    normal surface in both light and dark; ``flex: 1`` lets the four share one row.
    """
    with solara.Card(margin=0, elevation=2, style="flex: 1 1 0; min-width: 0;"):
        with solara.Column(gap="2px", style="align-items: center;"):
            solara.Text(
                value,
                style="font-weight: 700; font-size: 22px; line-height: 1.15;",
            )
            solara.Text(
                label,
                classes=["caption"],
                style="opacity: 0.7; line-height: 1.2; text-align: center;",
            )


@solara.component
def _DashboardKpiCards(results: dict):
    """Overall accuracy / confidence / samples / classes as stat tiles, one row."""
    k = dashboard_kpis(results)
    items = [
        ("Overall accuracy", f"{k['overall_accuracy_pct']:.1f}%"),
        ("Confidence level", f"{k['confidence_level']:.0f}%"),
        ("Reference samples", f"{k['n_samples']:,}"),
        ("Classes", f"{k['n_classes']}"),
    ]
    with solara.Row(gap="12px", style="flex-wrap: nowrap; margin-bottom: 8px;"):
        for label, value in items:
            _KpiCard(label, value)


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
        v_model=open.value, on_v_model=open.set, max_width=1100, eager=True
    ):
        with solara.v.Card():
            solara.v.CardTitle(children=["Accuracy assessment results"])
            with solara.v.CardText(style="max-height: 80vh; overflow-y: auto;"):
                solara.v.Html(tag="div", children=[resizer], style_="display: none;")
                _DashboardKpiCards(results)
                solara.Text(
                    _DASHBOARD_EXPLANATION,
                    style=(
                        "display: block; text-align: center; opacity: 0.7; "
                        "font-size: 13px; margin: 0 auto 16px; max-width: 720px;"
                    ),
                )
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
        AreaProportionChart(results, theme_toggle=theme_toggle, legend_width=None)
        solara.Button(
            "View dashboard",
            color="primary",
            block=True,
            small=True,
            on_click=lambda: open_modal.set(True),
        )
        _Downloads()
    AnalysisDashboardModal(open_modal, theme_toggle=theme_toggle)
