# tests/test_analysis_dashboard.py
import solara

from component.model import app_state
from component.widget.analysis_dashboard import (
    AnalysisDashboardModal,
    dashboard_kpis,
)
from component.widget.echarts import EChartsWidget, RawEChartsWidget
from tests.test_analysis_charts import _RESULTS


def test_dashboard_kpis():
    results = {**_RESULTS, "overall_accuracy": 0.85, "confidence_level": 95.0}
    k = dashboard_kpis(results)
    assert k["overall_accuracy_pct"] == 85.0
    assert k["confidence_level"] == 95.0
    assert k["n_samples"] == 20
    assert k["n_classes"] == 2


def test_modal_renders_all_charts_when_open():
    app_state.analysis_results.value = {
        **_RESULTS,
        "overall_accuracy": 0.85,
        "confidence_level": 95.0,
    }
    open_r = solara.reactive(True)
    _, rc = solara.render(
        AnalysisDashboardModal(open_r, theme_toggle=None), handle_error=False
    )
    # 3 typed charts (accuracy, area, pie) + 1 raw (heatmap)
    assert len(rc.find(EChartsWidget).widgets) == 3
    assert len(rc.find(RawEChartsWidget).widgets) == 1
