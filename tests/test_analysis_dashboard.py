# tests/test_analysis_dashboard.py
import ipyvuetify as v
import pytest
import solara

from component.model import app_state
from component.widget.analysis_dashboard import (
    AnalysisDashboardModal,
    dashboard_kpis,
)
from component.widget.analysis_results import AnalysisResultsView
from component.widget.echarts import EChartsWidget, RawEChartsWidget
from tests.test_analysis_charts import _RESULTS


@pytest.fixture(autouse=True)
def _reset_analysis_results():
    """Reset the shared analysis_results reactive after each test.

    Several tests in this module set ``app_state.analysis_results.value``
    directly on the shared singleton without resetting it, which could
    otherwise leak into tests that run later (in this file or others).
    """
    yield
    app_state.analysis_results.value = None


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


def test_summary_card_shows_kpis_and_button():
    app_state.analysis_results.value = {
        **_RESULTS,
        "overall_accuracy": 0.85,
        "confidence_level": 95.0,
    }
    _, rc = solara.render(AnalysisResultsView(theme_toggle=None), handle_error=False)
    text = " ".join(str(c) for w in rc.find(v.Html).widgets for c in (w.children or []))
    assert "85.0%" in text
    # a "Ver dashboard" button exists among the rendered buttons
    labels = [str(c) for b in rc.find(v.Btn).widgets for c in (b.children or [])]
    assert any("dashboard" in lbl.lower() for lbl in labels)
