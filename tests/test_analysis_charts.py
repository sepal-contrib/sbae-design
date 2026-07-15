# tests/test_analysis_charts.py
import solara

from component.widget.analysis_chart import (
    AccuracyByClassChart,
    ConfusionMatrixChart,
    confusion_heatmap_data,
)
from component.widget.echarts import EChartsWidget, RawEChartsWidget

_RESULTS = {
    "confusion_matrix": {
        "index": [11, 12],
        "columns": [11, 12],
        "data": [[8, 2], [1, 9]],
    },
    "accuracy_rows": [
        {
            "class_name": "Forest",
            "map_code": 11,
            "users_accuracy": 0.8,
            "producers_accuracy": 0.89,
            "weighted_producers_accuracy": 0.87,
        },
        {
            "class_name": "Non-forest",
            "map_code": 12,
            "users_accuracy": 0.9,
            "producers_accuracy": 0.82,
            "weighted_producers_accuracy": 0.83,
        },
    ],
    "class_estimates": [
        {
            "class_name": "Forest",
            "map_code": 11,
            "number_samples": 10,
            "area_estimate": 100.0,
            "confidence_interval": 5.0,
            "map_pixel_count": 90.0,
            "srs_area_estimate": 98.0,
        },
        {
            "class_name": "Non-forest",
            "map_code": 12,
            "number_samples": 10,
            "area_estimate": 50.0,
            "confidence_interval": 3.0,
            "map_pixel_count": 60.0,
            "srs_area_estimate": 52.0,
        },
    ],
}


def test_confusion_heatmap_data_shapes():
    x, y, triples, max_count = confusion_heatmap_data(_RESULTS["confusion_matrix"])
    assert x == ["11", "12"]
    assert y == ["11", "12"]
    assert len(triples) == 4
    assert [0, 0, 8.0] in triples and [1, 1, 9.0] in triples
    assert max_count == 9.0


def test_confusion_matrix_chart_renders_raw_widget():
    _, rc = solara.render(
        ConfusionMatrixChart(_RESULTS, theme_toggle=None), handle_error=False
    )
    widgets = rc.find(RawEChartsWidget).widgets
    assert len(widgets) == 1
    opt = widgets[0].option
    assert opt["backgroundColor"] == "#1e1e1e00"
    assert opt["series"][0]["type"] == "heatmap"
    assert "visualMap" in opt


def test_accuracy_by_class_chart_two_series():
    _, rc = solara.render(
        AccuracyByClassChart(_RESULTS, theme_toggle=None), handle_error=False
    )
    widgets = rc.find(EChartsWidget).widgets
    assert len(widgets) == 1
    series = widgets[0].option.series
    assert [s.name for s in series] == ["User's", "Producer's"]
    assert series[0].data == [80.0, 90.0]
