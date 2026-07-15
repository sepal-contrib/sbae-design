"""Charts for the analysis results.

Error-adjusted area bars, confusion-matrix heatmap, accuracy-by-class bars,
and estimated-area proportion pie.
"""

import solara
from ipecharts.option import Grid, Legend, Option, Title, Tooltip, XAxis, YAxis
from ipecharts.option.series import Bar, Custom, Pie
from ipecharts.tools import encode_js_fn

from component.model import app_state
from component.scripts.accuracy import convert_area
from component.widget.echarts import EChartsWidget, RawEChartsWidget


@solara.component
def AreaEstimateChart(results: dict, unit: str, theme_toggle=None):
    rows = results.get("class_estimates", [])
    if not rows:
        return
    u = "ha" if unit == "ha" else "m²"
    names = [r["class_name"] for r in rows]
    areas = [round(convert_area(r["area_estimate"], unit), 2) for r in rows]
    cis = [round(convert_area(r["confidence_interval"], unit), 2) for r in rows]
    colors = app_state.class_colors.value or {}
    codes = [r["map_code"] for r in rows]

    bar = Bar(
        name=f"Adjusted area ({u})",
        data=[
            {"value": a, "itemStyle": {"color": colors.get(code, "#5470c6")}}
            for a, code in zip(areas, codes)
        ],
    )
    # error bars: horizontal segments [area-ci, area+ci] per category, drawn via Custom
    err_data = [[i, areas[i] - cis[i], areas[i] + cis[i]] for i in range(len(rows))]
    error_series = Custom(
        name="CI",
        data=err_data,
        renderItem=encode_js_fn(
            ["params", "api"],
            (
                "var cat = api.value(0);"
                "var lo = api.coord([api.value(1), cat]);"
                "var hi = api.coord([api.value(2), cat]);"
                "var h = 6;"
                "var style = {stroke: '#333', lineWidth: 1.5};"
                "return {type:'group', children:["
                "  {type:'line', shape:{x1:lo[0],y1:lo[1],x2:hi[0],y2:hi[1]}, style:style},"
                "  {type:'line', shape:{x1:lo[0],y1:lo[1]-h,x2:lo[0],y2:lo[1]+h}, style:style},"
                "  {type:'line', shape:{x1:hi[0],y1:hi[1]-h,x2:hi[0],y2:hi[1]+h}, style:style}"
                "]};"
            ),
        ),
        encode={"x": [1, 2], "y": 0},
    )

    option = Option(
        backgroundColor="#1e1e1e00",
        title=Title(
            text="Error-adjusted area by class",
            left="center",
            textStyle={"fontSize": 13, "fontWeight": "normal"},
        ),
        xAxis=XAxis(
            type="value", name=f"Area ({u})", nameLocation="middle", nameGap=28
        ),
        yAxis=YAxis(type="category", data=names, axisLabel={"fontSize": 10}),
        series=[bar, error_series],
        tooltip=Tooltip(trigger="axis", axisPointer={"type": "shadow"}),
        grid=Grid(left="22%", right="8%", top="14%", bottom="12%"),
    )
    # Card-less: the chart carries its own title, and the right panel avoids cards.
    EChartsWidget.element(
        option=option,
        style={"height": "420px", "width": "100%"},
        theme_toggle=theme_toggle,
    )


def confusion_heatmap_data(confusion_matrix: dict):
    """Reshape a confusion-matrix dict into echarts heatmap inputs.

    Returns (x_labels, y_labels, triples, max_count):
      x_labels = reference classes (columns) as strings
      y_labels = mapped classes (index) as strings
      triples  = [x_index, y_index, count] for every cell
      max_count = largest cell value, at least 1 (for visualMap.max)
    """
    x_labels = [str(c) for c in confusion_matrix.get("columns", [])]
    y_labels = [str(i) for i in confusion_matrix.get("index", [])]
    triples = []
    max_count = 0.0
    for y, row in enumerate(confusion_matrix.get("data", [])):
        for x, value in enumerate(row):
            count = float(value)
            triples.append([x, y, count])
            max_count = max(max_count, count)
    return x_labels, y_labels, triples, max(max_count, 1.0)


@solara.component
def ConfusionMatrixChart(results: dict, theme_toggle=None):
    cm = results.get("confusion_matrix")
    if not cm or not cm.get("data"):
        return
    x_labels, y_labels, triples, max_count = confusion_heatmap_data(cm)
    option = {
        "backgroundColor": "#1e1e1e00",
        "title": {
            "text": "Confusion matrix (map -> reference)",
            "left": "center",
            "textStyle": {"fontSize": 13, "fontWeight": "normal"},
        },
        "tooltip": {"position": "top"},
        "grid": {"top": "14%", "bottom": "18%", "left": "16%", "right": "8%"},
        "xAxis": {
            "type": "category",
            "data": x_labels,
            "name": "reference",
            "nameLocation": "middle",
            "nameGap": 28,
            "splitArea": {"show": True},
            "axisLabel": {"fontSize": 10},
        },
        "yAxis": {
            "type": "category",
            "data": y_labels,
            "name": "map",
            "splitArea": {"show": True},
            "axisLabel": {"fontSize": 10},
        },
        "visualMap": {
            "min": 0,
            "max": max_count,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "2%",
        },
        "series": [
            {
                "type": "heatmap",
                "data": triples,
                "label": {"show": True, "fontSize": 9},
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 6,
                        "shadowColor": "rgba(0,0,0,0.3)",
                    }
                },
            }
        ],
    }
    RawEChartsWidget.element(
        option=option,
        style={"height": "420px", "width": "100%"},
        theme_toggle=theme_toggle,
    )


@solara.component
def AccuracyByClassChart(results: dict, theme_toggle=None):
    rows = results.get("accuracy_rows", [])
    if not rows:
        return
    names = [r["class_name"] for r in rows]
    users = [round(r["users_accuracy"] * 100, 1) for r in rows]
    producers = [round(r["producers_accuracy"] * 100, 1) for r in rows]
    option = Option(
        backgroundColor="#1e1e1e00",
        title=Title(
            text="Accuracy by class",
            left="center",
            textStyle={"fontSize": 13, "fontWeight": "normal"},
        ),
        tooltip=Tooltip(trigger="axis", axisPointer={"type": "shadow"}),
        legend=Legend(bottom=0),
        xAxis=XAxis(
            type="category",
            data=names,
            axisLabel={"fontSize": 10, "interval": 0, "rotate": 30},
        ),
        yAxis=YAxis(type="value", name="%", max=100),
        grid=Grid(left="10%", right="6%", top="14%", bottom="18%"),
        series=[Bar(name="User's", data=users), Bar(name="Producer's", data=producers)],
    )
    EChartsWidget.element(
        option=option,
        style={"height": "420px", "width": "100%"},
        theme_toggle=theme_toggle,
    )


_PIE_FALLBACK = [
    "#5470c6",
    "#91cc75",
    "#fac858",
    "#ee6666",
    "#73c0de",
    "#3ba272",
    "#fc8452",
    "#9a60b4",
    "#ea7ccc",
]


@solara.component
def AreaProportionChart(results: dict, theme_toggle=None):
    rows = results.get("class_estimates", [])
    if not rows:
        return
    colors = app_state.class_colors.value or {}
    total = sum(max(r["area_estimate"], 0.0) for r in rows) or 1.0
    pie_data = []
    chart_colors = []
    for idx, r in enumerate(rows):
        pct = 100.0 * max(r["area_estimate"], 0.0) / total
        chart_colors.append(
            colors.get(r["map_code"], _PIE_FALLBACK[idx % len(_PIE_FALLBACK)])
        )
        pie_data.append(
            {"value": round(pct, 2), "name": f"{r['class_name']} ({pct:.1f}%)"}
        )
    pie = Pie(
        data=pie_data,
        radius=[50, 100],
        itemStyle={"borderRadius": 5, "borderColor": "#fff", "borderWidth": 2},
        label={"show": False, "position": "center"},
        emphasis={"label": {"show": True, "fontSize": 12}},
    )
    option = Option(
        backgroundColor="#1e1e1e00",
        legend=Legend(bottom=0),
        series=[pie],
        color=chart_colors,
        title=Title(
            text="Estimated area proportion",
            left="center",
            textStyle={"fontSize": 13, "fontWeight": "normal"},
        ),
    )
    EChartsWidget.element(
        option=option,
        style={"height": "420px", "width": "100%"},
        theme_toggle=theme_toggle,
    )
