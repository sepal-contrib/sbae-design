"""Bar chart of error-adjusted area per class, with confidence-interval bars."""

import solara
from ipecharts.option import Grid, Option, Title, Tooltip, XAxis, YAxis
from ipecharts.option.series import Bar, Custom

from component.model import app_state
from component.scripts.accuracy import convert_area
from component.widget.echarts import EChartsWidget


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
        renderItem={
            "__raw__": (
                "function (params, api) {"
                "  var cat = api.value(0);"
                "  var lo = api.coord([api.value(1), cat]);"
                "  var hi = api.coord([api.value(2), cat]);"
                "  var h = 6;"
                "  var style = {stroke: '#333', lineWidth: 1.5};"
                "  return {type:'group', children:["
                "    {type:'line', shape:{x1:lo[0],y1:lo[1],x2:hi[0],y2:hi[1]}, style:style},"
                "    {type:'line', shape:{x1:lo[0],y1:lo[1]-h,x2:lo[0],y2:lo[1]+h}, style:style},"
                "    {type:'line', shape:{x1:hi[0],y1:hi[1]-h,x2:hi[0],y2:hi[1]+h}, style:style}"
                "  ]};"
                "}"
            )
        },
        encode={"x": [1, 2], "y": 0},
    )

    option = Option(
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
    with solara.Card("Area chart"):
        EChartsWidget.element(
            option=option,
            style={"height": "420px", "width": "100%"},
            theme_toggle=theme_toggle,
        )
