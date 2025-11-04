from typing import Dict, Optional

import pandas as pd
import solara
from ipecharts.option import Grid, Legend, Option, Title, Tooltip, XAxis, YAxis
from ipecharts.option.series import Bar, Line, Pie
from solara import v

from component.model import app_state
from component.scripts.stratified import calculate_per_class_moe_for_allocation
from component.tile.class_editor import class_editor_table
from component.widget.echarts import EChartsWidget


@solara.component
def Summary(theme_toggle=None):
    """Right panel content with progress and summary."""
    show_editor_dialog, set_show_editor_dialog = solara.use_state(False)

    # Subscribe to sample_results changes to ensure component re-renders
    sample_results = app_state.sample_results.value

    # Get sampling method from app_state (not sample_results, as it may be None)
    sampling_method = app_state.sampling_method.value

    with solara.Column():
        statistics_summary(
            area_data=app_state.area_data.value,
            sample_results=sample_results,
            sample_points=app_state.sample_points.value,
        )

        precision_curve_graph(theme_toggle=theme_toggle)

        # Only show per-class precision for stratified sampling
        if sampling_method == "stratified":
            area_proportion_pie_chart(theme_toggle=theme_toggle)

            per_class_precision_graph(
                show_editor_dialog=show_editor_dialog,
                set_show_editor_dialog=set_show_editor_dialog,
                theme_toggle=theme_toggle,
            )

    with v.Dialog(
        v_model=show_editor_dialog, on_v_model=set_show_editor_dialog, max_width=900
    ):
        with v.Card():
            v.CardTitle(children=["Edit Classes & Sample Allocation"])
            with v.CardText(style="max-height: 70vh; overflow-y: auto;"):
                if app_state.sample_results.value:
                    class_editor_table()


def statistics_summary(
    area_data: Optional[pd.DataFrame] = None,
    sample_results: Optional[Dict] = None,
    sample_points: Optional[pd.DataFrame] = None,
) -> None:
    """Display summary statistics.

    Args:
        area_data: DataFrame with area information
        sample_results: Sample calculation results
        sample_points: Generated sample points
    """
    with solara.Row(gap="4px", style="flex-wrap: wrap;"):
        if area_data is not None and not area_data.empty:
            total_area = area_data["map_area"].sum() / 10000
            n_classes = len(area_data)

            solara.v.Chip(
                small=True,
                label=True,
                outlined=True,
                children=[f"{total_area:,.1f} ha"],
            )
            solara.v.Chip(
                small=True,
                label=True,
                outlined=True,
                children=[f"{n_classes} classes"],
            )

        # Create MOE chip unconditionally so hooks count stays stable between renders.
        moe_label = (
            f"MOE: {sample_results.get('target_error', 'N/A')}%"
            if sample_results
            else "MOE: N/A"
        )
        moe_chip = solara.v.Chip(
            small=True,
            label=True,
            outlined=True,
            children=[moe_label],
        )

        def set_v_on():
            # Only attempt to enable tooltip behavior when we actually have results.
            if not sample_results:
                return
            try:
                widget = solara.get_widget(moe_chip)
                widget.v_on = "tooltip.on"
            except Exception:
                # If the widget isn't available yet, ignore and let effect run later.
                pass

        # Call the effect unconditionally to avoid conditional hook usage; its body
        # will early-return when there's no sample_results.
        solara.use_effect(set_v_on, [moe_chip])

        if sample_results:
            with solara.v.Tooltip(
                bottom=True,
                max_width=300,
                v_slots=[
                    {
                        "name": "activator",
                        "variable": "tooltip",
                        "children": [moe_chip],
                    }
                ],
            ):
                solara.Text(
                    "Margin of Error: The range of uncertainty in the overall accuracy estimate at the specified confidence level. Lower MOE indicates higher precision."
                )

            solara.v.Chip(
                small=True,
                label=True,
                outlined=True,
                children=[f"n={sample_results.get('total_samples', 'N/A')}"],
            )


def precision_curve_graph(theme_toggle=None) -> None:
    """Display precision curve graph showing MOE vs sample size relationship."""
    sample_results = app_state.sample_results.value
    if not sample_results:
        return

    precision_curve = sample_results.get("precision_curve", [])
    if not precision_curve:
        return

    solara.HTML(tag="div", style="height: 12px;")

    # Extract data from precision curve
    sample_sizes = [point["sample_size"] for point in precision_curve]
    moe_percents = [round(point["moe_percent"], 2) for point in precision_curve]

    # Get current design point
    current_total = sample_results.get("total_samples", 0)
    current_moe = sample_results.get("current_moe_percent", 0)

    # Create line series for the precision curve
    line = Line(
        name="MOE vs Sample Size",
        data=[[x, y] for x, y in zip(sample_sizes, moe_percents)],
        smooth=True,
        lineStyle={"color": "#5470c6", "width": 2},
        itemStyle={"color": "#5470c6"},
    )

    # Create scatter series for current design point
    current_point = Line(
        name=f"Current Design (n={current_total})",
        data=[[current_total, round(current_moe, 2)]],
        type="scatter",
        symbolSize=12,
        itemStyle={"color": "#ee6666"},
    )

    # Create the option with title
    option = Option(
        backgroundColor="#1e1e1e00",
        title=Title(
            text="Precision Curve",
            left="center",
            textStyle={"fontSize": 13, "fontWeight": "normal"},
        ),
        xAxis=XAxis(
            type="value",
            name="Sample Size",
            nameLocation="middle",
            nameGap=25,
            nameTextStyle={"fontSize": 11},
        ),
        yAxis=YAxis(
            type="value",
            name="MOE (%)",
            nameLocation="middle",
            nameGap=35,
            nameTextStyle={"fontSize": 11},
        ),
        series=[line, current_point],
        tooltip=Tooltip(trigger="axis"),
        grid=Grid(left="18%", right="8%", top="18%", bottom="18%"),
    )

    # Create and display the chart
    EChartsWidget.element(
        option=option,
        style={"height": "220px", "width": "100%"},
        theme_toggle=theme_toggle,
    )

    with solara.Row(
        gap="4px", style="flex-wrap: wrap; margin-top: 6px; justify-content: center;"
    ):
        solara.v.Chip(
            x_small=True,
            label=True,
            outlined=True,
            children=[f"n={current_total}, MOE=±{current_moe:.2f}%"],
        )


def per_class_precision_graph(
    show_editor_dialog=None, set_show_editor_dialog=None, theme_toggle=None
) -> None:
    """Display per-class precision (MOE) given current allocation."""
    sample_results = app_state.sample_results.value
    if not sample_results:
        return

    allocation_dict = sample_results.get("allocation_dict", {})
    if not allocation_dict:
        return

    area_df = app_state.area_data.value
    if area_df is None or area_df.empty:
        return

    confidence_level = sample_results.get("confidence_level", 95.0) / 100.0

    moe_df = calculate_per_class_moe_for_allocation(
        allocation=allocation_dict,
        area_df=area_df,
        confidence_level=confidence_level,
        expected_accuracies=None,
        population_sizes=None,
        deff=1.0,
    )

    moe_df = moe_df.sort_values("moe_percent", ascending=False)

    solara.HTML(tag="div", style="height: 12px;")

    if set_show_editor_dialog:
        with solara.Tooltip("Edit class names and sample allocations"):
            solara.Button(
                "Edit class names & Allocations",
                icon_name="mdi-pencil",
                on_click=lambda: set_show_editor_dialog(True),
                # color="secondary",
                outlined=True,
                block=True,
                style="margin-bottom: 12px;",
                small=True,
            )

    class_names = moe_df["class_name"].tolist()
    moe_values = moe_df["moe_percent"].tolist()
    moe_df["samples"].tolist()

    bar_colors = [
        "#ee6666" if moe > 15 else "#fac858" if moe > 10 else "#91cc75"
        for moe in moe_values
    ]

    bar_series = Bar(
        name="MOE (%)",
        data=[
            {
                "value": round(moe, 2),
                "itemStyle": {"color": color},
            }
            for moe, color in zip(moe_values, bar_colors)
        ],
        label={
            "show": True,
            "position": "right",
            "formatter": "{c}%",
            "fontSize": 10,
        },
    )

    option = Option(
        backgroundColor="#1e1e1e00",
        title=Title(
            text="Per-Class Precision",
            left="center",
            textStyle={"fontSize": 13, "fontWeight": "normal"},
        ),
        xAxis=XAxis(
            type="value",
            name="MOE (%)",
            nameLocation="middle",
            nameGap=25,
            nameTextStyle={"fontSize": 11},
        ),
        yAxis=YAxis(
            type="category",
            data=class_names,
            axisLabel={"fontSize": 10},
        ),
        series=[bar_series],
        tooltip=Tooltip(
            trigger="axis",
            axisPointer={"type": "shadow"},
            formatter="{b}: ±{c}%",
        ),
        grid=Grid(left="25%", right="12%", top="15%", bottom="12%"),
    )

    EChartsWidget.element(
        option=option,
        style={"height": "280px", "width": "100%"},
        theme_toggle=theme_toggle,
    )

    max_moe_row = moe_df.iloc[0]
    with solara.Row(
        gap="4px", style="flex-wrap: wrap; margin-top: 6px; justify-content: center;"
    ):
        solara.v.Chip(
            x_small=True,
            label=True,
            outlined=True,
            children=[
                f"Max MOE: {max_moe_row['class_name']} (±{max_moe_row['moe_percent']:.1f}%)"
            ],
        )


@solara.component
def area_proportion_pie_chart(theme_toggle=None):
    """Pie chart showing the proportion of each class by area."""
    area_data = app_state.area_data.value
    class_colors = app_state.class_colors.value

    if area_data is None or area_data.empty:
        return

    total_area = area_data["map_area"].sum()
    pie_data = []
    chart_colors = []

    # Default colors as fallback
    default_colors = [
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

    for idx, row in area_data.iterrows():
        map_code = row["map_code"]
        current_name = row.get("map_edited_class", f"Class {map_code}")
        area_pct = 100 * row["map_area"] / total_area

        # Get color from extracted palette or use default
        color = class_colors.get(map_code, default_colors[idx % len(default_colors)])
        chart_colors.append(color)

        pie_data.append(
            {"value": round(area_pct, 2), "name": f"{current_name} ({area_pct:.1f}%)"}
        )

    pie = Pie(
        data=pie_data,
        radius=[50, 100],
        itemStyle={"borderRadius": 5, "borderColor": "#fff", "borderWidth": 2},
        label={"show": False, "position": "center"},
        emphasis={
            "label": {
                "show": True,
                "fontSize": 12,
            }
        },
    )

    option = Option(
        backgroundColor="#1e1e1e00",
        legend=Legend(bottom=0),
        series=[pie],
        color=chart_colors,
        title=Title(
            text="Proportion by Area",
            left="center",
            textStyle={"fontSize": 13, "fontWeight": "normal"},
        ),
    )

    EChartsWidget.element(
        option=option,
        style={"height": "380px", "width": "100%"},
        # width="100%",
        # height="300px",
        theme_toggle=theme_toggle,
    )
