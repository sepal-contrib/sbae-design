import solara
from ipecharts.option import Grid, Legend, Option, Tooltip, XAxis, YAxis
from ipecharts.option.series import Bar, Line
from solara.alias import rv

from component.message import get_translator, use_translator
from component.model import app_state
from component.scripts.calculations import (
    calculate_current_moe,
    calculate_per_class_moe_for_allocation,
    calculate_precision_curve,
    calculate_sample_design,
)
from component.widget.echarts import EChartsWidget

# The formulas are notation, not prose, so they stay out of the catalog.
_PER_CLASS_MOE_FORMULA = (
    "$$MOE_h = Z \\times \\sqrt{DEFF \\times \\frac{p_h(1-p_h)}{n_h}} "
    "\\times \\sqrt{\\frac{N_h - n_h}{N_h - 1}} \\times 100$$"
)
_PRECISION_CURVE_FORMULA = "$$MOE = Z \\times \\sqrt{\\frac{OA \\times (1 - OA)}{n}}$$"


@solara.component
def SampleCalculationTile(theme_state=None):
    """Step 3: Calculate Sample Size Dialog."""
    ms = use_translator()
    with solara.Column():
        solara.HTML(tag="h2", unsafe_innerHTML=ms.calculator.title)
        with solara.Column(gap="8px", style="margin-bottom: 16px;"):
            solara.Text(ms.calculator.intro)
            solara.Text(
                ms.calculator.parameters_title,
                style="font-weight: bold; margin-top: 8px;",
            )
            solara.Text(ms.calculator.parameter_accuracy)
            solara.Text(ms.calculator.parameter_confidence)
            solara.Text(ms.calculator.parameter_minimum)

        sample_size_calculator(ms=ms)

        # Show allocation table and per-class precision only for stratified sampling
        if (
            app_state.sample_results.value is not None
            and app_state.sample_results.value
        ):
            sampling_method = app_state.sample_results.value.get(
                "sampling_method", "stratified"
            )

            # Allocation table and per-class charts only relevant for stratified
            if sampling_method == "stratified":
                sample_allocation_table(ms=ms)
                if app_state.sample_results.value.get("precision_curve"):
                    per_class_precision_chart(theme_state=theme_state, ms=ms)

            solara.Success(ms.calculator.complete)

        # Display precision curve for all methods
        if app_state.sample_results.value and app_state.sample_results.value.get(
            "precision_curve"
        ):
            precision_curve_info(theme_state=theme_state, ms=ms)


def sample_size_calculator(ms=None) -> None:
    """Sample size calculation component."""
    ms = ms if ms is not None else get_translator()

    def handle_calculate_samples():
        """Handle sample size calculation."""
        if not app_state.is_ready_for_calculation():
            app_state.add_error(ms.calculator.needs_map)
            return

        try:
            app_state.set_processing_status(ms.calculator.calculating)

            # Get current parameters
            area_data = app_state.area_data.value
            target_error = (
                app_state.target_error.value / 100.0
            )  # Convert percentage to decimal
            confidence_level = (
                app_state.confidence_level.value / 100.0
            )  # Convert percentage to decimal
            min_samples = app_state.min_samples_per_class.value
            expected_oa = app_state.expected_accuracy.value / 100.0

            # Determine sampling method and optional override
            sampling_method = app_state.sampling_method.value
            allocation_method = (
                app_state.stratified_allocation_method.value.capitalize()
            )
            total_override = None
            if sampling_method in ("simple", "systematic"):
                total_override = int(app_state.simple_total_samples.value)

            # Calculate sample allocation (allow override for simple/systematic sampling)
            allocation_dict = calculate_sample_design(
                area_df=area_data,
                objective="Overall Accuracy",
                target_oa=expected_oa,
                allowable_error=target_error,
                confidence_level=confidence_level,
                min_samples_per_class=min_samples,
                allocation_method=allocation_method,
                total_samples_override=total_override,
            )

            # Create properly formatted results dictionary
            total_samples = sum(allocation_dict.values()) if allocation_dict else 0

            # Only calculate precision curve for simple/systematic sampling
            # For stratified, the curve is not theoretically valid
            precision_curve_df = None
            current_moe = None
            if sampling_method in ("simple", "systematic"):
                precision_curve_df = calculate_precision_curve(
                    target_oa=expected_oa,
                    confidence_level=confidence_level,
                    min_sample_size=30,
                    max_sample_size=max(1000, int(total_samples * 2)),
                    num_points=50,
                )

                # Calculate current MOE for the calculated sample size
                current_moe = calculate_current_moe(
                    current_sample_size=total_samples,
                    target_oa=expected_oa,
                    confidence_level=confidence_level,
                )

            # Create samples per class list with class names
            samples_per_class = []
            for class_code, sample_count in allocation_dict.items():
                # Find class name from area data
                class_row = area_data[area_data["map_code"] == class_code]
                class_name = (
                    class_row["map_edited_class"].iloc[0]
                    if not class_row.empty
                    else ms.common.class_code.format(class_code)
                )

                samples_per_class.append(
                    {
                        "map_code": class_code,
                        "class_name": class_name,
                        "samples": int(sample_count),
                    }
                )

            results = {
                "target_error": app_state.target_error.value,
                "confidence_level": app_state.confidence_level.value,
                "total_samples": total_samples,
                "sampling_method": sampling_method,
                "allocation_method": (
                    allocation_method if sampling_method == "stratified" else None
                ),
                "samples_per_class": samples_per_class,
                "allocation_dict": allocation_dict,
                "precision_curve": (
                    precision_curve_df.to_dict("records")
                    if precision_curve_df is not None
                    else None
                ),
                "current_moe_percent": (
                    current_moe * 100 if current_moe is not None else None
                ),
                "current_moe_decimal": current_moe,
            }

            app_state.set_sample_results(results)
            app_state.set_processing_status("")

        except Exception as e:
            app_state.add_error(ms.design.error.calculating.format(e))
            app_state.set_processing_status("")

    with solara.Card(ms.calculator.title):
        solara.Text(ms.calculator.card_intro)

        if app_state.area_data.value is None or app_state.area_data.value.empty:
            solara.Info(ms.calculator.upload_first)
            return

        # Parameter update handlers
        def update_target_error(value):
            app_state.set_sampling_parameters(value, app_state.confidence_level.value)

        def update_confidence_level(value):
            app_state.set_sampling_parameters(app_state.target_error.value, value)

        def update_min_samples(value):
            app_state.set_sampling_parameters(
                app_state.target_error.value, app_state.confidence_level.value, value
            )

        def update_expected_accuracy(value):
            app_state.set_sampling_parameters(
                app_state.target_error.value,
                app_state.confidence_level.value,
                app_state.min_samples_per_class.value,
                value,
            )

        with solara.Row():
            solara.SliderFloat(
                ms.calculator.target_moe,
                value=app_state.target_error.value,
                min=1.0,
                max=10.0,
                step=0.5,
                on_value=update_target_error,
            )

        with solara.Row():
            solara.Select(
                label=ms.design.parameters.confidence_level,
                value=app_state.confidence_level.value,
                values=[90.0, 95.0, 99.0],
                on_value=update_confidence_level,
            )

        with solara.Row():
            solara.SliderFloat(
                ms.design.parameters.expected_accuracy,
                value=app_state.expected_accuracy.value,
                min=50.0,
                max=99.0,
                step=1.0,
                on_value=update_expected_accuracy,
            )

        with solara.Row():
            solara.SliderInt(
                ms.design.parameters.min_samples,
                value=app_state.min_samples_per_class.value,
                min=1,
                max=20,
                step=1,
                on_value=update_min_samples,
            )

        with solara.Row():
            solara.Button(
                ms.calculator.title,
                on_click=handle_calculate_samples,
                color="primary",
                outlined=True,
            )

        if app_state.sample_results.value:
            sample_results = app_state.sample_results.value
            sampling_method = sample_results.get("sampling_method", "stratified")

            with rv.Alert(type="info", text=True):
                confidence = f"{sample_results.get('confidence_level', 95):.0f}"
                if sampling_method in ("simple", "systematic"):
                    distribution = (
                        ms.calculator.distribution_random
                        if sampling_method == "simple"
                        else ms.calculator.distribution_systematic
                    )
                    with solara.Column(gap="4px"):
                        solara.Text(
                            ms.calculator.simple_results_title,
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.sampling_method.format(
                                sampling_method.capitalize()
                            ),
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.total_samples.format(
                                sample_results.get("total_samples", 0)
                            ),
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.confidence_level.format(confidence),
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.distribution_note.format(distribution),
                            style="margin-top: 8px;",
                        )
                else:
                    with solara.Column(gap="4px"):
                        solara.Text(
                            ms.calculator.stratified_results_title,
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.total_samples_needed.format(
                                sample_results.get("total_samples", 0)
                            ),
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.allocation_method.format(
                                sample_results.get(
                                    "allocation_method", ms.common.unknown
                                )
                            ),
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.current_moe.format(
                                f"{sample_results.get('current_moe_percent', 0):.2f}"
                            ),
                            style="font-weight: bold;",
                        )
                        solara.Text(
                            ms.calculator.samples_per_class,
                            style="font-weight: bold; margin-top: 8px;",
                        )

                        for class_info in sample_results.get("samples_per_class", []):
                            solara.Text(
                                ms.calculator.class_samples.format(
                                    class_info["class_name"], class_info["samples"]
                                )
                            )


def sample_allocation_table(ms=None) -> None:
    """Display sample allocation with manual editing - self-contained with its own logic."""
    ms = ms if ms is not None else get_translator()
    if not app_state.sample_results.value:
        return

    allocation_data = app_state.get_allocation_data()

    with solara.Card(ms.calculator.allocation.title):
        if not allocation_data:
            solara.Warning(ms.calculator.allocation.empty)
            return

        solara.Text(
            ms.calculator.allocation.manual_title,
            style="font-weight: bold; margin-bottom: 12px;",
        )

        for item in allocation_data:
            with solara.Columns([6, 6]):
                solara.Text(f"{item['class_name']}:")

                def make_update_callback(code):
                    def update_samples(samples):
                        app_state.update_manual_allocation(code, int(samples))

                    return update_samples

                solara.InputInt(
                    label=ms.calculator.allocation.samples,
                    value=item["samples"],
                    on_value=make_update_callback(item["map_code"]),
                )


def per_class_precision_chart(theme_state=None, ms=None):
    """Display per-class precision (MOE) given current allocation."""
    ms = ms if ms is not None else get_translator()
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

    per_class = ms.calculator.per_class
    with solara.Card(per_class.title):
        with solara.Column(gap="8px", style="margin-bottom: 12px;"):
            solara.Text(per_class.subtitle, style="font-weight: bold;")
            solara.Text(per_class.description)
            solara.Text(
                per_class.formula_title,
                style="font-weight: bold; margin-top: 8px;",
            )
            solara.Markdown(_PER_CLASS_MOE_FORMULA)
            with solara.Column(gap="2px", style="font-size: 0.9em; margin-top: 8px;"):
                solara.Text(per_class.where)
                solara.Text(per_class.term_n)
                solara.Text(per_class.term_p)
                solara.Text(per_class.term_z)
                solara.Text(per_class.term_population)
                solara.Text(per_class.term_deff)

        with rv.Alert(type="info", text=True, style="margin-bottom: 16px;"):
            max_moe_row = moe_df.iloc[0]
            min_moe_row = moe_df.iloc[-1]

            with solara.Column(gap="4px"):
                solara.Text(per_class.analysis_title, style="font-weight: bold;")
                solara.Text(
                    per_class.highest_moe.format(
                        max_moe_row["class_name"],
                        f"{max_moe_row['moe_percent']:.2f}",
                        max_moe_row["samples"],
                    ),
                    style="font-weight: bold;",
                )
                solara.Text(
                    per_class.lowest_moe.format(
                        min_moe_row["class_name"],
                        f"{min_moe_row['moe_percent']:.2f}",
                        min_moe_row["samples"],
                    ),
                    style="font-weight: bold;",
                )
                solara.Text(
                    ms.calculator.confidence_level.format(
                        f"{sample_results.get('confidence_level', 95):.0f}"
                    ),
                    style="font-weight: bold;",
                )
                solara.Text(per_class.advice, style="margin-top: 8px;")

        class_names = moe_df["class_name"].tolist()
        moe_values = moe_df["moe_percent"].tolist()
        sample_counts = moe_df["samples"].tolist()

        bar_colors = [
            "#ee6666" if moe > 15 else "#fac858" if moe > 10 else "#91cc75"
            for moe in moe_values
        ]

        bar_series = Bar(
            name=per_class.moe_axis,
            data=[
                {
                    "value": round(moe, 2),
                    "itemStyle": {"color": color},
                    "samples": n,
                }
                for moe, color, n in zip(moe_values, bar_colors, sample_counts)
            ],
            label={
                "show": True,
                "position": "right",
                "formatter": "{c}%",
                "fontSize": 11,
            },
        )

        option = Option(
            xAxis=XAxis(
                type="value",
                name=per_class.moe_axis,
                nameLocation="middle",
                nameGap=35,
                nameTextStyle={"fontSize": 14},
            ),
            yAxis=YAxis(
                type="category",
                data=class_names,
                nameTextStyle={"fontSize": 14},
                axisLabel={"fontSize": 11},
            ),
            series=[bar_series],
            tooltip=Tooltip(
                trigger="axis",
                axisPointer={"type": "shadow"},
                formatter="{b}: ±{c}%",
            ),
            grid=Grid(left="25%", right="15%", top="5%", bottom="15%"),
        )

        EChartsWidget.element(
            option=option,
            style={"height": "500px", "width": "100%"},
            theme_state=theme_state,
        )

        solara.Info(per_class.interpretation)


def precision_curve_info(theme_state=None, ms=None) -> None:
    """Display precision curve information showing MOE vs sample size relationship."""
    ms = ms if ms is not None else get_translator()
    sample_results = app_state.sample_results.value
    if not sample_results:
        return

    precision_curve = sample_results.get("precision_curve", [])
    if not precision_curve:
        return

    curve = ms.calculator.curve
    with solara.Card(curve.title):
        with solara.Column(gap="8px", style="margin-bottom: 12px;"):
            solara.Text(curve.subtitle, style="font-weight: bold;")
            solara.Text(curve.description)
            solara.Markdown(_PRECISION_CURVE_FORMULA)
            with solara.Column(gap="2px", style="font-size: 0.9em; margin-top: 8px;"):
                solara.Text(curve.where)
                solara.Text(curve.term_z)
                solara.Text(curve.term_oa)
                solara.Text(curve.term_n)

        with rv.Alert(type="success", text=True):
            current_total = sample_results.get("total_samples", 0)
            current_moe = sample_results.get("current_moe_percent", 0)

            with solara.Column(gap="4px"):
                solara.Text(curve.current_title, style="font-weight: bold;")
                solara.Text(
                    curve.current_size.format(current_total),
                    style="font-weight: bold;",
                )
                solara.Text(
                    curve.current_moe.format(f"{current_moe:.2f}"),
                    style="font-weight: bold;",
                )
                solara.Text(
                    ms.calculator.confidence_level.format(
                        f"{sample_results.get('confidence_level', 95):.0f}"
                    ),
                    style="font-weight: bold;",
                )

        # Extract data from precision curve
        sample_sizes = [point["sample_size"] for point in precision_curve]
        moe_percents = [round(point["moe_percent"], 2) for point in precision_curve]

        # Create line series for the precision curve
        current_series_name = curve.current_series.format(current_total)
        curve_line = Line(
            name=curve.series,
            data=[[x, y] for x, y in zip(sample_sizes, moe_percents)],
            smooth=True,
            lineStyle={"color": "#5470c6", "width": 3},
            itemStyle={"color": "#5470c6"},
            areaStyle={"color": "rgba(84, 112, 198, 0.2)"},
        )

        # Create scatter series for current design point
        current_point = Line(
            name=current_series_name,
            data=[[current_total, round(current_moe, 2)]],
            type="scatter",
            symbolSize=15,
            itemStyle={"color": "#ee6666", "borderColor": "#fff", "borderWidth": 2},
        )

        # Create the option
        option = Option(
            xAxis=XAxis(
                type="value",
                name=curve.sample_size_axis,
                nameLocation="middle",
                nameGap=35,
                nameTextStyle={"fontSize": 14},
            ),
            yAxis=YAxis(
                type="value",
                name=curve.moe_axis,
                nameLocation="middle",
                nameGap=50,
                nameTextStyle={"fontSize": 14},
            ),
            series=[curve_line, current_point],
            tooltip=Tooltip(trigger="axis", axisPointer={"type": "cross"}),
            # The legend selects series by name, so these must stay identical
            # to the series names above.
            legend=Legend(data=[curve.series, current_series_name], top="5%"),
            grid=Grid(left="15%", right="10%", top="15%", bottom="15%"),
        )

        # Create and display the chart
        EChartsWidget.element(
            option=option,
            style={"height": "500px", "width": "100%"},
            theme_state=theme_state,
        )

        solara.Info(curve.insight)
