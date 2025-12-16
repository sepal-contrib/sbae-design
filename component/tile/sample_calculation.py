import solara
from ipecharts.option import Grid, Legend, Option, Tooltip, XAxis, YAxis
from ipecharts.option.series import Bar, Line
from solara.alias import rv

from component.model import app_state
from component.scripts.calculations import (
    calculate_current_moe,
    calculate_per_class_moe_for_allocation,
    calculate_precision_curve,
    calculate_sample_design,
)
from component.widget.echarts import EChartsWidget


@solara.component
def SampleCalculationTile(theme_toggle=None):
    """Step 3: Calculate Sample Size Dialog."""
    with solara.Column():
        solara.Markdown("##Calculate Sample Size")
        solara.Markdown(
            """
            Configure sampling parameters and calculate the required sample size
            for your accuracy assessment based on statistical methods.
            
            **Parameters:**
            - **Target Overall Accuracy**: Desired accuracy level (e.g., 85%)
            - **Confidence Level**: Statistical confidence (typically 95%)
            - **Minimum Sample Size**: Safety minimum for small classes
            """
        )

        sample_size_calculator()

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
                sample_allocation_table()
                if app_state.sample_results.value.get("precision_curve"):
                    per_class_precision_chart(theme_toggle=theme_toggle)

            solara.Success("✅ Sample configuration complete! Ready to generate points.")

        # Display precision curve for all methods
        if app_state.sample_results.value and app_state.sample_results.value.get(
            "precision_curve"
        ):
            precision_curve_info(theme_toggle=theme_toggle)


def sample_size_calculator() -> None:
    """Sample size calculation component."""

    def handle_calculate_samples():
        """Handle sample size calculation."""
        if not app_state.is_ready_for_calculation():
            app_state.add_error(">>>>Please upload a classification map first.")
            return

        try:
            app_state.set_processing_status("Calculating sample sizes...")

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
                    else f"Class {class_code}"
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
            app_state.add_error(f"Error calculating samples: {str(e)}")
            app_state.set_processing_status("")

    with solara.Card("Calculate Sample Size"):
        solara.Markdown(
            """
        Configure sampling parameters to determine the required sample size
        for your accuracy assessment.
        """
        )

        if app_state.area_data.value is None or app_state.area_data.value.empty:
            solara.Info("Upload and configure your map first.")
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
                "Target Margin of Error (%)",
                value=app_state.target_error.value,
                min=1.0,
                max=10.0,
                step=0.5,
                on_value=update_target_error,
            )

        with solara.Row():
            solara.Select(
                label="Confidence Level",
                value=app_state.confidence_level.value,
                values=[90.0, 95.0, 99.0],
                on_value=update_confidence_level,
            )

        with solara.Row():
            solara.SliderFloat(
                "Expected Overall Accuracy (%)",
                value=app_state.expected_accuracy.value,
                min=50.0,
                max=99.0,
                step=1.0,
                on_value=update_expected_accuracy,
            )

        with solara.Row():
            solara.SliderInt(
                "Minimum Samples per Class",
                value=app_state.min_samples_per_class.value,
                min=1,
                max=20,
                step=1,
                on_value=update_min_samples,
            )

        with solara.Row():
            solara.Button(
                "Calculate Sample Size",
                on_click=handle_calculate_samples,
                color="primary",
                outlined=True,
            )

        if app_state.sample_results.value:
            sample_results = app_state.sample_results.value
            sampling_method = sample_results.get("sampling_method", "stratified")

            with rv.Alert(type="info", text=True):
                # Different summaries based on sampling method
                if sampling_method in ("simple", "systematic"):
                    # Simple/Systematic: no per-class allocation
                    solara.Markdown(
                        f"""
                    **Sample Size Configuration:**
                    - Sampling method: **{sampling_method.capitalize()}**
                    - Total samples: **{sample_results.get("total_samples", 0)}**
                    - Confidence Level: **{sample_results.get("confidence_level", 95):.0f}%**
                    
                    📍 Samples will be distributed {'randomly' if sampling_method == 'simple' else 'in a systematic grid'} across the entire study area.
                    """
                    )
                else:
                    # Stratified: show per-class allocation
                    solara.Markdown(
                        f"""
                    **Sample Size Calculation Results:**
                    - Total samples needed: **{sample_results.get("total_samples", 0)}**
                    - Allocation method: **{sample_results.get("allocation_method", "Unknown")}**
                    - Current Margin of Error: **{sample_results.get("current_moe_percent", 0):.2f}%**

                    **Samples per class:**
                    """
                    )

                    for class_info in sample_results.get("samples_per_class", []):
                        solara.Markdown(
                            f"- {class_info['class_name']}: {class_info['samples']}"
                        )


def sample_allocation_table() -> None:
    """Display sample allocation with manual editing - self-contained with its own logic."""
    if not app_state.sample_results.value:
        return

    allocation_data = app_state.get_allocation_data()

    with solara.Card("📋 Sample Allocation"):
        if not allocation_data:
            solara.Warning("No allocation data available")
            return

        solara.Markdown("**Manual Allocation Editing:**")

        for item in allocation_data:
            with solara.Columns([6, 6]):
                solara.Text(f"{item['class_name']}:")

                def make_update_callback(code):
                    def update_samples(samples):
                        app_state.update_manual_allocation(code, int(samples))

                    return update_samples

                solara.InputInt(
                    label="Samples",
                    value=item["samples"],
                    on_value=make_update_callback(item["map_code"]),
                )


def per_class_precision_chart(theme_toggle=None):
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

    with solara.Card("📊 Per-Class Precision (Given Current Allocation)"):
        solara.Markdown(
            """
            **Per-Class Margin of Error (MOE)**
            
            This chart shows the expected precision for each class based on your current sample allocation.
            Classes with larger MOE bars are under-powered and have less precise estimates.
            
            **Formula (binomial normal approximation):**
            
            $$MOE_h = Z \\times \\sqrt{DEFF \\times \\frac{p_h(1-p_h)}{n_h}} \\times \\sqrt{\\frac{N_h - n_h}{N_h - 1}} \\times 100$$
            
            Where:
            - **$n_h$**: samples allocated to the class
            - **$p_h$**: expected accuracy (0.5 used as conservative default)
            - **$Z$**: Z-score based on confidence level
            - **$N_h$**: finite population size (optional, using 1 if unknown)
            - **$DEFF$**: design effect (≈1.0 for spatially balanced; >1 if clustered)
            """
        )

        with rv.Alert(type="info", text=True, style="margin-bottom: 16px;"):
            max_moe_row = moe_df.iloc[0]
            min_moe_row = moe_df.iloc[-1]

            solara.Markdown(
                f"""
            **Current Allocation Analysis:**
            - Highest MOE: **{max_moe_row['class_name']}** (±{max_moe_row['moe_percent']:.2f}%, n={max_moe_row['samples']})
            - Lowest MOE: **{min_moe_row['class_name']}** (±{min_moe_row['moe_percent']:.2f}%, n={min_moe_row['samples']})
            - Confidence Level: **{sample_results.get('confidence_level', 95):.0f}%**
            
            💡 Classes with large MOE values may need more samples for better precision.
            """
            )

        class_names = moe_df["class_name"].tolist()
        moe_values = moe_df["moe_percent"].tolist()
        sample_counts = moe_df["samples"].tolist()

        bar_colors = [
            "#ee6666" if moe > 15 else "#fac858" if moe > 10 else "#91cc75"
            for moe in moe_values
        ]

        bar_series = Bar(
            name="Margin of Error (%)",
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
                name="Margin of Error (%)",
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
            theme_toggle=theme_toggle,
        )

        solara.Info(
            """
            💡 **Interpretation**: The bars show the margin of error for each class's 
            user accuracy estimate. Larger bars indicate less precise estimates. 
            If you oversample rare classes, this chart helps verify you achieved 
            the desired per-class precision.
            """
        )


def precision_curve_info(theme_toggle=None) -> None:
    """Display precision curve information showing MOE vs sample size relationship."""
    sample_results = app_state.sample_results.value
    if not sample_results:
        return

    precision_curve = sample_results.get("precision_curve", [])
    if not precision_curve:
        return

    with solara.Card("📊 Precision Curve Analysis"):
        solara.Markdown(
            """
            **How Margin of Error (MOE) Changes with Sample Size**
            
            The precision curve shows the inverse relationship between sample size 
            and margin of error. As you increase the total sample size (n), the 
            margin of error decreases following the formula:
            
            $$MOE = Z \\times \\sqrt{\\frac{OA \\times (1 - OA)}{n}}$$
            
            Where:
            - **Z** = Z-score based on confidence level
            - **OA** = Overall accuracy (expected)
            - **n** = Total sample size
            """
        )

        with rv.Alert(type="success", text=True):
            current_total = sample_results.get("total_samples", 0)
            current_moe = sample_results.get("current_moe_percent", 0)

            solara.Markdown(
                f"""
            **Your Current Design:**
            - Sample size: **{current_total}**
            - Margin of Error: **±{current_moe:.2f}%**
            - Confidence Level: **{sample_results.get("confidence_level", 95):.0f}%**
            """
            )

        # Extract data from precision curve
        sample_sizes = [point["sample_size"] for point in precision_curve]
        moe_percents = [round(point["moe_percent"], 2) for point in precision_curve]

        # Create line series for the precision curve
        curve_line = Line(
            name="MOE vs Sample Size",
            data=[[x, y] for x, y in zip(sample_sizes, moe_percents)],
            smooth=True,
            lineStyle={"color": "#5470c6", "width": 3},
            itemStyle={"color": "#5470c6"},
            areaStyle={"color": "rgba(84, 112, 198, 0.2)"},
        )

        # Create scatter series for current design point
        current_point = Line(
            name=f"Your Design (n={current_total})",
            data=[[current_total, round(current_moe, 2)]],
            type="scatter",
            symbolSize=15,
            itemStyle={"color": "#ee6666", "borderColor": "#fff", "borderWidth": 2},
        )

        # Create the option
        option = Option(
            xAxis=XAxis(
                type="value",
                name="Sample Size (n)",
                nameLocation="middle",
                nameGap=35,
                nameTextStyle={"fontSize": 14},
            ),
            yAxis=YAxis(
                type="value",
                name="Margin of Error (%)",
                nameLocation="middle",
                nameGap=50,
                nameTextStyle={"fontSize": 14},
            ),
            series=[curve_line, current_point],
            tooltip=Tooltip(trigger="axis", axisPointer={"type": "cross"}),
            legend=Legend(
                data=["MOE vs Sample Size", f"Your Design (n={current_total})"],
                top="5%",
            ),
            grid=Grid(left="15%", right="10%", top="15%", bottom="15%"),
        )

        # Create and display the chart
        EChartsWidget.element(
            option=option,
            style={"height": "500px", "width": "100%"},
            theme_toggle=theme_toggle,
        )

        solara.Info(
            """
            💡 **Key Insight**: Notice how the MOE decreases rapidly at first, 
            but the improvement slows as sample size increases. This is the 
            "diminishing returns" effect - doubling the sample size doesn't 
            halve the error.
            """
        )
