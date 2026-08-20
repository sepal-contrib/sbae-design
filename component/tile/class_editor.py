import solara

from component.message import use_translator
from component.model import app_state


@solara.component
def class_editor_table(show_sample_controls=True):
    """Editable table for class names, expected accuracies, and sample allocations."""
    ms = use_translator()
    editor = ms.design.class_editor
    area_data = app_state.area_data.value
    sample_results = app_state.sample_results.value
    eua_dict = app_state.expected_user_accuracies.value
    eua_modes = app_state.eua_modes.value
    class_colors = app_state.class_colors.value

    if area_data is None or area_data.empty:
        solara.Warning(editor.no_area_data)
        return

    samples_dict = {}

    if sample_results and "allocation_dict" in sample_results:
        samples_dict = sample_results["allocation_dict"]

    def update_high_eua(value):
        if value is not None and value != "":
            try:
                eua_decimal = float(value) / 100.0
                app_state.update_global_high_eua(eua_decimal)
            except (ValueError, TypeError):
                pass

    def update_low_eua(value):
        if value is not None and value != "":
            try:
                eua_decimal = float(value) / 100.0
                app_state.update_global_low_eua(eua_decimal)
            except (ValueError, TypeError):
                pass

    with solara.Column():

        with solara.Card(
            subtitle=editor.eua_card_title,
            style="margin-bottom: 16px; padding: 16px;",
        ):
            with solara.Row(gap="16px"):
                with solara.Column(style="flex: 1;"):
                    solara.v.TextField(
                        label=editor.high_eua,
                        v_model=app_state.high_eua.value * 100,
                        on_v_model=update_high_eua,
                        type="number",
                        min=30,
                        max=100,
                        step=1,
                        hint=editor.high_eua_hint,
                        style="width: 100%;",
                    )
                with solara.Column(style="flex: 1;"):
                    solara.v.TextField(
                        label=editor.low_eua,
                        v_model=app_state.low_eua.value * 100,
                        on_v_model=update_low_eua,
                        type="number",
                        min=30,
                        max=100,
                        step=1,
                        hint=editor.low_eua_hint,
                        style="width: 100%;",
                    )

        # Class table
        for idx, row in area_data.iterrows():
            map_code = row["map_code"]
            current_name = row.get(
                "map_edited_class", ms.common.class_code.format(map_code)
            )
            area_ha = row["map_area"] / 10000
            area_pct = 100 * row["map_area"] / area_data["map_area"].sum()
            samples = samples_dict.get(map_code, 0)
            eua_value = eua_dict.get(map_code, 0.85) * 100
            current_mode = eua_modes.get(map_code, "high")

            # Get color from extracted palette or use default
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
            color = class_colors.get(
                map_code, default_colors[idx % len(default_colors)]
            )

            def make_update_name_callback(code):
                def update_name(name):
                    app_state.update_class_name(code, name)

                return update_name

            def make_set_mode_callback(code, mode):
                def set_mode():
                    app_state.set_eua_mode(code, mode)

                return set_mode

            def make_update_custom_eua_callback(code):
                def update_eua(value):
                    if value is not None and value != "":
                        try:
                            eua_decimal = float(value) / 100.0
                            app_state.update_expected_accuracy(code, eua_decimal)
                        except (ValueError, TypeError):
                            pass

                return update_eua

            def make_update_samples_callback(code):
                def update_samples(count):
                    if count is not None and count != "":
                        try:
                            app_state.update_manual_allocation(code, int(float(count)))
                        except (ValueError, TypeError):
                            pass

                return update_samples

            with solara.Card(
                style=f"margin-bottom: 8px; padding: 12px; border-left: 8px solid {color};"
            ):
                with solara.Row(justify="space-between", style="align-items: center;"):
                    with solara.Column(style="flex: 0 0 50px;"):
                        solara.Text(
                            editor.code.format(map_code), style="font-weight: 500;"
                        )

                    with solara.Column(style="flex: 1 1 auto; margin: 0 8px;"):
                        solara.InputText(
                            label=editor.class_name,
                            value=current_name,
                            on_value=make_update_name_callback(map_code),
                            style="min-width: 120px;",
                        )

                    with solara.Column(style="flex: 0 0 200px; margin: 0 8px;"):
                        with solara.Row(gap="4px"):
                            solara.Button(
                                editor.mode_high,
                                on_click=make_set_mode_callback(map_code, "high"),
                                color="success" if current_mode == "high" else None,
                                outlined=current_mode != "high",
                                small=True,
                                style="min-width: 60px;",
                            )
                            solara.Button(
                                editor.mode_low,
                                on_click=make_set_mode_callback(map_code, "low"),
                                color="warning" if current_mode == "low" else None,
                                outlined=current_mode != "low",
                                small=True,
                                style="min-width: 60px;",
                            )
                            solara.Button(
                                editor.mode_custom,
                                on_click=make_set_mode_callback(map_code, "custom"),
                                color=("primary" if current_mode == "custom" else None),
                                outlined=current_mode != "custom",
                                small=True,
                                style="min-width: 60px;",
                            )

                    if current_mode == "custom":
                        with solara.Column(style="flex: 0 0 100px; margin: 0 8px;"):
                            solara.v.TextField(
                                label=editor.custom_eua,
                                v_model=eua_value,
                                on_v_model=make_update_custom_eua_callback(map_code),
                                type="number",
                                min=30,
                                max=100,
                                step=1,
                                dense=True,
                                style="width: 100%;",
                            )
                    else:
                        with solara.Column(
                            style="flex: 0 0 100px; margin: 0 8px; text-align: center;"
                        ):
                            solara.Text(
                                f"{eua_value:.0f}%",
                                style="font-size: 0.9em; line-height: 2.5;",
                            )

                    with solara.Column(
                        style="flex: 0 0 150px; text-align: right;", gap="4px"
                    ):
                        solara.Text(
                            editor.area_share.format(
                                f"{area_ha:,.2f}", f"{area_pct:.1f}"
                            ),
                            style="font-size: 0.9em;",
                        )
                        solara.HTML(
                            tag="div",
                            unsafe_innerHTML=f"""
                                <div style="width: 100%; background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 4px;">
                                    <div style="width: {area_pct:.1f}%; background: #90caf9; height: 100%; transition: width 0.3s ease;"></div>
                                </div>
                                """,
                        )

                    if samples_dict and show_sample_controls:
                        with solara.Column(style="flex: 0 0 110px;"):
                            solara.v.TextField(
                                label=editor.samples,
                                v_model=samples,
                                on_v_model=make_update_samples_callback(map_code),
                                type="number",
                                dense=True,
                                style="width: 100%;",
                            )
