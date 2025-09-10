import pandas as pd
import solara

from component.model import app_state


@solara.component
def ClassEditorTile():
    """Step 2: Edit Class Names Dialog."""
    with solara.Column():
        solara.Markdown(
            """
            Review and customize the class names for your map. These names will
            appear in the sampling results and exported files.
            """
        )

        if (
            app_state.area_data.value is not None
            and not app_state.area_data.value.empty
        ):
            area_table_display(app_state.area_data.value)
            class_editor_section()

            solara.Success("✅ Class names updated! Proceed to calculate sample sizes.")
        else:
            solara.Warning("⚠️ Please upload a classification map first in Step 1.")


def class_editor_section() -> None:
    """Section for editing class names - self-contained with its own logic."""
    with solara.Card("Edit Class Names"):
        solara.Markdown(
            """
        Review and edit the class names for your map. These names will be used
        in the sampling results and exports.

        **Tips:**
        
            - Use descriptive names that will be clear during field work
            - Avoid special characters that might cause export issues
            - Consider abbreviations if names are very long
        """
        )

        if app_state.area_data.value.empty:
            solara.Info("Upload a classification map first to edit class names.")
            return

        for idx, row in app_state.area_data.value.iterrows():
            with solara.Row():
                with solara.Columns([3, 9]):
                    solara.Text(f"Class {row['map_code']}:")

                    current_name = row.get(
                        "map_edited_class", f"Class {row['map_code']}"
                    )

                    def make_update_callback(code):
                        def update_name(name):
                            app_state.update_class_name(code, name)

                        return update_name

                    solara.InputText(
                        label="",
                        value=current_name,
                        on_value=make_update_callback(row["map_code"]),
                    )


def area_table_display(area_data: pd.DataFrame) -> None:
    """Display area data in a formatted table.

    Args:
        area_data: DataFrame with area information
    """
    with solara.Card("Map Areas by Class"):
        if area_data.empty:
            solara.Warning("No area data available")
            return

        # Format areas for display
        display_data = area_data.copy()
        if "map_area" in display_data.columns:
            display_data["Area (hectares)"] = (display_data["map_area"] / 10000).round(
                2
            )
            display_data["Area (%)"] = (
                100 * display_data["map_area"] / display_data["map_area"].sum()
            ).round(1)

        # Select columns to display
        display_columns = [
            "map_code",
            "map_edited_class",
            "Area (hectares)",
            "Area (%)",
        ]
        available_columns = [
            col for col in display_columns if col in display_data.columns
        ]

        solara.DataFrame(display_data[available_columns], items_per_page=15)
