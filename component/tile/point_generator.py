import solara
from solara.alias import rv

from component.model import app_state
from component.scripts.geospatial import generate_sample_points
from component.widget.map import SbaeMap


@solara.component
def PointGenerationTile(sbae_map: SbaeMap):
    """Step 4: Generate Sample Points Dialog."""
    with solara.Column():
        solara.Markdown("## 📍 Generate Sample Points")
        solara.Markdown(
            """
            Generate stratified random sample points based on your calculated
            sample allocation. Points will be distributed across map classes.
            
            **Process:**
            1. Points are randomly distributed within each class
            2. Each point includes coordinates and class information
            3. Points are displayed on the map for visualization
            """
        )

        point_generation_section()

        # Update map with generated points
        if (
            app_state.sample_points.value is not None
            and not app_state.sample_points.value.empty
        ):
            sbae_map.add_sample_points(app_state.sample_points.value)
            points_count = len(app_state.sample_points.value)
            solara.Success(
                f"✅ Generated {points_count:,} sample points! Check the map and export when ready."
            )


def point_generation_section() -> None:
    """Point generation component - self-contained with its own logic."""

    def handle_generate_points():
        """Handle sample point generation."""
        if not app_state.is_ready_for_point_generation():
            app_state.add_error("Please complete sample size calculation first.")
            return

        try:
            app_state.set_processing_status("Generating sample points...")
            app_state.set_points_status("Generating random sample points...")

            # Generate points
            points_df = generate_sample_points(
                file_path=app_state.file_path.value,
                samples_per_class=app_state.samples_per_class.value,
                class_lookup=app_state.get_class_lookup(),
            )

            app_state.set_sample_points(points_df)
            app_state.set_points_status(
                f"Successfully generated {len(points_df)} sample points"
            )
            app_state.set_processing_status("")

        except Exception as e:
            app_state.add_error(f"Error generating points: {str(e)}")
            app_state.set_processing_status("")
            app_state.set_points_status(None)

    with solara.Card("📍 Step 4: Generate Sample Points"):
        solara.Markdown(
            """
        Generate random sample points based on your calculated sample allocation.
        Points will be stratified by class according to your sampling design.
        """
        )

        with solara.Row():
            solara.Button(
                "Generate Sample Points",
                on_click=handle_generate_points,
                color="success",
                outlined=True,
            )

        if app_state.points_generation_status.value:
            with rv.Alert(type="info", text=True):
                solara.Markdown(app_state.points_generation_status.value)

        if (
            app_state.sample_points.value is not None
            and not app_state.sample_points.value.empty
        ):
            sample_points = app_state.sample_points.value
            with rv.Alert(type="success", text=True):
                solara.Markdown(
                    f"""
                **Points generated successfully!**
                - Total points: {len(sample_points)}
                - Classes represented: {sample_points["map_code"].nunique()}
                """
                )

            # Show sample of points
            solara.Markdown("**Sample of generated points:**")
            display_points = sample_points.head(10)
            solara.DataFrame(display_points, items_per_page=5)
