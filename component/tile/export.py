import solara

from component.model import app_state


@solara.component
def ExportTile():
    """Step 5: Export Results Dialog."""
    with solara.Column():
        solara.Markdown("##Export Sample Points")
        solara.Markdown(
            """
            Download your sample points for field work or further analysis.
            Available in CSV and GeoJSON formats for use in GIS software.
            
            **Export Formats:**
            - **CSV**: Simple table format with coordinates and class information
            - **GeoJSON**: Spatial format for direct import into GIS applications
            
            **Field Work Tips:**
            - Load points into a mobile GIS app for navigation
            - Include the class information for validation reference
            - Consider downloading offline basemaps for remote areas
            """
        )

        export_section()


def export_section() -> None:
    """Export options component - self-contained with its own logic."""
    with solara.Card("Export Sample Points"):
        solara.Markdown(
            """
        Download your sample points for field work or further analysis.
        Available formats:
        - **CSV**: Tabular format with coordinates
        - **GeoJSON**: Spatial format for GIS software
        """
        )

        if app_state.sample_points.value is None or app_state.sample_points.value.empty:
            solara.Info("Generate sample points first to enable export.")
            return

        csv_data = app_state.export_csv()
        geojson_data = app_state.export_geojson()

        with solara.Columns([6, 6]):
            with solara.FileDownload(
                data=csv_data.encode(),
                filename="sample_points.csv",
                mime_type="text/csv",
            ):
                solara.Button(
                    "Download CSV",
                    icon_name="mdi-cloud-download-outline",
                    color="primary",
                    outlined=True,
                )

            with solara.FileDownload(
                data=geojson_data.encode(),
                filename="sample_points.geojson",
                mime_type="application/geo+json",
            ):
                solara.Button(
                    "Download GeoJSON",
                    icon_name="mdi-cloud-download-outline",
                    color="success",
                    outlined=True,
                )
