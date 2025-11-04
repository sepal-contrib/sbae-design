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
    show_csv_download, set_show_csv_download = solara.use_state(False)
    show_geojson_download, set_show_geojson_download = solara.use_state(False)

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

        with solara.Columns([6, 6]):
            solara.Button(
                "Download CSV",
                on_click=lambda: set_show_csv_download(True),
                color="primary",
                outlined=True,
            )

            solara.Button(
                "Download GeoJSON",
                on_click=lambda: set_show_geojson_download(True),
                color="success",
                outlined=True,
            )

        if show_csv_download:
            csv_data = app_state.export_csv()
            solara.FileDownload(
                data=csv_data.encode(),
                filename="sample_points.csv",
                mime_type="text/csv",
            )
            set_show_csv_download(False)

        if show_geojson_download:
            geojson_data = app_state.export_geojson()
            solara.FileDownload(
                data=geojson_data.encode(),
                filename="sample_points.geojson",
                mime_type="application/geo+json",
            )
            set_show_geojson_download(False)
