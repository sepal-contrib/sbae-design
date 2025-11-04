import solara

from component.model import app_state


@solara.component
def Export():
    """Export options component - self-contained with its own logic."""
    csv_data = app_state.export_csv()
    geojson_data = app_state.export_geojson()

    with solara.Row():
        if app_state.sample_points.value is None or app_state.sample_points.value.empty:
            solara.Info("Generate sample points first to enable export.")
            return
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
