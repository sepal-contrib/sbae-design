"""SBAE State Management Module.

Contains reactive state management for the SBAE application.
"""

import json
import os
import shutil
import tempfile
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import solara


class AppState:
    """Centralized state management for SBAE application using Solara reactive variables."""

    def __init__(self):
        # File handling
        self.uploaded_file_info = solara.reactive(None)
        self.temp_dir = solara.reactive(tempfile.mkdtemp())
        self.file_path = solara.reactive(None)
        self.file_error = solara.reactive(None)

        # Area data
        self.area_data = solara.reactive(pd.DataFrame())
        self.original_area_data = solara.reactive(pd.DataFrame())

        # Sample calculation parameters
        self.target_error = solara.reactive(5.0)
        self.confidence_level = solara.reactive(95.0)

        # Sample results
        self.sample_results = solara.reactive(None)
        self.samples_per_class = solara.reactive({})
        self.allocation_method = solara.reactive("proportional")

        # Generated points
        self.sample_points = solara.reactive(pd.DataFrame())
        self.points_generation_status = solara.reactive(None)

        # UI state
        self.current_step = solara.reactive(1)
        self.processing_status = solara.reactive("")
        self.error_messages = solara.reactive([])

        # Export state
        self.last_export_csv = solara.reactive("")
        self.last_export_geojson = solara.reactive("")

    def reset_state(self):
        """Reset all state to initial values."""
        self.uploaded_file_info.value = None
        self.file_path.value = None
        self.file_error.value = None
        self.area_data.value = pd.DataFrame()
        self.original_area_data.value = pd.DataFrame()
        self.sample_results.value = None
        self.samples_per_class.value = {}
        self.sample_points.value = pd.DataFrame()
        self.points_generation_status.value = None
        self.current_step.value = 1
        self.processing_status.value = ""
        self.error_messages.value = []
        self.last_export_csv.value = ""
        self.last_export_geojson.value = ""

        # Clean up temp directory
        if self.temp_dir.value and os.path.exists(self.temp_dir.value):

            try:
                shutil.rmtree(self.temp_dir.value)
            except (OSError, PermissionError):
                pass
        self.temp_dir.value = tempfile.mkdtemp()

    def set_file_info(self, file_info: Dict, file_path: str):
        """Set uploaded file information."""
        self.uploaded_file_info.value = file_info
        self.file_path.value = file_path
        self.file_error.value = None
        self.current_step.value = max(self.current_step.value, 2)

    def set_file_error(self, error: str):
        """Set file processing error."""
        self.file_error.value = error
        self.uploaded_file_info.value = None
        self.file_path.value = None

    def set_area_data(self, area_df: pd.DataFrame):
        """Set area data from file processing."""
        self.area_data.value = area_df.copy()
        self.original_area_data.value = area_df.copy()
        self.current_step.value = max(self.current_step.value, 2)

    def update_class_name(self, map_code: int, new_name: str):
        """Update class name in area data."""
        if not self.area_data.value.empty:
            area_df = self.area_data.value.copy()
            mask = area_df["map_code"] == map_code
            if mask.any():
                area_df.loc[mask, "map_edited_class"] = new_name
                self.area_data.value = area_df

    def set_sampling_parameters(self, target_error: float, confidence_level: float):
        """Update sampling parameters."""
        self.target_error.value = target_error
        self.confidence_level.value = confidence_level

    def set_sample_results(self, results: Dict):
        """Set sample calculation results."""
        self.sample_results.value = results

        # Extract samples per class for easy access
        samples_dict = {}
        for class_info in results.get("samples_per_class", []):
            samples_dict[class_info["map_code"]] = class_info["samples"]
        self.samples_per_class.value = samples_dict

        self.allocation_method.value = results.get("allocation_method", "proportional")
        self.current_step.value = max(self.current_step.value, 4)

    def update_manual_allocation(self, map_code: int, samples: int):
        """Update manual sample allocation."""
        current_samples = self.samples_per_class.value.copy()
        current_samples[map_code] = samples
        self.samples_per_class.value = current_samples

        # Update the results object too
        if self.sample_results.value:
            results = self.sample_results.value.copy()
            for class_info in results.get("samples_per_class", []):
                if class_info["map_code"] == map_code:
                    class_info["samples"] = samples

            # Recalculate total
            total_samples = sum(samples for samples in current_samples.values())
            results["total_samples"] = total_samples
            results["allocation_method"] = "manual"

            self.sample_results.value = results
            self.allocation_method.value = "manual"

    def set_sample_points(self, points_df: pd.DataFrame):
        """Set generated sample points."""
        self.sample_points.value = points_df
        self.current_step.value = max(self.current_step.value, 5)

    def set_points_status(self, status: str):
        """Set point generation status message."""
        self.points_generation_status.value = status

    def add_error(self, error_message: str):
        """Add error message to the list."""
        current_errors = self.error_messages.value.copy()
        current_errors.append(error_message)
        self.error_messages.value = current_errors

    def clear_errors(self):
        """Clear all error messages."""
        self.error_messages.value = []

    def set_processing_status(self, status: str):
        """Set current processing status."""
        self.processing_status.value = status

    def get_class_lookup(self) -> Dict[int, str]:
        """Get mapping of class codes to names."""
        if self.area_data.value.empty:
            return {}

        return dict(
            zip(
                self.area_data.value["map_code"],
                self.area_data.value["map_edited_class"],
            )
        )

    def get_allocation_data(self) -> List[Dict]:
        """Get allocation data for display."""
        if not self.sample_results.value or self.area_data.value.empty:
            return []

        allocation_data = []
        class_lookup = self.get_class_lookup()

        for map_code, samples in self.samples_per_class.value.items():
            # Get area for proportion calculation
            area_row = self.area_data.value[
                self.area_data.value["map_code"] == map_code
            ]
            area = area_row["map_area"].iloc[0] if not area_row.empty else 0
            total_area = self.area_data.value["map_area"].sum()
            proportion = area / total_area if total_area > 0 else 0

            allocation_data.append(
                {
                    "map_code": map_code,
                    "class_name": class_lookup.get(map_code, f"Class {map_code}"),
                    "samples": samples,
                    "proportion": proportion,
                    "area_hectares": area / 10000,  # Convert to hectares
                }
            )

        return allocation_data

    def is_ready_for_calculation(self) -> bool:
        """Check if ready for sample size calculation."""
        return (
            not self.area_data.value.empty
            and self.target_error.value > 0
            and self.confidence_level.value > 0
        )

    def is_ready_for_point_generation(self) -> bool:
        """Check if ready for point generation."""
        return (
            self.is_ready_for_calculation()
            and self.sample_results.value is not None
            and bool(self.samples_per_class.value)
            and self.file_path.value is not None
        )

    def is_ready_for_export(self) -> bool:
        """Check if ready for export."""
        return not self.sample_points.value.empty

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the project."""
        stats = {}

        if not self.area_data.value.empty:
            total_area = self.area_data.value["map_area"].sum()
            stats.update(
                {
                    "total_area_hectares": total_area / 10000,
                    "n_classes": len(self.area_data.value),
                    "classes": self.get_class_lookup(),
                }
            )

        if self.sample_results.value:
            stats.update(
                {
                    "target_error": self.target_error.value,
                    "confidence_level": self.confidence_level.value,
                    "total_samples": self.sample_results.value.get("total_samples", 0),
                    "allocation_method": self.allocation_method.value,
                }
            )

        if not self.sample_points.value.empty:
            points_per_class = (
                self.sample_points.value.groupby("map_code").size().to_dict()
            )
            stats.update(
                {
                    "total_points_generated": len(self.sample_points.value),
                    "points_per_class": points_per_class,
                }
            )

        return stats

    def export_csv(self) -> str:
        """Export sample points to CSV format."""
        if self.sample_points.value.empty:
            return ""

        csv_content = self.sample_points.value.to_csv(index=False)
        self.last_export_csv.value = csv_content
        return csv_content

    def export_geojson(self) -> str:
        """Export sample points to GeoJSON format."""
        if self.sample_points.value.empty:
            return ""

        try:

            gdf = gpd.GeoDataFrame(
                self.sample_points.value,
                geometry=gpd.points_from_xy(
                    self.sample_points.value.longitude,
                    self.sample_points.value.latitude,
                ),
                crs="EPSG:4326",
            )
            geojson_content = gdf.to_json()
            self.last_export_geojson.value = geojson_content
            return geojson_content

        except ImportError:
            # Fallback to simple GeoJSON structure
            features = []
            for _, row in self.sample_points.value.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["longitude"], row["latitude"]],
                    },
                    "properties": {
                        "map_code": row["map_code"],
                        "map_edited_class": row["map_edited_class"],
                    },
                }
                features.append(feature)

            geojson = {"type": "FeatureCollection", "features": features}

            geojson_content = json.dumps(geojson, indent=2)
            self.last_export_geojson.value = geojson_content
            return geojson_content


app_state = AppState()
