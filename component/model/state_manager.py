"""SBAE State Management Module.

Contains reactive state management for the SBAE application.
"""

import json
import tempfile
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import solara

from component.scripts.precision import calculate_current_moe


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
        # AOI data for simple/systematic sampling (contains gdf and area)
        self.aoi_data = solara.reactive(None)
        # Computed GeoDataFrame from AOI feature_collection
        self.aoi_gdf = solara.reactive(None)
        # AOI computation status
        self.aoi_computing = solara.reactive(False)
        # Color palette extracted from raster or default
        self.class_colors = solara.reactive({})
        # Expected User's Accuracy per class (EUA)
        self.expected_user_accuracies = solara.reactive({})
        # Global high and low EUA values
        self.high_eua = solara.reactive(0.90)  # 90% for high confidence
        self.low_eua = solara.reactive(0.70)  # 70% for low confidence
        # EUA selection mode per class: 'high', 'low', or 'custom'
        self.eua_modes = solara.reactive({})  # {class_code: 'high'/'low'/'custom'}

        # Sample calculation parameters
        self.target_error = solara.reactive(1.0)
        self.confidence_level = solara.reactive(95.0)
        self.min_samples_per_class = solara.reactive(30)
        self.expected_accuracy = solara.reactive(85.0)
        # Sampling mode: 'stratified' (default), 'simple' (user-specified n), 'systematic'
        self.sampling_method = solara.reactive("stratified")
        # Allocation method for stratified sampling: 'proportional', 'equal', 'neyman', 'balanced'
        self.stratified_allocation_method = solara.reactive("proportional")
        # When sampling_method == 'simple' this value is used as total sample size
        self.simple_total_samples = solara.reactive(100)

        # Sample results
        self.sample_results = solara.reactive(None)
        self.samples_per_class = solara.reactive({})

        # Generated points
        self.sample_points = solara.reactive(pd.DataFrame())
        self.points_generation_status = solara.reactive(None)
        self.points_sampling_method = solara.reactive(
            None
        )  # Track method used for generated points

        # UI state
        self.current_step = solara.reactive(1)
        self.processing_status = solara.reactive("")
        self.error_messages = solara.reactive([])

        # Export state
        self.last_export_csv = solara.reactive("")
        self.last_export_geojson = solara.reactive("")

    def update_class_name(self, map_code: int, new_name: str):
        """Update class name in area data."""
        if self.area_data.value is not None and not self.area_data.value.empty:
            area_df = self.area_data.value.copy()
            mask = area_df["map_code"] == map_code
            if mask.any():
                area_df.loc[mask, "map_edited_class"] = new_name
                self.area_data.value = area_df

    def update_global_high_eua(self, value: float):
        """Update global high EUA value and apply to all 'high' classes.

        Args:
            value: High EUA value (0.0-1.0)
        """
        if value < 0.3 or value > 1.0:
            raise ValueError("High EUA must be between 0.3 and 1.0")

        self.high_eua.value = value

        # Update all classes that are set to 'high' mode
        eua_dict = self.expected_user_accuracies.value.copy()
        for code, mode in self.eua_modes.value.items():
            if mode == "high":
                eua_dict[code] = value
        self.expected_user_accuracies.value = eua_dict

    def update_global_low_eua(self, value: float):
        """Update global low EUA value and apply to all 'low' classes.

        Args:
            value: Low EUA value (0.0-1.0)
        """
        if value < 0.3 or value > 1.0:
            raise ValueError("Low EUA must be between 0.3 and 1.0")

        self.low_eua.value = value

        # Update all classes that are set to 'low' mode
        eua_dict = self.expected_user_accuracies.value.copy()
        for code, mode in self.eua_modes.value.items():
            if mode == "low":
                eua_dict[code] = value
        self.expected_user_accuracies.value = eua_dict

    def set_eua_mode(self, map_code: int, mode: str):
        """Set EUA mode for a class (high/low/custom).

        Args:
            map_code: Class code
            mode: 'high', 'low', or 'custom'
        """
        if mode not in ("high", "low", "custom"):
            raise ValueError("Mode must be 'high', 'low', or 'custom'")

        mode_dict = self.eua_modes.value.copy()
        mode_dict[map_code] = mode
        self.eua_modes.value = mode_dict

        # Update the actual EUA value based on mode
        eua_dict = self.expected_user_accuracies.value.copy()
        if mode == "high":
            eua_dict[map_code] = self.high_eua.value
        elif mode == "low":
            eua_dict[map_code] = self.low_eua.value
        # For 'custom', keep the current value
        self.expected_user_accuracies.value = eua_dict

    def update_expected_accuracy(self, map_code: int, eua_value: float):
        """Update expected user's accuracy for a class.

        Args:
            map_code: Class code
            eua_value: Expected User's Accuracy (0.0-1.0)
        """
        if eua_value < 0.3 or eua_value > 1.0:
            raise ValueError("Expected accuracy must be between 0.3 and 1.0")

        eua_dict = self.expected_user_accuracies.value.copy()
        eua_dict[map_code] = eua_value
        self.expected_user_accuracies.value = eua_dict

        # When manually updating, set mode to custom
        mode_dict = self.eua_modes.value.copy()
        mode_dict[map_code] = "custom"
        self.eua_modes.value = mode_dict

    def set_sampling_parameters(
        self,
        target_error: float,
        confidence_level: float,
        min_samples_per_class: int = None,
        expected_accuracy: float = None,
        sampling_method: str = None,
        simple_total_samples: int = None,
        stratified_allocation_method: str = None,
    ):
        """Update sampling parameters.

        Args:
            target_error: Target margin of error percentage (1-10)
            confidence_level: Confidence level percentage (90-99)
            min_samples_per_class: Minimum samples per class (optional, maintains current if not provided)
            expected_accuracy: Expected overall accuracy percentage (50-99, optional)
            sampling_method: 'stratified', 'simple', or 'systematic' (optional)
            simple_total_samples: Total samples for simple/systematic sampling (optional)
            stratified_allocation_method: 'proportional', 'equal', 'neyman', or 'balanced' for stratified (optional)

        Raises:
            ValueError: If parameters are out of valid range
        """
        if target_error <= 0:
            raise ValueError("Target error must be greater than 0")
        if target_error < 1.0:
            raise ValueError("Target error must be at least 1%")
        if target_error > 10.0:
            raise ValueError("Target error must not exceed 10%")

        if confidence_level < 90.0:
            raise ValueError("Confidence level must be at least 90%")
        if confidence_level > 99.0:
            raise ValueError("Confidence level must not exceed 99%")

        if min_samples_per_class is not None:
            if min_samples_per_class < 1:
                raise ValueError("Minimum samples per class must be at least 1")
            if min_samples_per_class > 100:
                raise ValueError("Minimum samples per class must not exceed 100")
            self.min_samples_per_class.value = min_samples_per_class

        if expected_accuracy is not None:
            if expected_accuracy < 50.0:
                raise ValueError("Expected accuracy must be at least 50%")
            if expected_accuracy > 99.0:
                raise ValueError("Expected accuracy must not exceed 99%")
            self.expected_accuracy.value = expected_accuracy

        if sampling_method is not None:
            if sampling_method not in ("stratified", "simple", "systematic"):
                raise ValueError(
                    "sampling_method must be one of: stratified, simple, systematic"
                )
            self.sampling_method.value = sampling_method

        if simple_total_samples is not None:
            if simple_total_samples < 1:
                raise ValueError("simple_total_samples must be at least 1")
            if simple_total_samples > 1000000:
                raise ValueError("simple_total_samples is unreasonably large")
            self.simple_total_samples.value = simple_total_samples

        if stratified_allocation_method is not None:
            if stratified_allocation_method not in (
                "proportional",
                "equal",
                "neyman",
                "balanced",
            ):
                raise ValueError(
                    "stratified_allocation_method must be one of: proportional, equal, neyman"
                )
            self.stratified_allocation_method.value = stratified_allocation_method

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

            # Only recalculate MOE for simple/systematic sampling
            # For stratified, MOE calculation requires per-class variances
            sampling_method = results.get("sampling_method", "stratified")
            if sampling_method in ("simple", "systematic"):
                confidence_level = results.get("confidence_level", 95.0) / 100.0
                expected_oa = self.expected_accuracy.value / 100.0
                current_moe = calculate_current_moe(
                    current_sample_size=total_samples,
                    target_oa=expected_oa,
                    confidence_level=confidence_level,
                )
                results["current_moe_percent"] = current_moe * 100
                results["current_moe_decimal"] = current_moe
            else:
                # For stratified, set MOE to None (not calculable with single OA value)
                results["current_moe_percent"] = None
                results["current_moe_decimal"] = None

            # Update allocation_dict with new values
            allocation_dict = results.get("allocation_dict", {}).copy()
            allocation_dict[map_code] = samples
            results["allocation_dict"] = allocation_dict

            self.sample_results.value = results

    def set_sample_points(self, points_df: pd.DataFrame):
        """Set generated sample points."""
        self.sample_points.value = points_df
        # Store the sampling method used when generating these points
        if self.sample_results.value:
            self.points_sampling_method.value = self.sample_results.value.get(
                "sampling_method", "stratified"
            )
        self.current_step.value = max(self.current_step.value, 5)

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
        if self.area_data.value is None or self.area_data.value.empty:
            return {}

        return dict(
            zip(
                self.area_data.value["map_code"],
                self.area_data.value["map_edited_class"],
            )
        )

    @property
    def allocation_method(self) -> str:
        """Get the current allocation method."""
        if self.sample_results.value:
            # For stratified sampling, get the actual allocation method used
            sampling_method = self.sample_results.value.get(
                "sampling_method", "stratified"
            )
            if sampling_method == "stratified":
                return self.sample_results.value.get(
                    "allocation_method", self.stratified_allocation_method.value
                ).lower()
            else:
                return sampling_method
        return self.stratified_allocation_method.value

    def get_allocation_data(self) -> List[Dict]:
        """Get allocation data for display."""
        if (
            not self.sample_results.value
            or self.area_data.value is None
            or self.area_data.value.empty
        ):
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
        # For stratified sampling, need area_data
        if self.sampling_method.value == "stratified":
            return (
                self.area_data.value is not None
                and not self.area_data.value.empty
                and self.target_error.value > 0
                and self.confidence_level.value > 0
            )
        # For simple/systematic sampling, need AOI
        elif self.sampling_method.value in ("simple", "systematic"):
            return (
                self.aoi_gdf.value is not None
                and self.confidence_level.value > 0
                and self.simple_total_samples.value > 0
            )
        return False

    def is_ready_for_point_generation(self) -> bool:
        """Check if ready for point generation."""
        if self.sample_results.value:
            sampling_method = self.sample_results.value.get(
                "sampling_method", "stratified"
            )
            if sampling_method in ("simple", "systematic"):
                # For non-stratified methods, need total_samples and AOI
                return (
                    self.sample_results.value.get("total_samples", 0) > 0
                    and self.aoi_gdf.value is not None
                )
            else:
                # For stratified sampling, need samples_per_class and file_path
                return (
                    bool(self.samples_per_class.value)
                    and self.file_path.value is not None
                )
        return False

    def export_csv(self) -> str:
        """Export sample points to CSV format."""
        if self.sample_points.value is None or self.sample_points.value.empty:
            return ""

        csv_content = self.sample_points.value.to_csv(index=False)
        self.last_export_csv.value = csv_content
        return csv_content

    def export_geojson(self) -> str:
        """Export sample points to GeoJSON format."""
        if self.sample_points.value is None or self.sample_points.value.empty:
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

    def clear_file_data(self):
        """Clear all file-related data (for stratified sampling).

        This centralizes the logic for clearing uploaded file state,
        avoiding duplication across components.
        """
        self.uploaded_file_info.value = None
        self.file_path.value = None
        self.area_data.value = None
        self.original_area_data.value = None
        self.file_error.value = None
        self.error_messages.value = []
        self.sample_results.value = None
        self.samples_per_class.value = {}
        self.sample_points.value = pd.DataFrame()
        self.points_generation_status.value = None

    def clear_aoi_data(self):
        """Clear all AOI-related data (for simple/systematic sampling).

        This centralizes the logic for clearing AOI state,
        avoiding duplication across components.
        """
        self.aoi_data.value = None
        self.aoi_gdf.value = None
        self.aoi_computing.value = False
        self.sample_results.value = None
        self.sample_points.value = pd.DataFrame()
        self.points_generation_status.value = None

    def clear_all_sampling_data(self):
        """Clear all sampling-related data (both file and AOI).

        Use this when switching between sampling methods or doing a full reset.
        """
        # Clear file data (stratified)
        self.uploaded_file_info.value = None
        self.file_path.value = None
        self.area_data.value = None
        self.original_area_data.value = None

        # Clear AOI data (simple/systematic)
        self.aoi_data.value = None
        self.aoi_gdf.value = None
        self.aoi_computing.value = False

        # Clear shared data
        self.file_error.value = None
        self.error_messages.value = []
        self.sample_results.value = None
        self.samples_per_class.value = {}
        self.sample_points.value = pd.DataFrame()
        self.points_generation_status.value = None


app_state = AppState()
