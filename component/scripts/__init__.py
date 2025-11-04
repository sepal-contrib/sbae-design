"""SBAE Scripts Package.

Contains calculation and processing functions for the SBAE application.
"""

from .calc_utils import (
    get_z_score,
)
from .calculations import (
    apply_minimum_constraints,
    calculate_overall_accuracy_sample_size,
    calculate_sample_design,
    calculate_target_class_sample_size,
    validate_parameters,
)
from .geospatial import (
    compute_area_from_raster,
    compute_area_from_vector,
    compute_file_areas,
    export_points_to_csv,
    export_points_to_geojson,
    extract_raster_colormap,
    generate_sample_points,
    generate_sample_points_raster,
    generate_sample_points_vector,
    get_color_palette,
    get_file_info,
    is_raster_file,
    is_vector_file,
    save_uploaded_file,
)
from .stratified import (
    allocate_samples_equal,
    allocate_samples_neyman,
    allocate_samples_proportional,
    calculate_allocation_summary,
)

__all__ = [
    # Calculations
    "get_z_score",
    "calculate_overall_accuracy_sample_size",
    "calculate_target_class_sample_size",
    "allocate_samples_proportional",
    "allocate_samples_neyman",
    "allocate_samples_equal",
    "apply_minimum_constraints",
    "calculate_sample_design",
    "validate_parameters",
    "calculate_allocation_summary",
    # Geospatial Processing
    "is_raster_file",
    "is_vector_file",
    "compute_area_from_raster",
    "compute_area_from_vector",
    "compute_file_areas",
    "save_uploaded_file",
    "generate_sample_points_raster",
    "generate_sample_points_vector",
    "generate_sample_points",
    "export_points_to_csv",
    "export_points_to_geojson",
    "get_file_info",
    "extract_raster_colormap",
    "get_color_palette",
]
