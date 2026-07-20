"""SBAE Geospatial Processing Module.

Contains functions for file I/O, area calculation, and point generation.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from shapely.geometry import Point


def generate_simple_random_points_from_aoi(
    aoi_gdf: gpd.GeoDataFrame,
    total_samples: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate simple random sample points within AOI boundaries.

    Args:
        aoi_gdf: GeoDataFrame with AOI geometry
        total_samples: Total number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample points in EPSG:4326
    """
    if seed is not None:
        np.random.seed(seed)

    sample_points = []

    try:
        if aoi_gdf.crs is None:
            aoi_gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = aoi_gdf
        elif not aoi_gdf.crs.is_geographic:
            gdf_geo = aoi_gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = aoi_gdf.copy()

        bounds = gdf_geo.total_bounds
        minx, miny, maxx, maxy = bounds

        samples_generated = 0
        max_attempts = total_samples * 100

        for attempt in range(max_attempts):
            if samples_generated >= total_samples:
                break

            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)

            for geom in gdf_geo.geometry:
                if geom.contains(point):
                    sample_points.append(
                        {
                            "longitude": x,
                            "latitude": y,
                            "map_code": 0,
                            "map_edited_class": "Sample",
                        }
                    )
                    samples_generated += 1
                    break

    except Exception as e:
        raise ValueError(f"Error generating simple random points from AOI: {e!s}")

    return pd.DataFrame(sample_points)


def generate_systematic_points_from_aoi(
    aoi_gdf: gpd.GeoDataFrame,
    total_samples: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate systematic sample points within AOI boundaries (grid-based).

    Args:
        aoi_gdf: GeoDataFrame with AOI geometry
        total_samples: Total number of samples to generate
        seed: Random seed for starting point offset

    Returns:
        DataFrame with sample points in EPSG:4326
    """
    if seed is not None:
        np.random.seed(seed)

    sample_points = []

    try:
        if aoi_gdf.crs is None:
            aoi_gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = aoi_gdf
        elif not aoi_gdf.crs.is_geographic:
            gdf_geo = aoi_gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = aoi_gdf.copy()

        bounds = gdf_geo.total_bounds
        minx, miny, maxx, maxy = bounds

        area = (maxx - minx) * (maxy - miny)
        grid_spacing = np.sqrt(area / total_samples)

        offset_x = (
            np.random.uniform(0, grid_spacing) if seed is not None else grid_spacing / 2
        )
        offset_y = (
            np.random.uniform(0, grid_spacing) if seed is not None else grid_spacing / 2
        )

        x = minx + offset_x
        while x < maxx:
            y = miny + offset_y
            while y < maxy:
                point = Point(x, y)
                for geom in gdf_geo.geometry:
                    if geom.contains(point):
                        sample_points.append(
                            {
                                "longitude": x,
                                "latitude": y,
                                "map_code": 0,
                                "map_edited_class": "Sample",
                            }
                        )
                        break
                y += grid_spacing
            x += grid_spacing

    except Exception as e:
        raise ValueError(f"Error generating systematic points from AOI: {e!s}")

    return pd.DataFrame(sample_points)


def is_raster_file(file_path: str) -> bool:
    """Check if file is a supported raster format by attempting to open with rasterio."""
    try:
        with rasterio.open(file_path) as _:
            return True
    except Exception:
        return False


def is_vector_file(file_path: str) -> bool:
    """Check if file is a supported vector format."""
    vector_extensions = [".shp", ".geojson", ".gpkg", ".kml"]
    return Path(file_path).suffix.lower() in vector_extensions


def extract_raster_colormap(file_path: str) -> Dict[int, str]:
    """Extract color palette from raster file.

    Args:
        file_path: Path to raster file

    Returns:
        Dictionary mapping class codes to hex color strings.
        Returns empty dict if no colormap is found.
    """
    colors = {}

    try:
        with rasterio.open(file_path) as raster:
            # Try to get the colormap from the raster
            colormap = raster.colormap(1)

            if colormap:
                for class_code, rgba in colormap.items():
                    # Convert RGBA tuple to hex color
                    # rasterio returns values in 0-255 range
                    r, g, b, _ = rgba
                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    colors[class_code] = hex_color

    except Exception:
        # If we can't extract colors, return empty dict
        pass

    return colors


def get_color_palette(file_path: str, class_codes: List[int]) -> Dict[int, str]:
    """Get color palette for given class codes, extracting from file if possible.

    Args:
        file_path: Path to raster or vector file
        class_codes: List of class codes to get colors for

    Returns:
        Dictionary mapping class codes to hex color strings
    """
    # Default color palette (ECharts default colors)
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

    # Try to extract colors from raster file
    extracted_colors = {}
    if is_raster_file(file_path):
        extracted_colors = extract_raster_colormap(file_path)

    # Build color mapping for each class code
    color_map = {}
    for idx, class_code in enumerate(sorted(class_codes)):
        if class_code in extracted_colors:
            # Use extracted color if available
            color_map[class_code] = extracted_colors[class_code]
        else:
            # Fall back to default palette
            color_map[class_code] = default_colors[idx % len(default_colors)]

    return color_map


def compute_area_from_raster(file_path: str) -> pd.DataFrame:
    """Compute area for each class in a raster file.

    Args:
        file_path: Path to raster file

    Returns:
        DataFrame with columns: map_code, map_area, map_edited_class

    Raises:
        ValueError: If file cannot be read or processed
    """
    try:
        with rasterio.open(file_path) as raster:
            # Calculate pixel area
            transform = raster.transform
            pixel_area = abs(transform.a * transform.e)

            nodata_value = raster.nodata
            dtype = np.dtype(raster.dtypes[0])
            # bincount is O(n) and avoids sorting the whole band, but only
            # works for non-negative integers. Fall back to per-block unique
            # for float/signed class maps.
            use_bincount = np.issubdtype(dtype, np.integer) or np.issubdtype(
                dtype, np.unsignedinteger
            )

            # Stream the raster block by block so we never materialize the
            # full band in memory (a single Hansen tile is ~3.5 GB read whole).
            counts: Dict[float, int] = {}
            for _, window in raster.block_windows(1):
                block = raster.read(1, window=window).ravel()

                if use_bincount and block.size and block.min() >= 0:
                    hist = np.bincount(block)
                    for code in np.nonzero(hist)[0]:
                        counts[int(code)] = counts.get(int(code), 0) + int(hist[code])
                else:
                    values, block_counts = np.unique(block, return_counts=True)
                    for value, count in zip(values, block_counts):
                        key = value.item()
                        counts[key] = counts.get(key, 0) + int(count)

            # Drop nodata (only if the raster actually declares one)
            if nodata_value is not None:
                counts.pop(nodata_value, None)

            if not counts:
                raise ValueError("No valid data found in raster")

            unique_values = sorted(counts)
            areas = np.array([counts[code] for code in unique_values]) * pixel_area

            return pd.DataFrame(
                {
                    "map_code": unique_values,
                    "map_area": areas,
                    "map_edited_class": [
                        f"Class {int(code)}" for code in unique_values
                    ],
                }
            )

    except Exception as e:
        raise ValueError(f"Error processing raster file: {e!s}")


def compute_area_from_vector(file_path: str) -> pd.DataFrame:
    """Compute area for each class in a vector file.

    Args:
        file_path: Path to vector file

    Returns:
        DataFrame with columns: map_code, map_area, map_edited_class

    Raises:
        ValueError: If file cannot be read or no suitable class column found
    """
    try:
        gdf = gpd.read_file(file_path)

        if len(gdf) == 0:
            raise ValueError("Vector file contains no features")

        # Find the first non-geometry column as class column
        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                # Check if column has reasonable values for classification
                unique_vals = gdf[col].dropna().unique()
                if (
                    len(unique_vals) > 0 and len(unique_vals) <= 50
                ):  # Reasonable number of classes
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found in vector file")

        # Calculate areas (ensure we're in a projected CRS for accurate areas)
        if gdf.crs and gdf.crs.is_geographic:
            # Convert to appropriate UTM zone for area calculation
            gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
            areas = gdf_projected.geometry.area
        else:
            areas = gdf.geometry.area

        # Group by class and sum areas
        gdf_with_areas = gdf.copy()
        gdf_with_areas["area"] = areas
        area_by_class = gdf_with_areas.groupby(class_column)["area"].sum()

        return pd.DataFrame(
            {
                "map_code": area_by_class.index,
                "map_area": area_by_class.values,
                "map_edited_class": [f"Class {code}" for code in area_by_class.index],
            }
        )

    except Exception as e:
        raise ValueError(f"Error processing vector file: {e!s}")


def compute_file_areas(file_path: str) -> pd.DataFrame:
    """Automatically detect file type and compute areas.

    Args:
        file_path: Path to classification file

    Returns:
        DataFrame with area information

    Raises:
        ValueError: If file format is not supported
    """
    if is_raster_file(file_path):
        return compute_area_from_raster(file_path)
    elif is_vector_file(file_path):
        return compute_area_from_vector(file_path)
    else:
        file_ext = Path(file_path).suffix.lower()
        raise ValueError(f"Unsupported file format: {file_ext}")


def save_uploaded_file(file_info, temp_dir: Optional[str] = None) -> str:
    """Save uploaded file to temporary directory.

    Args:
        file_info: FileInfo object from Solara FileDrop
        temp_dir: Optional temporary directory (created if None)

    Returns:
        Path to saved file
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    file_path = os.path.join(temp_dir, file_info["name"])

    with open(file_path, "wb") as f:
        file_info["file_obj"].seek(0)
        f.write(file_info["file_obj"].read())

    return file_path


def _choose_ordinals(n_avail: int, k: int):
    """``k`` distinct sorted random ordinals in ``[0, n_avail)``.

    For huge populations we must not use ``np.random.choice(n, k,
    replace=False)`` — it permutes an array of size ``n`` (tens of GB for a
    Hansen tile). Since ``k`` is tiny relative to ``n``, rejection sampling of
    random integers gives distinct ordinals with negligible retries and O(k)
    memory.
    """
    k = min(int(k), int(n_avail))
    if k <= 0:
        return np.array([], dtype=np.int64)
    if n_avail <= 1_000_000:
        return np.sort(np.random.choice(n_avail, size=k, replace=False)).astype(
            np.int64
        )
    ords = np.unique(np.random.randint(0, n_avail, size=k))
    while ords.size < k:
        extra = np.random.randint(0, n_avail, size=k - ords.size)
        ords = np.unique(np.concatenate([ords, extra]))
    return ords[:k].astype(np.int64)


def _empty_pixels():
    z = np.array([], dtype=np.int64)
    return z, z, z


def _iter_row_chunks(raster, target_pixels: int = 16_000_000):
    """Yield full-width row-chunk windows sized to ~target_pixels each.

    Reading in chunks (instead of one internal strip at a time) keeps the work
    numpy-bound rather than Python-bound: a striped tile has ~60k 1-row strips,
    which is far too many Python iterations, while chunk memory stays bounded
    (~target_pixels bytes for a Byte raster).
    """
    width = raster.width
    height = raster.height
    chunk_rows = max(1, target_pixels // max(1, width))
    for row_off in range(0, height, chunk_rows):
        rows = min(chunk_rows, height - row_off)
        yield Window(0, row_off, width, rows)


def _sample_pixels_per_class(raster, targets: Dict[int, int]):
    """Uniform random pixel coords per class, read block-by-block.

    Never materializes the full band, so it is safe on very large rasters
    (a full Hansen tile is ~3.5 GB read whole, and ``np.where(data == code)``
    on the dominant class would allocate tens of GB of indices).

    Returns ``{class_code: (rows, cols)}`` with ``min(n, available)`` entries.
    """
    caps = {int(c): int(n) for c, n in targets.items() if n and int(n) > 0}
    if not caps:
        return {}
    codes = list(caps)

    # Pass 1: count pixels per class (bool-sum only; no index materialization).
    counts = {c: 0 for c in codes}
    for window in _iter_row_chunks(raster):
        block = raster.read(1, window=window)
        for c in codes:
            counts[c] += int((block == c).sum())

    # Pick target ordinals per class within [0, N_c).
    ordinals = {c: _choose_ordinals(counts[c], caps[c]) for c in codes}

    # Pass 2: walk chunks, advance a running per-class counter, and pull only
    # the targeted pixels. np.where runs only on chunks that hold a target.
    seen = {c: 0 for c in codes}
    ptr = {c: 0 for c in codes}
    out_rows = {c: [] for c in codes}
    out_cols = {c: [] for c in codes}

    for window in _iter_row_chunks(raster):
        block = raster.read(1, window=window)
        for c in codes:
            ords = ordinals[c]
            if ptr[c] >= ords.size:
                continue
            mask = block == c
            cc = int(mask.sum())
            if cc == 0:
                continue
            lo = seen[c]
            hi = lo + cc
            p = ptr[c]
            local = []
            while p < ords.size and ords[p] < hi:
                local.append(ords[p] - lo)  # index among class-c pixels here
                p += 1
            if local:
                br, bc = np.where(mask)  # row-major order == global scan order
                take = np.asarray(local, dtype=np.int64)
                out_rows[c].extend((br[take] + window.row_off).tolist())
                out_cols[c].extend((bc[take] + window.col_off).tolist())
            ptr[c] = p
            seen[c] = hi

    return {
        c: (
            np.asarray(out_rows[c], dtype=np.int64),
            np.asarray(out_cols[c], dtype=np.int64),
        )
        for c in codes
    }


def _sample_pixels_valid(raster, cap: int, nodata):
    """Uniform random sample of ``cap`` valid pixels, read block-by-block.

    Returns ``(rows, cols, values)`` with ``min(cap, available)`` entries.
    """

    def _valid_mask(block):
        return block != nodata if nodata is not None else np.full(block.shape, True)

    # Pass 1: count valid pixels.
    total_valid = 0
    for window in _iter_row_chunks(raster):
        block = raster.read(1, window=window)
        total_valid += int(_valid_mask(block).sum())

    ordinals = _choose_ordinals(total_valid, cap)
    if ordinals.size == 0:
        return _empty_pixels()

    # Pass 2: locate the targeted valid pixels (and their class values).
    seen = 0
    ptr = 0
    out_rows: list = []
    out_cols: list = []
    out_vals: list = []
    for window in _iter_row_chunks(raster):
        if ptr >= ordinals.size:
            break
        block = raster.read(1, window=window)
        mask = _valid_mask(block)
        cc = int(mask.sum())
        if cc == 0:
            continue
        lo = seen
        hi = lo + cc
        p = ptr
        local = []
        while p < ordinals.size and ordinals[p] < hi:
            local.append(ordinals[p] - lo)
            p += 1
        if local:
            br, bc = np.where(mask)
            take = np.asarray(local, dtype=np.int64)
            out_rows.extend((br[take] + window.row_off).tolist())
            out_cols.extend((bc[take] + window.col_off).tolist())
            out_vals.extend(block[br[take], bc[take]].astype(np.int64).tolist())
        ptr = p
        seen = hi

    return (
        np.asarray(out_rows, dtype=np.int64),
        np.asarray(out_cols, dtype=np.int64),
        np.asarray(out_vals, dtype=np.int64),
    )


def _pixels_to_lonlat(transform, crs, rows, cols):
    """Vectorized pixel-center (row, col) -> (lon, lat) in EPSG:4326."""
    xs, ys = xy(transform, np.asarray(rows), np.asarray(cols), offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if crs and not crs.is_geographic:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs, ys)
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return xs, ys


def generate_sample_points_raster(
    file_path: str,
    samples_per_class: Dict[int, int],
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate stratified random sample points from raster data.

    Args:
        file_path: Path to raster file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility (None for random)

    Returns:
        DataFrame with sample points (longitude, latitude, map_code, map_edited_class)
        Points are always in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        with rasterio.open(file_path) as raster:
            transform = raster.transform
            crs = raster.crs

            sampled = _sample_pixels_per_class(raster, samples_per_class)

            for class_code, (rows, cols) in sampled.items():
                if rows.size == 0:
                    print(f"Warning: No pixels found for class {class_code}")
                    continue

                lon, lat = _pixels_to_lonlat(transform, crs, rows, cols)
                class_name = class_lookup.get(class_code, f"Class {class_code}")

                for lo, la in zip(lon, lat):
                    sample_points.append(
                        {
                            "longitude": float(lo),
                            "latitude": float(la),
                            "map_code": class_code,
                            "map_edited_class": class_name,
                        }
                    )

    except Exception as e:
        raise ValueError(f"Error generating points from raster: {e!s}")

    return pd.DataFrame(sample_points)


def generate_sample_points_vector(
    file_path: str,
    samples_per_class: Dict[int, int],
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate stratified random sample points from vector data.

    Args:
        file_path: Path to vector file
        samples_per_class: Dictionary of samples needed per class
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility (None for random)

    Returns:
        DataFrame with sample points (longitude, latitude, map_code, map_edited_class)
        Points are always in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        gdf = gpd.read_file(file_path)

        # Find the class column (same logic as in area computation)
        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                unique_vals = gdf[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found")

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = gdf
        elif not gdf.crs.is_geographic:
            gdf_geo = gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = gdf.copy()

        for class_code, n_samples in samples_per_class.items():
            if n_samples <= 0:
                continue

            # Filter geometries for this class
            class_geometries = gdf_geo[gdf_geo[class_column] == class_code]

            if class_geometries.empty:
                print(f"Warning: No geometries found for class {class_code}")
                continue

            # Generate random points within geometries
            samples_generated = 0
            max_attempts = n_samples * 100

            # Get bounds for efficiency
            bounds = class_geometries.total_bounds
            minx, miny, maxx, maxy = bounds

            for attempt in range(max_attempts):
                if samples_generated >= n_samples:
                    break

                # Generate random point within bounds
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                point = Point(x, y)

                # Check if point falls within any class geometry
                if any(geom.contains(point) for geom in class_geometries.geometry):
                    sample_points.append(
                        {
                            "longitude": x,
                            "latitude": y,
                            "map_code": class_code,
                            "map_edited_class": class_lookup.get(
                                class_code, f"Class {class_code}"
                            ),
                        }
                    )
                    samples_generated += 1

            if samples_generated < n_samples:
                print(
                    f"Warning: Only generated {samples_generated}/{n_samples} points for class {class_code}"
                )

    except Exception as e:
        raise ValueError(f"Error generating points from vector: {e!s}")

    return pd.DataFrame(sample_points)


def generate_simple_random_points_raster(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate simple random sample points from raster (no stratification).

    Args:
        file_path: Path to raster file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        with rasterio.open(file_path) as raster:
            transform = raster.transform
            crs = raster.crs

            rows, cols, vals = _sample_pixels_valid(
                raster, total_samples, raster.nodata
            )

            if rows.size == 0:
                raise ValueError("No valid pixels found in raster")

            lon, lat = _pixels_to_lonlat(transform, crs, rows, cols)

            for lo, la, class_code in zip(lon, lat, vals):
                class_code = int(class_code)
                sample_points.append(
                    {
                        "longitude": float(lo),
                        "latitude": float(la),
                        "map_code": class_code,
                        "map_edited_class": class_lookup.get(
                            class_code, f"Class {class_code}"
                        ),
                    }
                )

    except Exception as e:
        raise ValueError(f"Error generating simple random points from raster: {e!s}")

    return pd.DataFrame(sample_points)


def generate_systematic_points_raster(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate systematic sample points from raster (grid-based sampling).

    Args:
        file_path: Path to raster file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for starting point offset

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        with rasterio.open(file_path) as raster:
            transform = raster.transform
            crs = raster.crs
            nodata = raster.nodata
            height, width = raster.height, raster.width

            total_pixels = height * width
            grid_interval = int(np.sqrt(total_pixels / total_samples))
            if grid_interval < 1:
                grid_interval = 1

            offset_row = (
                np.random.randint(0, grid_interval)
                if seed is not None
                else grid_interval // 2
            )
            offset_col = (
                np.random.randint(0, grid_interval)
                if seed is not None
                else grid_interval // 2
            )

            grid_cols = np.arange(offset_col, width, grid_interval)
            sel_rows: list = []
            sel_cols: list = []
            sel_vals: list = []

            # Read only the sampled rows (one strip at a time) instead of the
            # whole band, so this stays cheap on very large rasters.
            for row in range(offset_row, height, grid_interval):
                if len(sel_rows) >= total_samples:
                    break

                line = raster.read(1, window=Window(0, row, width, 1))[0]
                vals = line[grid_cols]

                if nodata is not None:
                    keep = vals != nodata
                    cols_here = grid_cols[keep]
                    vals_here = vals[keep]
                else:
                    cols_here = grid_cols
                    vals_here = vals

                remaining = total_samples - len(sel_rows)
                if cols_here.size > remaining:
                    cols_here = cols_here[:remaining]
                    vals_here = vals_here[:remaining]

                sel_rows.extend([row] * cols_here.size)
                sel_cols.extend(cols_here.tolist())
                sel_vals.extend(vals_here.tolist())

            if sel_rows:
                lon, lat = _pixels_to_lonlat(
                    transform, crs, np.array(sel_rows), np.array(sel_cols)
                )
                for lo, la, class_code in zip(lon, lat, sel_vals):
                    class_code = int(class_code)
                    sample_points.append(
                        {
                            "longitude": float(lo),
                            "latitude": float(la),
                            "map_code": class_code,
                            "map_edited_class": class_lookup.get(
                                class_code, f"Class {class_code}"
                            ),
                        }
                    )

    except Exception as e:
        raise ValueError(f"Error generating systematic points from raster: {e!s}")

    return pd.DataFrame(sample_points)


def generate_simple_random_points_vector(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate simple random sample points from vector (no stratification).

    Args:
        file_path: Path to vector file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        gdf = gpd.read_file(file_path)

        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                unique_vals = gdf[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found")

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = gdf
        elif not gdf.crs.is_geographic:
            gdf_geo = gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = gdf.copy()

        bounds = gdf_geo.total_bounds
        minx, miny, maxx, maxy = bounds

        samples_generated = 0
        max_attempts = total_samples * 100

        for attempt in range(max_attempts):
            if samples_generated >= total_samples:
                break

            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)

            for idx, geom in enumerate(gdf_geo.geometry):
                if geom.contains(point):
                    class_code = int(gdf_geo.iloc[idx][class_column])
                    sample_points.append(
                        {
                            "longitude": x,
                            "latitude": y,
                            "map_code": class_code,
                            "map_edited_class": class_lookup.get(
                                class_code, f"Class {class_code}"
                            ),
                        }
                    )
                    samples_generated += 1
                    break

    except Exception as e:
        raise ValueError(f"Error generating simple random points from vector: {e!s}")

    return pd.DataFrame(sample_points)


def generate_systematic_points_vector(
    file_path: str,
    total_samples: int,
    class_lookup: Dict[int, str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate systematic sample points from vector (grid-based sampling).

    Args:
        file_path: Path to vector file
        total_samples: Total number of samples to generate
        class_lookup: Mapping of class codes to names
        seed: Random seed for starting point offset

    Returns:
        DataFrame with sample points in EPSG:4326 (WGS84) geographic coordinates.
    """
    sample_points = []

    if seed is not None:
        np.random.seed(seed)

    try:
        gdf = gpd.read_file(file_path)

        class_column = None
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype in [
                "int64",
                "int32",
                "object",
                "string",
            ]:
                unique_vals = gdf[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    class_column = col
                    break

        if class_column is None:
            raise ValueError("No suitable class column found")

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
            gdf_geo = gdf
        elif not gdf.crs.is_geographic:
            gdf_geo = gdf.to_crs("EPSG:4326")
        else:
            gdf_geo = gdf.copy()

        bounds = gdf_geo.total_bounds
        minx, miny, maxx, maxy = bounds

        area = (maxx - minx) * (maxy - miny)
        grid_spacing = np.sqrt(area / total_samples)

        offset_x = (
            np.random.uniform(0, grid_spacing) if seed is not None else grid_spacing / 2
        )
        offset_y = (
            np.random.uniform(0, grid_spacing) if seed is not None else grid_spacing / 2
        )

        x_coords = np.arange(minx + offset_x, maxx, grid_spacing)
        y_coords = np.arange(miny + offset_y, maxy, grid_spacing)

        for x in x_coords:
            for y in y_coords:
                if len(sample_points) >= total_samples:
                    break

                point = Point(x, y)

                for idx, geom in enumerate(gdf_geo.geometry):
                    if geom.contains(point):
                        class_code = int(gdf_geo.iloc[idx][class_column])
                        sample_points.append(
                            {
                                "longitude": x,
                                "latitude": y,
                                "map_code": class_code,
                                "map_edited_class": class_lookup.get(
                                    class_code, f"Class {class_code}"
                                ),
                            }
                        )
                        break

            if len(sample_points) >= total_samples:
                break

    except Exception as e:
        raise ValueError(f"Error generating systematic points from vector: {e!s}")

    return pd.DataFrame(sample_points)


def generate_sample_points(
    file_path: Optional[str] = None,
    samples_per_class: Optional[Dict[int, int]] = None,
    class_lookup: Optional[Dict[int, str]] = None,
    seed: Optional[int] = None,
    sampling_method: str = "stratified",
    total_samples: Optional[int] = None,
    aoi_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """Automatically detect file type and generate sample points.

    Args:
        file_path: Path to classification file (required for stratified sampling)
        samples_per_class: Dictionary of samples needed per class (empty for simple/systematic)
        class_lookup: Mapping of class codes to names
        seed: Random seed for reproducibility (None for random)
        sampling_method: "stratified", "simple", or "systematic"
        total_samples: Total samples to generate (for simple/systematic methods)
        aoi_gdf: GeoDataFrame with AOI boundaries (for simple/systematic sampling)

    Returns:
        DataFrame with sample points

    Raises:
        ValueError: If file format is not supported or required parameters missing
    """
    if samples_per_class is None:
        samples_per_class = {}
    if class_lookup is None:
        class_lookup = {}

    # For simple random or systematic sampling with AOI
    if (
        sampling_method in ("simple", "systematic")
        and total_samples
        and aoi_gdf is not None
    ):
        if sampling_method == "simple":
            return generate_simple_random_points_from_aoi(aoi_gdf, total_samples, seed)
        else:  # systematic
            return generate_systematic_points_from_aoi(aoi_gdf, total_samples, seed)

    # For simple random or systematic sampling from file
    if sampling_method == "simple" and total_samples and file_path:
        if is_raster_file(file_path):
            return generate_simple_random_points_raster(
                file_path, total_samples, class_lookup, seed
            )
        elif is_vector_file(file_path):
            return generate_simple_random_points_vector(
                file_path, total_samples, class_lookup, seed
            )
    elif sampling_method == "systematic" and total_samples and file_path:
        if is_raster_file(file_path):
            return generate_systematic_points_raster(
                file_path, total_samples, class_lookup, seed
            )
        elif is_vector_file(file_path):
            return generate_systematic_points_vector(
                file_path, total_samples, class_lookup, seed
            )

    # Default to stratified sampling
    if not file_path:
        raise ValueError("file_path is required for stratified sampling")

    if is_raster_file(file_path):
        return generate_sample_points_raster(
            file_path, samples_per_class, class_lookup, seed
        )
    elif is_vector_file(file_path):
        return generate_sample_points_vector(
            file_path, samples_per_class, class_lookup, seed
        )
    else:
        file_ext = Path(file_path).suffix.lower()
        raise ValueError(f"Unsupported file format: {file_ext}")


def export_points_to_csv(points_df: pd.DataFrame) -> str:
    """Export points to CSV format.

    Args:
        points_df: DataFrame with sample points

    Returns:
        CSV string
    """
    return points_df.to_csv(index=False)


def export_points_to_geojson(points_df: pd.DataFrame) -> str:
    """Export points to GeoJSON format.

    Args:
        points_df: DataFrame with sample points

    Returns:
        GeoJSON string
    """
    gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
        crs="EPSG:4326",
    )
    return gdf.to_json()


def get_file_info(file_path: str) -> Dict:
    """Get basic information about a geospatial file.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information
    """
    info = {
        "file_type": "unknown",
        "size_mb": Path(file_path).stat().st_size / (1024 * 1024),
        "crs": None,
        "bounds": None,
        "feature_count": 0,
    }

    def get_crs_string(crs):
        """Extract clean CRS representation."""
        if not crs:
            return None
        epsg = crs.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        # Try to extract EPSG from the CRS string representation
        crs_str = str(crs)
        if 'AUTHORITY["EPSG"' in crs_str:
            # Extract EPSG code from WKT string like AUTHORITY["EPSG","3116"]
            import re

            match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', crs_str)
            if match:
                return f"EPSG:{match.group(1)}"
        return "Custom CRS"

    try:
        if is_raster_file(file_path):
            with rasterio.open(file_path) as raster:
                info["file_type"] = "raster"
                info["crs"] = get_crs_string(raster.crs)
                info["bounds"] = list(raster.bounds)
                info["width"] = raster.width
                info["height"] = raster.height
                info["feature_count"] = raster.width * raster.height

        elif is_vector_file(file_path):
            gdf = gpd.read_file(file_path)
            info["file_type"] = "vector"
            info["crs"] = get_crs_string(gdf.crs)
            info["bounds"] = list(gdf.total_bounds)
            info["feature_count"] = len(gdf)

    except Exception as e:
        info["error"] = str(e)

    return info


def extract_map_codes(
    reference_df: pd.DataFrame,
    raster_path: str,
    x_col: str,
    y_col: str,
    points_crs: str = "EPSG:4326",
    drop_missing: bool = True,
) -> "tuple[pd.DataFrame, int]":
    """Sample a classification raster at each reference point to fill map_code.

    Reprojects the (x_col, y_col) points from points_crs to the raster CRS,
    samples band 1, and returns (df_with_map_code, dropped_count). Points that
    fall outside the raster footprint or on the nodata value are "missing".

    ``drop_missing=True`` (default, the analysis path) drops those rows.
    ``drop_missing=False`` (the design path, where the sample is fixed) keeps
    every row and leaves missing points at their prior ``map_code`` (or 0).
    In both cases ``dropped_count`` reports how many were missing.
    """
    df = reference_df.copy()
    xs = df[x_col].astype(float).to_numpy()
    ys = df[y_col].astype(float).to_numpy()
    with rasterio.open(raster_path) as src:
        if src.crs is not None and points_crs and str(src.crs) != str(points_crs):
            rxs, rys = warp_transform(points_crs, src.crs, xs.tolist(), ys.tolist())
        else:
            rxs, rys = xs.tolist(), ys.tolist()
        nodata = src.nodata
        pts = list(zip(rxs, rys))
        codes = []
        for (x, y), val in zip(pts, src.sample(pts, indexes=1)):
            row, col = src.index(x, y)
            v = val[0]
            inside = 0 <= row < src.height and 0 <= col < src.width
            if inside and not (nodata is not None and v == nodata):
                codes.append(int(v))
            else:
                codes.append(None)
    df["map_code"] = codes
    missing = df["map_code"].isna()
    dropped = int(missing.sum())
    if drop_missing:
        df = df[~missing].copy()
    else:
        # Keep every point; unsampleable ones fall back to their prior map_code
        # (0 for freshly-generated design points) instead of being dropped.
        fallback = reference_df["map_code"] if "map_code" in reference_df.columns else 0
        df["map_code"] = df["map_code"].where(~missing, fallback)
    df["map_code"] = df["map_code"].astype(int)
    return df, dropped
