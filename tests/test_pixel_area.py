"""pixel_area: planar where the CRS allows it, geodesic tile weights otherwise."""

import math

import pytest
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from component.scripts.pixel_area import (
    TileWeights,
    choose_area_method,
    planar_pixel_area,
    tile_geodesic_areas,
)
from component.scripts.raster_source import NotThematicError

UTM = from_origin(500_000, 4_650_000, 30, 30)
GEO_45N = from_origin(12.0, 45.0, 0.00027, 0.00027)
GEO_60N = from_origin(12.0, 60.0, 0.00027, 0.00027)
MERC_60N = from_origin(1_335_833, 8_399_737, 60, 60)
FEET = from_origin(6_000_000, 2_100_000, 100, 100)


def test_planar_pixel_area_in_metres_is_the_pixel_product():
    assert planar_pixel_area(UTM, CRS.from_epsg(32633)) == 900.0


def test_planar_pixel_area_converts_survey_feet_to_square_metres():
    area = planar_pixel_area(FEET, CRS.from_epsg(2227))
    assert area == pytest.approx(10_000 * 0.30480060960121924**2)


def test_tile_geodesic_areas_match_geod_on_one_geographic_cell():
    transform = from_origin(12.0, 45.0, 0.5, 0.5)

    areas = tile_geodesic_areas(transform, CRS.from_epsg(4326), [0, 1], [0, 1])

    expected, _ = Geod(ellps="WGS84").polygon_area_perimeter(
        [12.0, 12.5, 12.5, 12.0], [45.0, 45.0, 44.5, 44.5]
    )
    assert areas.shape == (1, 1)
    assert areas[0, 0] == pytest.approx(abs(expected))


def test_tile_geodesic_areas_of_a_utm_cell_are_close_to_planar():
    areas = tile_geodesic_areas(UTM, CRS.from_epsg(32633), [0, 10], [0, 10])
    assert areas[0, 0] == pytest.approx(100 * 900.0, rel=2e-3)


def test_choose_area_method_without_crs_raises_no_crs():
    with pytest.raises(NotThematicError) as info:
        choose_area_method(UTM, None, 10, 10)
    assert info.value.code == "no_crs"


def test_geographic_crs_is_always_geodesic():
    method, departure = choose_area_method(GEO_45N, CRS.from_epsg(4326), 300, 300)
    assert method == "geodesic"
    assert math.isinf(departure)


def test_utm_stays_planar():
    method, departure = choose_area_method(UTM, CRS.from_epsg(32633), 300, 300)
    assert method == "planar"
    assert departure < 0.01


def test_equal_area_stays_planar():
    method, departure = choose_area_method(
        from_origin(0, 0, 30, 30), CRS.from_epsg(6933), 300, 300
    )
    assert method == "planar"
    assert departure < 1e-6


def test_web_mercator_at_60n_is_geodesic():
    method, departure = choose_area_method(MERC_60N, CRS.from_epsg(3857), 300, 300)
    assert method == "geodesic"
    assert departure > 0.5


def test_tiny_rasters_get_a_method_too():
    method, _ = choose_area_method(UTM, CRS.from_epsg(32633), 3, 2)
    assert method == "planar"


def test_tile_weights_follow_the_lattice():
    weights = TileWeights(GEO_60N, CRS.from_epsg(4326), width=300, height=300, tile=256)

    assert weights.per_pixel.shape == (2, 2)
    # further south (second row of tiles) a pixel covers more ground
    assert weights.per_pixel[1, 0] > weights.per_pixel[0, 0]

    w = weights.for_window(Window(col_off=250, row_off=250, width=20, height=20))

    assert w.shape == (20, 20)
    assert w[0, 0] == weights.per_pixel[0, 0]  # row 250, col 250 -> tile (0, 0)
    assert w[-1, -1] == weights.per_pixel[1, 1]  # row 269, col 269 -> tile (1, 1)


def test_tile_weights_absorb_float_noise_in_window_offsets():
    weights = TileWeights(GEO_60N, CRS.from_epsg(4326), width=300, height=300, tile=256)

    exact = weights.for_window(Window(col_off=250, row_off=250, width=20, height=20))
    noisy = weights.for_window(
        Window(col_off=249.9999999, row_off=249.9999999, width=20.0000001, height=20)
    )

    assert noisy.shape == exact.shape
    assert (noisy == exact).all()


def test_planar_pixel_area_without_crs_fails_loud():
    with pytest.raises(AttributeError):
        planar_pixel_area(UTM, None)
