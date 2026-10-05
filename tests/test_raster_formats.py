"""Any raster GDAL can open is a valid classification map: one grid, many drivers.

The pipeline must give the same class areas and sample points for every
container format, and every point must read back its own class. Drivers the
installed GDAL cannot write are skipped, not failed: the local wheel and the
conda-forge build CI uses do not carry the same set.
"""

import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin

from component.scripts.geospatial import (
    compute_area_from_raster,
    extract_map_codes,
    generate_sample_points,
    get_color_palette,
    get_file_info,
)

# driver -> extension
DRIVERS = {
    "GTiff": "tif",
    "COG": "tif",
    "HFA": "img",
    "ENVI": "dat",
    "EHdr": "bil",
    "RST": "rst",
    "PCIDSK": "pix",
    "GPKG": "gpkg",
    "AAIGrid": "asc",
    "SAGA": "sdat",
    "ERS": "ers",
    "PNG": "png",
    "VRT": "vrt",
    "KEA": "kea",
    "netCDF": "nc",
    "JP2OpenJPEG": "jp2",
}
# drivers whose container stores a colour table
PALETTED = {"GTiff", "COG", "HFA", "PNG", "VRT"}
CODES = (1, 2, 3, 5)
COLORMAP = {
    1: (0, 100, 0, 255),
    2: (255, 255, 0, 255),
    3: (0, 0, 255, 255),
    5: (255, 0, 0, 255),
}
PIXEL_AREA = 900.0  # 30 m UTM pixels


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    rng = np.random.default_rng(0)
    data = rng.choice(
        [0, *CODES], size=(200, 300), p=[0.1, 0.4, 0.3, 0.15, 0.05]
    ).astype("uint8")
    path = tmp_path_factory.mktemp("formats") / "source.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=300,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500000, 4650000, 30, 30),
        nodata=0,
    ) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, COLORMAP)
    return path, data


@pytest.fixture(params=sorted(DRIVERS), ids=sorted(DRIVERS))
def raster(request, source, tmp_path):
    driver = request.param
    src, data = source
    dst = tmp_path / f"copy.{DRIVERS[driver]}"
    try:
        rio_copy(str(src), str(dst), driver=driver)
    except Exception as e:  # driver absent or read-only in this GDAL build
        pytest.skip(f"{driver}: cannot write here ({type(e).__name__}: {e})")
    return driver, str(dst), data


def test_areas_match_the_source_grid(raster):
    _, path, data = raster

    info = get_file_info(path)
    df = compute_area_from_raster(path).sort_values("map_code")

    assert info["file_type"] == "raster"
    assert info["crs"] == "EPSG:32633"
    assert df["map_code"].tolist() == list(CODES)
    counts = np.array([(data == code).sum() for code in CODES])
    np.testing.assert_allclose(df["map_area"].to_numpy(), counts * PIXEL_AREA)


def test_palette_covers_every_class(raster):
    driver, path, _ = raster

    palette = get_color_palette(path, list(CODES))

    assert set(palette) == set(CODES)
    if driver in PALETTED:
        assert palette == {1: "#006400", 2: "#ffff00", 3: "#0000ff", 5: "#ff0000"}


def test_points_read_back_their_own_class(raster):
    _, path, _ = raster

    points = generate_sample_points(
        file_path=path,
        samples_per_class={code: 5 for code in CODES},
        class_lookup={},
        seed=42,
    )
    back, dropped = extract_map_codes(
        points.assign(map_code=0), path, "longitude", "latitude"
    )

    assert len(points) == 20
    assert dropped == 0
    assert back["map_code"].tolist() == points["map_code"].tolist()
