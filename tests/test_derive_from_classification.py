import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from component.scripts.accuracy import derive_from_classification


def _write(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(data, 1)


def test_derive_fills_map_code_and_areas(tmp_path):
    data = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    p = tmp_path / "clas.tif"
    _write(p, data)
    ref = pd.DataFrame({"lon": [0.5, 2.5], "lat": [3.5, 0.5], "ref_code": [1, 2]})
    mapping = {"x": "lon", "y": "lat", "ref": "ref_code"}
    ref_out, area_out, dropped = derive_from_classification(ref, mapping, str(p))
    assert dropped == 0
    assert ref_out["map_code"].tolist() == [1, 4]
    assert set(area_out.columns) >= {"map_code", "map_area"}
    assert sorted(area_out["map_code"].tolist()) == [1, 2, 3, 4]
