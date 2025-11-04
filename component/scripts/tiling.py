# file: tiling_prepare.py
import hashlib
import os
import pathlib
import shutil
import subprocess

import rasterio as rio


def _hash_for_cache(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(path.encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def _gdal_ok():
    return (
        shutil.which("gdalinfo")
        and shutil.which("gdal_translate")
        and shutil.which("gdaladdo")
    )


def _is_categorical(ds: rio.io.DatasetReader) -> bool:
    # crude heuristic: integer dtype and few unique categories indicated by colormap or QL metadata
    if ds.count != 1:
        return False
    if ds.dtypes[0].startswith(("int8", "uint8", "int16", "uint16", "int32", "uint32")):
        return True
    return False


def _has_overviews(ds):
    return any(ds.overviews(i + 1) for i in range(ds.count))


def _is_tiled(ds):
    # block_shapes is None on some drivers; treat as not tiled
    try:
        bs = ds.block_shapes
        return bs and all((b[0] > 1 and b[1] > 1) for b in bs)
    except Exception:
        return False


def _needs_reproject(ds, target_epsg: int | None):
    if not target_epsg or not ds.crs:
        return False
    try:
        return ds.crs.to_epsg() != target_epsg
    except Exception:
        return True


def _target_overview_levels(width, height, block=512):
    # pyramid down to roughly block size
    levels = []
    longest = max(width, height)
    lvl = 2
    while longest / lvl > block:
        levels.append(lvl)
        lvl *= 2
    return levels or [2, 4, 8, 16]


def analyze_tif(path: str) -> dict:
    with rio.open(path) as ds:
        return {
            "path": path,
            "crs": str(ds.crs),
            "epsg": (ds.crs.to_epsg() if ds.crs else None),
            "width": ds.width,
            "height": ds.height,
            "bands": ds.count,
            "dtype": ds.dtypes[0],
            "tiled": _is_tiled(ds),
            "overviews": [ds.overviews(i + 1) for i in range(ds.count)],
            "categorical_guess": _is_categorical(ds),
        }


def _build_overviews_inplace(path: str, categorical: bool):
    resamp = "NEAREST" if categorical else "AVERAGE"
    with rio.open(path) as ds:
        levels = _target_overview_levels(ds.width, ds.height)
    # Rasterio can build in place too:
    try:
        with rio.open(path, "r+") as ds:
            ds.build_overviews(levels, resampling=resamp.lower())
            ds.update_tags(ns="rio_overview", resampling=resamp.lower())
    except Exception:
        # fallback to gdaladdo if rasterio fails
        if not shutil.which("gdaladdo"):
            raise
        cmd = [
            "gdaladdo",
            "-r",
            resamp,
            "--config",
            "COMPRESS_OVERVIEW",
            "DEFLATE",
            "--config",
            "PREDICTOR_OVERVIEW",
            "2",
            path,
            *map(str, levels),
        ]
        subprocess.run(cmd, check=True)


def _translate_to_cog(src: str, dst: str, resampling: str, block=512):
    cmd = [
        "gdal_translate",
        src,
        dst,
        "-of",
        "COG",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "LEVEL=6",
        "-co",
        "PREDICTOR=2",
        "-co",
        f"BLOCKSIZE={block}",
        "-co",
        "NUM_THREADS=ALL_CPUS",
        "-co",
        f"RESAMPLING={resampling}",
    ]
    subprocess.run(cmd, check=True)


def _warp_to_epsg(src: str, dst: str, epsg: int, resampling: str, block=512):
    cmd = [
        "gdalwarp",
        "-t_srs",
        f"EPSG:{epsg}",
        "-r",
        resampling,
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-co",
        "TILED=YES",
        "-co",
        f"BLOCKXSIZE={block}",
        "-co",
        f"BLOCKYSIZE={block}",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "PREDICTOR=2",
        "-co",
        "BIGTIFF=IF_SAFER",
        src,
        dst,
    ]
    subprocess.run(cmd, check=True)


def prepare_for_tiles(
    path: str,
    cache_dir: str | None = None,
    warp_to_3857: bool = False,
    force: bool = False,
) -> dict:
    """Returns dict: {"path": optimized_path, "report": analysis_dict}."""
    path = os.path.abspath(path)
    rep = analyze_tif(path)
    categorical = rep["categorical_guess"]
    resamp = "NEAREST" if categorical else "AVERAGE"
    need_reproj = _needs_reproject(rio.open(path), 3857) if warp_to_3857 else False
    good_enough = rep["tiled"] and _has_overviews(rio.open(path)) and not need_reproj

    if good_enough and not force:
        return {"path": path, "report": rep}

    cache_dir = cache_dir or os.path.join(pathlib.Path.home(), ".cache", "localtiles")
    os.makedirs(cache_dir, exist_ok=True)
    tag = _hash_for_cache(path)
    tmp_base = os.path.join(cache_dir, f"{os.path.basename(path)}.{tag}")

    if _gdal_ok():
        # Prefer building a clean COG (and reprojection if requested)
        out = tmp_base + (".3857.cog.tif" if warp_to_3857 else ".cog.tif")
        if warp_to_3857:
            # first warp to an intermediate tiled TIFF, then translate to COG
            inter = tmp_base + ".warp.tif"
            _warp_to_epsg(path, inter, 3857, resamp)
            _translate_to_cog(inter, out, resamp)
            try:
                os.remove(inter)
            except (OSError, PermissionError):
                pass
        else:
            _translate_to_cog(path, out, resamp)
        final_rep = analyze_tif(out)
        return {"path": out, "report": final_rep}
    else:
        # No GDAL CLI: do the minimum—build overviews in place or copy to temp and add overviews
        dst = tmp_base + ".ovr.tif"
        shutil.copy2(path, dst)
        _build_overviews_inplace(dst, categorical)
        final_rep = analyze_tif(dst)
        return {"path": dst, "report": final_rep}
