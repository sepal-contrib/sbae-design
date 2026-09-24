"""Classification raster source: acceptance codes and the VRT sidecar.

A classification map reaches the pipeline as a path whose band 1 is the class
band and whose nodata tag is the value to ignore. When the user picks another
band or nodata value, ``write_vrt`` re-describes the source through a VRT
instead of copying pixels, so every reader keeps opening ``file_path`` with
rasterio unchanged. The VRT references the source by absolute path: moving
the source breaks it, exactly as moving an uploaded file does today.
"""

import math
from typing import ClassVar, Optional


class NotThematicError(ValueError):
    """The raster cannot serve as a classification map.

    ``code`` is one of ``no_crs``, ``non_integral``, ``too_many_classes``,
    ``no_valid_data``; the UI maps it to a catalogue key. The message is plain
    English for logs and for callers that never reach the UI.
    """

    MESSAGES: ClassVar = {
        "no_crs": "The raster has no coordinate reference system",
        "non_integral": "The raster holds non-integer values",
        "too_many_classes": "The raster has too many distinct values for a class map",
        "no_valid_data": "No valid data found in raster",
    }

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        message = self.MESSAGES.get(code, code)
        super().__init__(f"{message}: {detail}" if detail else message)


def selection_reject_code(info: dict) -> Optional[str]:
    """Why a just-selected file cannot be the classification map, or ``None``.

    Decided from ``get_file_info`` metadata alone: ``read_error`` when the
    reader failed (``info["error"]`` holds its message), ``not_a_raster`` for
    vectors and unopenable files, ``no_crs`` when rasterio found no CRS.
    """
    if "error" in info:
        return "read_error"
    if info.get("file_type") != "raster":
        return "not_a_raster"
    if info.get("crs") is None:
        return "no_crs"
    return None


def parse_nodata(text) -> Optional[float]:
    """The nodata value a preview field holds: ``None`` for blank, else a float.

    Raises ``ValueError`` for text that is not a finite number; the confirm
    button stays disabled on that.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    value = float(text)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(text)
    return value


def format_nodata(value) -> str:
    """Prefill text for the nodata field: blank for none (or NaN), ``0`` not ``0.0``."""
    value = _normalize_nodata(value)
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _normalize_nodata(value) -> Optional[float]:
    """``None`` for an absent or NaN nodata, else the value as a float.

    NaN can never be a class code, so ``compute_area_from_raster`` always
    drops it; declaring it adds nothing, which is why it counts as absent here.
    """
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def needs_vrt(info: dict, band: int, nodata: Optional[float]) -> bool:
    """Whether the chosen band or nodata differ from what the file declares."""
    if int(band) != 1:
        return True
    return _normalize_nodata(info.get("nodata")) != _normalize_nodata(nodata)
