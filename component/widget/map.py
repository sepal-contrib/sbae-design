import logging
import shutil

import pandas as pd
import solara
from pysepal.mapping import SepalMap
from pysepal.scripts.scratch import scratch_dir
from pysepal.solara import ThemeState

from component.scripts.vector_tiles import (
    CORRECT_COLOR,
    INCORRECT_COLOR,
    REFERENCE_NEUTRAL_COLOR,
    SAMPLE_POINT_COLOR,
    VectorTileError,
    build_points_pmtiles_layer,
)

logger = logging.getLogger("sbae.map")

# On-map legend entries (label -> hex). Composed from whichever point layers are
# currently shown; see ``_compose_points_legend``.
_SAMPLE_LEGEND_LABEL = "Sample point"
_CORRECT_LEGEND_LABEL = "Correct"
_INCORRECT_LEGEND_LABEL = "Incorrect"
_REFERENCE_LEGEND_LABEL = "Reference point"


def _points_signature(df):
    """Cheap content signature of a points DataFrame (``None`` when empty).

    Lets ``add_reference_points`` skip the expensive tippecanoe rebuild when the
    incoming points are unchanged -- switching to the Analysis tab remounts its
    render task, which re-submits identical points on every entry.
    """
    if df is None or df.empty:
        return None
    return (
        len(df),
        tuple(str(c) for c in df.columns),
        int(pd.util.hash_pandas_object(df, index=False).sum()),
    )


def _compose_points_legend(
    has_sample: bool, has_reference: bool, reference_evaluated: bool
) -> dict:
    """Legend entries (label -> hex) for the point layers currently shown.

    Sample points contribute one neutral entry; reference points contribute the
    green/red correctness key once evaluated, else a single neutral entry.
    """
    legend = {}
    if has_sample:
        legend[_SAMPLE_LEGEND_LABEL] = SAMPLE_POINT_COLOR
    if has_reference:
        if reference_evaluated:
            legend[_CORRECT_LEGEND_LABEL] = CORRECT_COLOR
            legend[_INCORRECT_LEGEND_LABEL] = INCORRECT_COLOR
        else:
            legend[_REFERENCE_LEGEND_LABEL] = REFERENCE_NEUTRAL_COLOR
    return legend


class SbaeMap(SepalMap):
    """SBAE Map class extending SepalMap for map visualization and interactions."""

    def __init__(self, theme_state: ThemeState, gee: bool = False, min_zoom: int = 5):
        super().__init__(
            fullscreen=True, theme_state=theme_state, gee=gee, min_zoom=min_zoom
        )

        self.sample_points_layer = None
        self.sample_points_dir = None
        self._pending_points_dir = None
        # Analysis reference points -- a separate layer so they coexist with the
        # design sample points (distinct key/name/color; see add_reference_points).
        self.reference_points_layer = None
        self.reference_points_dir = None
        # Whether the reference points are coloured by correctness (green/red)
        # vs a single neutral colour -- drives the legend entries.
        self._reference_evaluated = False
        # Signature of the last-rendered reference points, to skip redundant
        # rebuilds when the same points are re-submitted (e.g. Analysis-tab
        # re-entry remounts the render task).
        self._reference_points_sig = None

    async def build_sample_points_layer(
        self,
        points_data,
        class_colors=None,
        *,
        point_color=None,
        color_field="map_code",
        radius=5,
        stroke_width=1,
    ):
        """Build (off the UI thread) a PMTiles layer for the sample points.

        Returns ``None`` for an empty DataFrame. Does NOT mutate the map, so it
        is safe to call from a worker thread. Raises ``VectorTileError`` on
        failure. Empty/omitted ``class_colors`` yields a single uniform
        ``point_color`` layer (design sample points); a non-empty
        ``class_colors`` colours one flat layer per ``color_field`` value (the
        analysis green/red).
        """
        if points_data is None or points_data.empty:
            return None
        dest_dir = str(scratch_dir(prefix="sbae_points_"))
        try:
            layer = await build_points_pmtiles_layer(
                points_data,
                class_colors or {},
                dest_dir=dest_dir,
                color_field=color_field,
                default_color=point_color or SAMPLE_POINT_COLOR,
                radius=radius,
                stroke_width=stroke_width,
            )
        except Exception:
            # Don't leak the dir when the build itself failed.
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        # Ride in leaflet's overlayPane (z 400) so points always draw above
        # raster layers: both the PMTiles points and the localtileserver rasters
        # are GridLayers, and rasters sit in the tilePane (z 200); without this a
        # classification map added after the points would cover them.
        try:
            layer.pane = "overlayPane"
        except Exception:
            pass
        self._pending_points_dir = dest_dir
        return layer

    def _zoom_to_points(self, layer):
        """Zoom to the points layer's own extent, when it has one.

        Layers built via ``vector_tiles`` carry a fit-ready ``.bounds``
        (``[[S,W],[N,E]]``); a single point yields ``None`` -- nothing sensible to
        fit -- and the view is left untouched.
        """
        bounds = getattr(layer, "bounds", None)
        if bounds:
            self.fit_bounds(bounds)

    def attach_sample_points_layer(self, layer):
        """Swap the sample-points layer on the map.

        Mutates the map, so the main thread is preferred. ``add_sample_points``
        deliberately calls this off the UI thread anyway, matching the
        pre-existing off-thread ``add_raster`` mutation in
        ``analysis_tab.py``.
        """
        old_dir = getattr(self, "sample_points_dir", None)
        if self.sample_points_layer is not None:
            self.remove_layer(self.sample_points_layer, none_ok=True)
            self.sample_points_layer = None
        self.sample_points_dir = None
        if layer is not None:
            self.add_layer(layer, key="sample_pts")
            self.sample_points_layer = layer
            self.sample_points_dir = getattr(self, "_pending_points_dir", None)
            SbaeMap._zoom_to_points(self, layer)
        self._pending_points_dir = None
        SbaeMap._refresh_points_legend(self)
        if old_dir:
            # Reclaim the previous layer's backing dir now that the swap is done.
            shutil.rmtree(old_dir, ignore_errors=True)

    async def add_sample_points(self, points_data):
        """Build + attach the design sample-points PMTiles layer (uniform colour).

        Called off the UI thread by the analysis derivation path -- see
        ``attach_sample_points_layer``'s docstring for why that's fine here.
        On failure, notify and skip -- the map shows no points layer; other
        results are unaffected.
        """
        from component.model import app_state

        # Explicit class dispatch (SbaeMap.foo(self, ...), not self.foo(...)) keeps
        # this method callable unbound against duck-typed test doubles; for a real
        # SbaeMap instance it behaves identically to self.foo(...).
        try:
            layer = await SbaeMap.build_sample_points_layer(
                self, points_data, point_color=SAMPLE_POINT_COLOR
            )
        except VectorTileError as e:
            logger.warning("Sample points layer failed: %s", e)
            app_state.add_error(f"Could not render sample points on the map: {e}")
            return
        SbaeMap.attach_sample_points_layer(self, layer)

    async def add_reference_points(self, points_data, *, layer_name="Reference points"):
        """Render the analysis reference points on their own layer, by agreement.

        Kept separate from the design sample (``add_sample_points`` / the
        ``"sample_pts"`` layer): distinct key (``"ref_pts"``) and name. When both
        the map and reference class are known, points are coloured green where
        they agree (correct) and red where they don't; otherwise (e.g. map class
        not sampled yet) they fall back to a single neutral colour. Builds off the
        UI thread; on failure, notify and skip.
        """
        from component.model import app_state

        # Unchanged points already on the map -> keep the existing layer and skip
        # the tippecanoe rebuild (see _points_signature).
        sig = _points_signature(points_data)
        if (
            sig is not None
            and sig == getattr(self, "_reference_points_sig", None)
            and self.reference_points_layer is not None
        ):
            return

        pts = points_data
        color_field = "map_code"
        class_colors = {}
        evaluated = False
        if {"map_code", "ref_code"} <= set(pts.columns) and bool(
            pts[["map_code", "ref_code"]].notna().all(axis=1).all()
        ):
            pts = pts.copy()
            pts["correct"] = (pts["map_code"] == pts["ref_code"]).astype(int)
            class_colors = {0: INCORRECT_COLOR, 1: CORRECT_COLOR}
            color_field = "correct"
            evaluated = True

        try:
            layer = await SbaeMap.build_sample_points_layer(
                self,
                pts,
                class_colors,
                point_color=REFERENCE_NEUTRAL_COLOR,
                color_field=color_field,
                radius=6,
            )
        except VectorTileError as e:
            logger.warning("Reference points layer failed: %s", e)
            app_state.add_error(f"Could not render reference points on the map: {e}")
            return
        old_dir = self.reference_points_dir
        if self.reference_points_layer is not None:
            self.remove_layer(self.reference_points_layer, none_ok=True)
            self.reference_points_layer = None
        self.reference_points_dir = None
        if layer is not None:
            try:
                layer.name = layer_name
            except Exception:
                pass
            self.add_layer(layer, key="ref_pts")
            self.reference_points_layer = layer
            self.reference_points_dir = getattr(self, "_pending_points_dir", None)
            self._reference_points_sig = sig
            SbaeMap._zoom_to_points(self, layer)
        self._reference_evaluated = evaluated and layer is not None
        self._pending_points_dir = None
        SbaeMap._refresh_points_legend(self)
        if old_dir:
            shutil.rmtree(old_dir, ignore_errors=True)

    def clear_reference_points(self):
        """Take the reference-points layer off the map and reclaim its temp dir.

        The inverse of the attach half of ``add_reference_points``: called when
        the reference data is cleared (or is no longer renderable) so the map
        layer doesn't outlive the data that produced it.
        """
        old_dir = self.reference_points_dir
        self.reference_points_layer = None
        self.reference_points_dir = None
        self._reference_evaluated = False
        self._reference_points_sig = None
        # Remove by the stable "ref_pts" key rather than the tracked object: the
        # map is shared and its widgets can be swapped out from under us (theme
        # change / re-style), leaving the handle stale so an identity lookup would
        # miss the still-visible layer and raise. none_ok: it may already be gone.
        self.remove_layer("ref_pts", none_ok=True)
        SbaeMap._refresh_points_legend(self)
        if old_dir:
            shutil.rmtree(old_dir, ignore_errors=True)

    def _refresh_points_legend(self):
        """Publish the on-map points legend to reactive state.

        The legend is a declarative Solara overlay (``PointsLegend`` ->
        pysepal ``LegendComponent``), so this just pushes ``{label: hex}`` to
        ``app_state.points_legend``; the component re-renders itself. Safe to call
        from the worker threads that mutate the map.
        """
        from component.model import app_state

        app_state.points_legend.value = _compose_points_legend(
            getattr(self, "sample_points_layer", None) is not None,
            getattr(self, "reference_points_layer", None) is not None,
            getattr(self, "_reference_evaluated", False),
        )


@solara.component
def PointsLegend():
    """Floating map legend for the sample/reference points.

    Renders the modern pysepal ``LegendComponent`` overlay, driven by
    ``app_state.points_legend`` ({label: hex}); hidden when there is nothing to
    show. Place it once in the page alongside the map.
    """
    from dataclasses import asdict

    from pysepal.solara.components.legend import (
        DiscreteEntry,
        LegendComponent,
        LegendData,
    )

    from component.model import app_state

    legend = app_state.points_legend.value or {}
    data = LegendData(
        items=[
            DiscreteEntry(label=label, color=color) for label, color in legend.items()
        ]
    )
    LegendComponent(legend_data=asdict(data), visible=bool(legend))
