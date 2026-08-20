"""RasterMapWatcher waits for the class palette before drawing the raster.

The tiled COG and the class palette come from two independent async paths. When
the COG wins the race the palette is still empty, and adding the raster then
renders every class down the dark end of the default continuous ramp -- with no
error, and permanently, because the layer is never redrawn.
"""

import pytest
import solara

from component.model import app_state
from component.tile.upload import RasterMapWatcher

_COLORS = {1: "#00ff00", 2: "#ff0000"}


class _FakeMap:
    """Records add_raster calls instead of standing up a tile server."""

    def __init__(self):
        self.calls = []

    def add_raster(self, image, layer_name=None, key=None, class_colors=None, **kw):
        self.calls.append({"image": image, "class_colors": class_colors})

    def remove_layer(self, *args, **kwargs):
        pass


@pytest.fixture
def watcher():
    """Render the watcher, then unmount it.

    app_state is a process-wide singleton, so a watcher left mounted keeps
    observing it and services the *next* test's state change.
    """
    contexts = []

    def render(colors, map_=None):
        app_state.sampling_method.value = "stratified"
        app_state.optimized_raster_path.value = "/tmp/whatever.cog.tif"
        app_state.class_colors.value = colors
        app_state.raster_optimization_status.value = "adding_to_map"
        fake = map_ or _FakeMap()
        _, rc = solara.render(RasterMapWatcher(fake), handle_error=False)
        contexts.append(rc)
        return fake

    yield render
    for rc in contexts:
        rc.close()
    app_state.clear_file_data()


def test_raster_waits_for_the_palette(watcher):
    fake = watcher({})

    assert fake.calls == []
    # still pending, so the palette arriving re-runs the effect
    assert app_state.raster_optimization_status.value == "adding_to_map"


def test_raster_is_drawn_once_the_palette_lands(watcher):
    fake = watcher(_COLORS)

    assert [c["class_colors"] for c in fake.calls] == [_COLORS]
    assert app_state.raster_optimization_status.value == "finished"


def test_a_late_palette_still_reaches_the_map(watcher):
    # the race as it actually happens: COG first, palette second
    fake = watcher({})
    assert fake.calls == []

    app_state.class_colors.value = _COLORS

    assert [c["class_colors"] for c in fake.calls] == [_COLORS]


def test_clearing_a_file_drops_the_palette():
    # otherwise the gate passes on the previous file's colours and the next
    # raster is drawn in them
    app_state.class_colors.value = _COLORS

    app_state.clear_file_data()

    assert app_state.class_colors.value == {}
