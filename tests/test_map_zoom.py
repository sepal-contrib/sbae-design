"""SbaeMap floors zoom-out at level 5; SepalMap leaves min_zoom unset (leaflet's 0)."""

from pysepal.solara import ThemeState

from component.widget.map import SbaeMap


def test_map_defaults_to_min_zoom_five():
    assert SbaeMap(theme_state=ThemeState()).min_zoom == 5


def test_map_min_zoom_is_overridable():
    assert SbaeMap(theme_state=ThemeState(), min_zoom=0).min_zoom == 0
