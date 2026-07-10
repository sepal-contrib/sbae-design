"""The categorical colormap must show every sampleable class (incl. 0 and >255)."""

from component.widget.map import _build_class_colormap, _tile_client_kwargs_for_runtime


def test_colormap_colors_class_zero():
    cm = _build_class_colormap({0: "#ff0000", 5: "#00ff00"})
    assert cm[0] == (255, 0, 0, 255)  # class 0 is a real class, not background
    assert cm[5] == (0, 255, 0, 255)


def test_colormap_colors_codes_above_255():
    cm = _build_class_colormap({300: "#0000ff", 1024: "#ffffff"})
    assert cm[300] == (0, 0, 255, 255)
    assert cm[1024] == (255, 255, 255, 255)


def test_colormap_leaves_unknown_values_transparent():
    cm = _build_class_colormap({5: "#00ff00"})
    assert cm[7] == (0, 0, 0, 0)
    assert cm[5] == (0, 255, 0, 255)


def test_tile_client_kwargs_force_loopback_url_in_voila():
    assert _tile_client_kwargs_for_runtime(is_voila=True) == {
        "client_host": "127.0.0.1",
        "client_port": True,
        "cors_all": True,
    }


def test_tile_client_kwargs_keep_localtileserver_defaults_outside_voila():
    assert _tile_client_kwargs_for_runtime(is_voila=False) == {}


def test_tile_client_kwargs_uses_solara_voila_detector(monkeypatch):
    monkeypatch.setattr("component.widget.map.solara.util.is_running_in_voila", lambda: True)
    assert _tile_client_kwargs_for_runtime()["client_host"] == "127.0.0.1"
