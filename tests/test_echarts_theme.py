# tests/test_echarts_theme.py
from component.widget.echarts import EChartsWidget, RawEChartsWidget


class _Toggle:
    dark = True

    def observe(self, *_args, **_kwargs):
        pass


def test_raw_widget_binds_toggle_and_theme():
    tt = _Toggle()
    w = RawEChartsWidget(theme_toggle=tt, option={"series": []})
    assert w.theme_toggle is tt
    assert w.theme == "dark"
    assert w.renderer == "svg"


def test_typed_widget_still_themes():
    tt = _Toggle()
    w = EChartsWidget(theme_toggle=tt)
    assert w.theme == "dark"
    assert w.renderer == "svg"
