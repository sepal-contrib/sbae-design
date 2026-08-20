# tests/test_echarts_theme.py
from component.widget.echarts import EChartsWidget, RawEChartsWidget


class _State:
    dark = True

    def observe(self, *_args, **_kwargs):
        pass


def test_raw_widget_binds_state_and_theme():
    tt = _State()
    w = RawEChartsWidget(theme_state=tt, option={"series": []})
    assert w.theme_state is tt
    assert w.theme == "dark"
    assert w.renderer == "svg"


def test_typed_widget_still_themes():
    tt = _State()
    w = EChartsWidget(theme_state=tt)
    assert w.theme == "dark"
    assert w.renderer == "svg"
