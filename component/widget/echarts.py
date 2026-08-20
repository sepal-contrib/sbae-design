import ipyvuetify as v
from ipecharts import EChartsRawWidget as BaseEChartsRawWidget
from ipecharts import EChartsWidget as BaseEChartsWidget


class _EChartsThemeMixin:
    """Shared light/dark theming for ipecharts widgets.

    Observes an optional ThemeState (or the global ``v.theme``) and keeps the
    echarts ``theme`` trait in sync. SVG renderer for crisp static output.
    """

    def _init_theme(self, theme_state):
        self.renderer = "svg"
        self.theme_state = theme_state
        self.theme = self.get_theme()
        target = self.theme_state if self.theme_state else v.theme
        target.observe(self.set_theme, "dark")

    def get_theme(self):
        obj = self.theme_state if self.theme_state else v.theme
        return "dark" if getattr(obj, "dark") else "light"

    def set_theme(self, _):
        self.theme = self.get_theme()


class EChartsWidget(BaseEChartsWidget, _EChartsThemeMixin):
    def __init__(self, theme_state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_theme(theme_state)


class RawEChartsWidget(BaseEChartsRawWidget, _EChartsThemeMixin):
    def __init__(self, theme_state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_theme(theme_state)
