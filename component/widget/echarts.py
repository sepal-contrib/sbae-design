import ipyvuetify as v
from ipecharts import EChartsRawWidget as BaseEChartsRawWidget
from ipecharts import EChartsWidget as BaseEChartsWidget


class _EChartsThemeMixin:
    """Shared light/dark theming for ipecharts widgets.

    Observes an optional ThemeToggle (or the global ``v.theme``) and keeps the
    echarts ``theme`` trait in sync. SVG renderer for crisp static output.
    """

    def _init_theme(self, theme_toggle):
        self.renderer = "svg"
        self.theme_toggle = theme_toggle
        self.theme = self.get_theme()
        target = self.theme_toggle if self.theme_toggle else v.theme
        target.observe(self.set_theme, "dark")

    def get_theme(self):
        obj = self.theme_toggle if self.theme_toggle else v.theme
        return "dark" if getattr(obj, "dark") else "light"

    def set_theme(self, _):
        self.theme = self.get_theme()


class EChartsWidget(BaseEChartsWidget, _EChartsThemeMixin):
    def __init__(self, theme_toggle=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_theme(theme_toggle)


class RawEChartsWidget(BaseEChartsRawWidget, _EChartsThemeMixin):
    def __init__(self, theme_toggle=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_theme(theme_toggle)
