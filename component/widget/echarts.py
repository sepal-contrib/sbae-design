import ipyvuetify as v
from ipecharts import EChartsWidget as BaseEChartsWidget


class EChartsWidget(BaseEChartsWidget):
    def __init__(self, theme_toggle=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.renderer = "svg"
        self.theme_toggle = theme_toggle
        self.theme = self.get_theme()

        if self.theme_toggle:
            self.theme_toggle.observe(self.set_theme, "dark")
        else:
            v.theme.observe(self.set_theme, "dark")

    def get_theme(self):

        obj = self.theme_toggle if self.theme_toggle else v.theme

        return "dark" if getattr(obj, "dark") else "light"

    def set_theme(self, _):
        self.theme = self.get_theme()
