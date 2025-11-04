from sepal_ui.model import Model
from traitlets import Int


class AppModel(Model):

    current_step = Int(allow_none=True).tag(sync=True)
