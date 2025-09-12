from sepal_ui.model import Model
from traitlets import Int


class AppModel(Model):

    current_step = Int().tag(sync=True)
