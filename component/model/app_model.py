# sampling_design_tool/app_model.py

import os

import traitlets
from sepal_ui import model as sumodel

# pandas is implicitly needed if traitlets.Any will store DataFrames,
# but not directly used for type hinting here unless being more specific.


class AppModel(sumodel.Model):
    current_step = traitlets.Int(1).tag(sync=True)
    current_dir = traitlets.Unicode(os.getcwd()).tag(sync=True)
    selected_file = traitlets.Unicode(None, allow_none=True).tag(sync=True)
    file_path = traitlets.Unicode(None, allow_none=True).tag(sync=True)
    raw_area_df = traitlets.Any(None).tag(
        sync=True
    )  # Holds result from compute_map_area
    final_area_df = traitlets.Any(None).tag(
        sync=True
    )  # Holds area_df after class name edits
    sample_points_df = traitlets.Any(None).tag(
        sync=True
    )  # Holds generated sample points DF
    alert_message = traitlets.Unicode("Select a map file and click 'Load File'.").tag(
        sync=True
    )
    alert_type = traitlets.Unicode("info").tag(sync=True)
