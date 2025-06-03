# component/tile/app_controller.py

import html
import os
import traceback # Keep for direct use if needed, logger.exception is preferred
from pathlib import Path
import logging # Added

import ipyvuetify as v
import ipywidgets as widgets
import numpy as np
import pandas as pd
import sepal_ui.sepalwidgets as sui
import traitlets
from ipyleaflet import (
    CircleMarker, FullScreenControl, GeoJSON, ImageOverlay, LayerGroup,
    LayersControl, Map, ScaleControl, TileLayer, ZoomControl, basemaps,
)

# Local module imports
from component.model.app_model import AppModel
from component.scripts import map_utils # Import map_utils and then access its attributes
from component.scripts import processing as proc # Ensures processing is imported
# Import the new logger configuration setup
from component.logger_config import setup_logging, LOG_APP_NAME # Added

# Ensures map_utils functions are available (already imported via map_utils)
from component.scripts.map_utils import (_add_overlay_layer, generate_class_color_map)
from component.scripts.map_utils import target_crs_epsg as map_utils_target_crs_epsg # Constant
from component.scripts.processing import compute_map_area, get_output_dir # Specific functions

# Configure logging for the application
# This should ideally be called once. If AppController is re-instantiated in Jupyter,
# the setup_logging function has a clear_handlers mechanism.
# Pass self.model.current_dir so logs can be relative to the app's starting CWD if desired,
# or keep it default to save logs in the script's CWD/logs.
# For simplicity, using default base_dir (os.getcwd()) for setup_logging here.
# You can change log_to_console to True if you want to see logs in console as well.
setup_logging(log_to_console=False) # Or True for console output

# Get a logger for this controller module, child of the main app logger
logger = logging.getLogger(f"{LOG_APP_NAME}.{__name__}")


# Optional dependency flags from submodules for convenience in controller
# These flags are set within their respective modules after attempting imports.
_rasterio_available_map = getattr(map_utils, "_rasterio_available", False)
_rasterio_available_proc = getattr(proc, "_rasterio_available", False)
_geopandas_available_proc = getattr(proc, "_geopandas_available", False)
_pyproj_available_proc = getattr(proc, "_pyproj_available", False)

# For direct use of rasterio and gpd in handlers if needed (though checks are preferred)
try:
    import rasterio
    import rasterio.warp
except ImportError:
    pass # Failures are logged by map_utils or proc

try:
    import geopandas as gpd
except ImportError:
    pass # Failures are logged by proc


logger.info("SEPAL-UI Application components and Backend Functions (AppController) loading...")
logger.debug(f"Controller Top Level Flags: _rasterio_available_map = {_rasterio_available_map}, _pyproj_available_proc = {_pyproj_available_proc}")


class AppController:
    def __init__(self, model: AppModel):
        self.model = model
        logger.info("Initializing AppController...")
        self.sample_points_layer = None # To keep track of the sample points layer on the map
        self.base_map_overlay_layer = None # To keep track of the loaded map overlay
        self.calculated_total_sample_size_display = None # Initialize attribute
        self.per_class_sample_size_display = None # Initialize attribute for per-class display
        self.per_class_sample_size_display_card_text = None # Initialize attribute for card text of per-class display


        self._create_map_widget()
        self._create_stage2_widgets() # Placeholders for class name/UA widgets
        self._create_stage3_widgets() # Parameter widgets for sampling design
        self._create_stepper()        # Main UI navigation

        self.app = v.Card(
            class_="pa-4",
            children=[v.CardTitle(children=["Sampling Design Tool"]), self.stepper],
        )
        self.model.observe(self._update_stepper_value, names="current_step")
        
        if hasattr(self, "objective_widget"): # Ensure widget exists before observing
            self.objective_widget.observe(self._handle_objective_change, names="v_model")
        else:
            logger.warning("objective_widget not created at observer setup time in AppController __init__.")
        
        logger.debug("AppController initialized.")

    def _create_map_widget(self):
        logger.debug("Creating ipyleaflet.Map widget...")
        try:
            self.map_widget = Map(
                center=[0, 0], zoom=2, scroll_wheel_zoom=True,
                zoom_control=False, # We add it manually for placement
                layout={"height": "450px", "border": "1px solid #ccc"},
            )
            self.map_widget.add_layer(
                TileLayer(
                    url=basemaps.OpenStreetMap.Mapnik.url, name="OpenStreetMap",
                    base=True, attribution=basemaps.OpenStreetMap.Mapnik.attribution,
                )
            )
            self.map_widget.add_control(LayersControl(position="topright"))
            self.map_widget.add_control(ScaleControl(position="bottomleft"))
            self.map_widget.add_control(ZoomControl(position="topleft"))
            self.map_widget.add_control(FullScreenControl())
            logger.debug("ipyleaflet.Map created with default controls.")
        except Exception as map_init_err:
            logger.exception("Error creating ipyleaflet Map widget:")
            # Fallback UI if map fails to initialize
            self.map_widget = widgets.HTML(
                "<p style='color:red; font-weight:bold;'>Error initializing map widget. Check logs.</p>"
            )

    def _create_stage2_widgets(self):
        logger.debug("Creating Stage 2 control widget holders (placeholders).")
        self.class_name_widgets = {} # Dict to store TextFields for class names
        self.expected_ua_sliders = {} # Dict to store Sliders for expected user accuracy
        self.stage2_alert = sui.Alert().hide() # Alert for feedback in Stage 2
        self.stage2_submit_btn = sui.Btn(
            "Submit Classes & Proceed to Stage 3", icon="mdi-check",
            class_="ma-2", disabled=True, # Initially disabled
        )

    def _create_stage3_widgets(self):
        logger.debug("Creating Stage 3 control widgets for sampling parameters.")
        self.objective_widget = sui.Select(
            label="Calculation Objective", dense=True, outlined=True, class_="mb-2", clearable=False,
            items=["Overall Accuracy", "Target Class Precision"], v_model="Overall Accuracy",
        )
        self.target_class_widget = sui.Select(
            label="Target Class (for Precision Objective)", dense=True, outlined=True, class_="mb-2",
            items=[], v_model=None, clearable=True, disabled=True, # Populated later
        )
        self.target_class_allowable_error_widget = v.Slider(
            label="Target Class Allowable Error (Ej)", min=0.01, max=0.25, step=0.001,
            v_model=0.05, thumb_label="always", class_="mt-2 mb-2", disabled=True,
        )
        self.target_overall_accuracy_widget = v.Slider(
            label="Target Overall Accuracy (OA)", min=0.01, max=0.99, step=0.01,
            v_model=0.90, thumb_label="always", class_="mt-2 mb-2",
        )
        self.allowable_error_widget = v.Slider(
            label="Allowable Error (Overall)", min=0.001, max=0.10, step=0.001,
            v_model=0.05, thumb_label="always", class_="mt-2 mb-2",
        )
        self.confidence_level_widget = v.Slider(
            label="Confidence Level", min=0.80, max=0.999, step=0.001, # Max < 1 for Z-score calc
            v_model=0.95, thumb_label="always", class_="mt-2 mb-2",
        )
        self.min_samples_per_class_widget = v.Slider(
            label="Min Samples per Class", min=5, max=500, step=5,
            v_model=20, thumb_label="always", class_="mt-2 mb-2",
        )
        self.allocation_method_widget = sui.Select(
            label="Allocation Method", dense=True, outlined=True, class_="mb-2", clearable=False,
            items=["Proportional", "Neyman", "Equal"], v_model="Proportional",
        )
        self.calculated_total_sample_size_display = v.Html(
            tag="h4", # Or v.Chip, sui.ListItem, etc.
            class_="mt-3 mb-2 text-center success--text", # Vuetify classes for styling
            children=["Calculated Total Sample Size: N/A"]
        )
        self.calculated_total_sample_size_display.hide() # Initially hidden
        
        # NEW WIDGET FOR PER-CLASS SAMPLE SIZES
        self.per_class_sample_size_display_card_text = v.CardText(children=["N/A"], class_="pa-1") # Content holder
        self.per_class_sample_size_display = v.Card(
            outlined=True,
            class_="pa-2 mt-2 mb-2",
            children=[
                v.CardTitle(children=["Final Samples per Class"], class_="subtitle-1 pa-1 grey"), # Styled title
                self.per_class_sample_size_display_card_text 
            ]
        )
        self.per_class_sample_size_display.hide() # Initially hidden
        
        self.stage3_calculate_btn = sui.Btn(
            "Calculate Sample & Generate Design", icon="mdi-cogs", class_="ma-2", disabled=True,
        )
        self.update_map_btn = sui.Btn( # To show/refresh sample points on map
            "Show/Update Samples on Map", icon="mdi-map-marker-check", class_="ma-2", disabled=True,
        )
        self.show_map_overlay_btn = sui.Btn( # To show/refresh base map overlay
            "Show/Update Map Overlay", icon="mdi-layers-refresh", class_="ma-2", disabled=True,
        )
        self.stage3_output = sui.Alert(class_="mt-2 pa-2", type="info").hide() # For calculation feedback

    def _create_stepper(self):
        logger.debug("Creating Stepper UI structure.")
        self.step_1_title_text = "1. Select File & Load" 
        self.step_2_title_text = "2. Edit Classes & Accuracy" 
        self.step_3_title_text = "3. Sample Design & Results" 

        self.stepper_step_1 = v.StepperStep(step=1, complete=False, children=[self.step_1_title_text]) 
        self.stepper_step_2 = v.StepperStep(step=2, complete=False, children=[self.step_2_title_text]) 
        self.stepper_step_3 = v.StepperStep(step=3, complete=False, children=[self.step_3_title_text]) 

        try:
            step1_content_ui = self._create_step_1_content() 
        except Exception as e_step1:
            logger.critical(f"FATAL ERROR creating Step 1 UI content: {e_step1}", exc_info=True)
            step1_content_ui = [v.Html(tag="p", class_="error--text", children=[f"Error creating UI for Step 1: {e_step1}. Check logs."])]

        self.stepper_content_1 = v.StepperContent(step=1, children=step1_content_ui) 
        self.stepper_content_2 = v.StepperContent(step=2, children=[v.ProgressCircular(indeterminate=True, class_="ma-4")]) 
        self.stepper_content_3 = v.StepperContent(step=3, children=[v.ProgressCircular(indeterminate=True, class_="ma-4")]) 

        self.stepper = v.Stepper(
            class_="mb-5", v_model=self.model.current_step, # Links to model's current_step
            children=[
                v.StepperHeader(children=[self.stepper_step_1, self.stepper_step_2, self.stepper_step_3]),
                v.StepperItems(children=[self.stepper_content_1, self.stepper_content_2, self.stepper_content_3]),
            ],
        )
        logger.debug("Stepper UI structure created.")

    def _create_step_1_content(self):
        logger.debug("Creating Step 1 UI Content (File Input, Load Button, Map).")
        self.file_input = sui.FileInput(
            label="Select Map File (.tif, .shp, .geojson, etc.)",
            folder=self.model.current_dir, 
            extensions=[ 
                ".tif", ".tiff", ".img", ".pix", ".rst", ".grd", ".vrt", ".hdf", ".h5", ".jpeg2000", # Rasters
                ".shp", ".sqlite", ".gdb", ".geojson", ".json", ".gml", ".kml", ".tab", ".mif", # Vectors
            ]
        )
        traitlets.link((self.file_input, "file"), (self.model, "selected_file")) 

        self.load_btn = sui.Btn("Load File, Compute Area & Display Map", icon="mdi-upload", class_="ma-2")
        self.alert_step1 = sui.Alert(
            children=[str(self.model.alert_message)], type=self.model.alert_type
        ).show() 

        content_layout_step1 = v.Layout( 
            column=True,
            children=[
                self.file_input,
                v.Layout(row=True, class_="d-flex justify-center", children=[self.load_btn]), 
                self.alert_step1,
                self.map_widget, 
            ],
        )
        self.load_btn.on_event("click", self._on_load_click) 
        logger.debug("Step 1 UI Content created and event handlers attached.")
        return [content_layout_step1]

    def _create_step_2_content(self):
        logger.debug("Building dynamic Stage 2 UI Content (Class Edits, UA Sliders).")
        self.class_name_widgets.clear() 
        self.expected_ua_sliders.clear()

        if self.model.raw_area_df is None or self.model.raw_area_df.empty:
            logger.error("Cannot create Stage 2 content: No raw area data (raw_area_df) loaded from Step 1.")
            return [v.Html(tag="p", class_="warning--text", children=["Error: No area data loaded from Step 1. Please go back and load a file."])]

        area_df_sorted_stage2 = self.model.raw_area_df.copy().sort_values("map_code").reset_index(drop=True) 

        name_editor_items = [v.ListItemGroup(children=[ 
            v.Html(class_="pl-4 pt-2", tag="b", children=["Edit Class Names (Optional):"])
        ])]
        ua_slider_items = [v.ListItemGroup(children=[ 
            v.Html(class_="pl-4 pt-2", tag="b", children=["Expected User's Accuracy (for Neyman Allocation):"])
        ])]

        for _, row_data in area_df_sorted_stage2.iterrows(): 
            map_code_val = row_data["map_code"]
            initial_display_name = str(row_data.get("map_edited_class", map_code_val)) 

            name_tf = v.TextField( 
                label=f"Name for Code {map_code_val}", v_model=initial_display_name,
                outlined=True, dense=True, class_="ma-2",
            )
            self.class_name_widgets[map_code_val] = name_tf
            name_editor_items.append(v.ListItem(children=[name_tf], class_="ma-0 pa-0"))

            slider_dynamic_label = f"'{name_tf.v_model}' (Code: {map_code_val})" 
            ua_slider = v.Slider( 
                label=slider_dynamic_label, v_model=0.80, 
                min=0.01, max=0.99, step=0.01, thumb_label="always", class_="ma-2",
            )
            self.expected_ua_sliders[map_code_val] = ua_slider
            ua_slider_items.append(v.ListItem(children=[ua_slider], class_="ma-0 pa-0"))

            def _create_ua_label_observer(slider_widget, text_field_widget, code_value): 
                def _observer_func(change): 
                    new_name_val = text_field_widget.v_model.strip() if text_field_widget.v_model else f"Code {code_value}" 
                    slider_widget.label = f"'{new_name_val}' (Code: {code_value})"
                return _observer_func
            
            name_tf.observe(_create_ua_label_observer(ua_slider, name_tf, map_code_val), names="v_model")

        edit_names_card = v.Card(outlined=True, class_="mb-4", children=[v.List(dense=True, children=name_editor_items)]) 
        edit_ua_card = v.Card(outlined=True, class_="mb-4", children=[v.List(dense=True, children=ua_slider_items)]) 
        
        self.stage2_submit_btn.disabled = False 
        self.stage2_alert.reset().hide() 

        try: self.stage2_submit_btn.on_event("click", None, remove=True)
        except Exception as e_remove: logger.debug(f"Error removing previous Stage 2 submit handler (non-critical): {e_remove}")
        self.stage2_submit_btn.on_event("click", self._on_stage2_submit)

        content_layout_stage2 = v.Layout( 
            column=True,
            children=[
                v.Html(tag="h3", children=["Stage 2: Review Classes & Set Accuracies"]),
                self.stage2_alert, edit_names_card, edit_ua_card, self.stage2_submit_btn,
            ],
        )
        logger.debug("Stage 2 UI Content created and populated.")
        return [content_layout_stage2]

    def _create_step_3_content(self):
        logger.debug("Creating Step 3 UI Content (Parameters, Calculation, Results Map).")
        if self.model.final_area_df is None or self.model.final_area_df.empty:
            logger.error("Cannot create Stage 3 content: Finalized class data (final_area_df) is not available from Step 2.")
            return [v.Html(tag="p", class_="warning--text", children=["Error: Finalized class data not available. Please complete Step 2."])]

        class_summary_sheet = v.Sheet( 
            outlined=True, class_="pa-2 mb-4",
            style_="max-height: 150px; overflow-y: auto; background-color: #f5f5f5;", 
            children=[
                v.Html(tag="b", children=["Final Class Summary (Map Code, Name, Area):"]),
                v.Html(tag="pre", style_="color: black; font-family: monospace;", 
                       children=[self.model.final_area_df[["map_code", "map_edited_class", "map_area"]]
                                 .round(2).to_string(index=False)]),
            ]
        )
        
        params_column_layout = v.Layout( 
            column=True, class_="mb-2",
            children=[
                self.objective_widget, self.target_class_widget, self.target_class_allowable_error_widget,
                v.Divider(class_="my-2"),
                v.Html(tag="i", children=["Standard Sampling Parameters:"]),
                self.target_overall_accuracy_widget, self.allowable_error_widget,
                self.confidence_level_widget, self.min_samples_per_class_widget,
                self.allocation_method_widget,
            ]
        )
        self._handle_objective_change({"new": self.objective_widget.v_model}) 

        self.stage3_output.reset().hide() 
        self.stage3_calculate_btn.disabled = False 
        self.update_map_btn.disabled = True 
        self.show_map_overlay_btn.disabled = not (self.model.file_path and self.base_map_overlay_layer is not None)


        try: self.stage3_calculate_btn.on_event("click", None, remove=True)
        except Exception as e_remove_calc: logger.debug(f"Error removing calc handler: {e_remove_calc}")
        self.stage3_calculate_btn.on_event("click", self._on_stage3_calculate)

        try: self.update_map_btn.on_event("click", None, remove=True)
        except Exception as e_remove_map: logger.debug(f"Error removing map update handler: {e_remove_map}")
        self.update_map_btn.on_event("click", self._on_update_map_click)
        
        try: self.show_map_overlay_btn.on_event("click", None, remove=True)
        except Exception as e_remove_overlay: logger.debug(f"Error removing show map overlay handler: {e_remove_overlay}")
        self.show_map_overlay_btn.on_event("click", self._on_show_map_overlay_click)

        if not hasattr(self, 'calculated_total_sample_size_display') or self.calculated_total_sample_size_display is None:
            self._create_stage3_widgets() 
        if not hasattr(self, 'per_class_sample_size_display') or self.per_class_sample_size_display is None:
             self._create_stage3_widgets() # Ensure it's created if this method is somehow called out of order

        content_layout_stage3 = v.Layout(
            column=True,
            children=[
                v.Html(tag="h3", children=["Stage 3: Set Sample Design Parameters & Calculate"]),
                class_summary_sheet, 
                v.Card(outlined=True, class_="pa-4 mb-4", children=[params_column_layout]),
                
                self.calculated_total_sample_size_display, 
                self.per_class_sample_size_display,       # ADDED PER-CLASS DISPLAY
                
                v.Layout(row=True, class_="d-flex justify-center flex-wrap", children=[
                    self.stage3_calculate_btn, self.update_map_btn, self.show_map_overlay_btn,
                ]),
                self.stage3_output, 
                v.Divider(class_="my-4"),
                v.Html(tag="h4", children=["Map Visualization"], class_="mb-2"),
                self.map_widget,
            ],
        )
        logger.debug("Step 3 UI Content created with sample size display placeholder.")
        return [content_layout_stage3]

    def _update_stepper_value(self, change): 
        if hasattr(self, "stepper") and self.stepper:
            new_step_val = change["new"]
            self.stepper.v_model = new_step_val
            logger.debug(f"Stepper value model updated to: {new_step_val}")
            
            if new_step_val == 3 and hasattr(self, "map_widget") and self.map_widget:
                try:
                    logger.debug("Step 3 activated, invalidating map size.")
                    self.map_widget.invalidate_size()
                except Exception as e:
                    logger.error(f"Error invalidating map size on step 3 activation: {e}")
        else:
            logger.warning("Stepper widget not available when trying to update its value.")

    def _handle_objective_change(self, change): 
        new_objective_val = change.get("new") 
        is_target_class_active = new_objective_val == "Target Class Precision" 
        
        can_enable_target_specifics = (
            is_target_class_active and
            hasattr(self.target_class_widget, "items") and
            len(self.target_class_widget.items) > 0 and
            self.target_class_widget.items[0].get("value") is not None 
        )

        self.target_class_widget.disabled = not can_enable_target_specifics
        self.target_class_allowable_error_widget.disabled = not is_target_class_active 
        
        logger.debug(f"Objective changed to '{new_objective_val}'. Target class specific widgets enabled: {not self.target_class_widget.disabled}, Allowable error enabled: {not self.target_class_allowable_error_widget.disabled}")

    def _update_target_class_options(self):
        logger.debug("Updating target class selector options based on final_area_df.")
        target_select_widget = self.target_class_widget 
        
        if (self.model.final_area_df is not None and
            isinstance(self.model.final_area_df, pd.DataFrame) and
            not self.model.final_area_df.empty):
            try:
                sorted_final_df = (
                    self.model.final_area_df[["map_code", "map_edited_class"]]
                    .drop_duplicates()
                    .sort_values("map_edited_class")
                )
                class_choices = [ 
                    {"text": f"{row['map_edited_class']} (Code: {row['map_code']})", "value": row["map_code"]}
                    for _, row in sorted_final_df.iterrows()
                ]
                if not class_choices: 
                    class_choices = [{"text": "No classes eligible for targeting", "value": None}]
                
                target_select_widget.items = class_choices
                target_select_widget.v_model = (
                    None if not class_choices or class_choices[0].get("value") is None
                    else class_choices[0]["value"] 
                )
                logger.debug(f"Updated target class items (count: {len(class_choices)}). Default selection: {target_select_widget.v_model}")
            except Exception as e_format:
                logger.exception("Error formatting target class choices for dropdown:")
                target_select_widget.items = [{"text": "Error loading classes", "value": None}]
                target_select_widget.v_model = None
        else:
            logger.warning("final_area_df is not available or empty. Setting target class options to 'No classes available'.")
            target_select_widget.items = [{"text": "No classes available", "value": None}]
            target_select_widget.v_model = None
        
        self._handle_objective_change({"new": self.objective_widget.v_model})


    def _on_load_click(self, widget, event, data):
        logger.info("Load button clicked. Initiating file processing and map display.")
        self.model.alert_type = "info"
        selected_filename_str = os.path.basename(str(self.model.selected_file)) if self.model.selected_file else "file" 
        self.model.alert_message = f"Loading {selected_filename_str} and computing areas..."
        self.alert_step1.type = self.model.alert_type
        self.alert_step1.children = [self.model.alert_message]
        widget.loading = True 

        self.model.file_path = None
        self.model.raw_area_df = None
        self.model.final_area_df = None 
        self.model.sample_points_df = None
        if hasattr(self, "stepper_step_1"): self.stepper_step_1.complete = False
        if hasattr(self, "stepper_step_2"): self.stepper_step_2.complete = False
        if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = False

        layers_to_remove_on_load = [] 
        if self.sample_points_layer: layers_to_remove_on_load.append(self.sample_points_layer)
        if self.base_map_overlay_layer: layers_to_remove_on_load.append(self.base_map_overlay_layer)

        for layer_obj in layers_to_remove_on_load: 
            try:
                if layer_obj in self.map_widget.layers:
                    self.map_widget.remove_layer(layer_obj)
                    logger.debug(f"Removed tracked map layer: {getattr(layer_obj, 'name', 'Unnamed Layer')}")
            except Exception as e_remove:
                logger.warning(f"Could not remove tracked layer '{getattr(layer_obj, 'name', 'Unnamed Layer')}': {e_remove}")
        self.sample_points_layer = None 
        self.base_map_overlay_layer = None
        
        current_map_layers = list(self.map_widget.layers) 
        for lyr in current_map_layers:
            if isinstance(lyr, (ImageOverlay, GeoJSON, LayerGroup)) and getattr(lyr, "base", False) is False:
                if lyr not in layers_to_remove_on_load:
                    try:
                        self.map_widget.remove_layer(lyr)
                        logger.debug(f"Removed other non-basemap layer: {getattr(lyr, 'name', 'Unknown Overlay')}")
                    except Exception as e_remove_other:
                        logger.warning(f"Could not remove other layer '{getattr(lyr, 'name', 'Unknown Overlay')}': {e_remove_other}")

        if hasattr(self, "update_map_btn"): self.update_map_btn.disabled = True
        if hasattr(self, "show_map_overlay_btn"): self.show_map_overlay_btn.disabled = True

        try:
            selected_file_path = self.model.selected_file 
            if not (selected_file_path and Path(selected_file_path).exists()):
                raise ValueError("Please select a valid file first.")
            
            file_path_obj_load = Path(selected_file_path) 
            basename_load = file_path_obj_load.name 
            
            raster_exts = (".tif", ".tiff", ".img", ".pix", ".rst", ".grd", ".vrt", ".hdf", ".h5", ".jpeg2000")
            vector_exts = (".shp", ".sqlite", ".gdb", ".geojson", ".json", ".gml", ".kml", ".tab", ".mif")
            is_raster_file = file_path_obj_load.suffix.lower() in raster_exts 
            is_vector_file = file_path_obj_load.suffix.lower() in vector_exts 

            logger.info(f"Selected file: {selected_file_path}. Type: {'Raster' if is_raster_file else 'Vector' if is_vector_file else 'Unknown'}")
            logger.debug(f"Dependency flags before map load: Rasterio (map_utils): {_rasterio_available_map}, PyProj (processing): {_pyproj_available_proc}, GeoPandas (processing): {_geopandas_available_proc}")


            area_data_computed = compute_map_area(selected_file_path) 
            if not (area_data_computed is not None and isinstance(area_data_computed, pd.DataFrame) and not area_data_computed.empty):
                raise ValueError("Area computation failed or returned empty. Cannot proceed.")
            
            if "map_edited_class" not in area_data_computed.columns:
                area_data_computed["map_edited_class"] = area_data_computed["map_code"].astype(str)
            
            self.model.file_path = selected_file_path 
            self.model.raw_area_df = area_data_computed 

            new_overlay = None 
            calculated_bounds_for_raster = None 
            loaded_vector_gdf = None 

            if is_raster_file:
                logger.debug("Attempting to load raster for map overlay...")
                if _rasterio_available_map and _pyproj_available_proc: 
                    try:
                        with rasterio.open(selected_file_path) as src_raster_ds: 
                            if src_raster_ds.crs: 
                                dst_bounds_transformed = rasterio.warp.transform_bounds( 
                                    src_raster_ds.crs, f"EPSG:{map_utils_target_crs_epsg}", *src_raster_ds.bounds
                                )
                                calculated_bounds_for_raster = [
                                    [dst_bounds_transformed[1], dst_bounds_transformed[0]],
                                    [dst_bounds_transformed[3], dst_bounds_transformed[2]],
                                ]
                                logger.debug(f"Calculated EPSG:4326 bounds for raster overlay: {calculated_bounds_for_raster}")
                            else:
                                logger.warning("Raster missing CRS. Cannot calculate accurate bounds for map overlay.")
                    except Exception as bounds_calc_err:
                        logger.error(f"Error calculating raster bounds for map overlay: {bounds_calc_err}", exc_info=True)
                else:
                    logger.warning("Rasterio or PyProj (or both) are unavailable. Cannot calculate raster bounds for overlay.")
                
                if calculated_bounds_for_raster: 
                    new_overlay = _add_overlay_layer(
                        self.map_widget, selected_file_path, self.model.raw_area_df,
                        calculated_bounds_for_raster, None 
                    )
                else:
                    logger.warning("Skipping raster overlay on map due to missing bounds or dependencies.")

            elif is_vector_file:
                logger.debug("Attempting to load vector for map overlay...")
                if _geopandas_available_proc: 
                    try:
                        gdf_temp = gpd.read_file(selected_file_path) 
                        if not gdf_temp.empty:
                            if gdf_temp.crs is None:
                                logger.warning(f"Vector file '{basename_load}' has no CRS defined. Assuming EPSG:{map_utils_target_crs_epsg} for display, but this might be incorrect.")
                            loaded_vector_gdf = gdf_temp.to_crs(epsg=map_utils_target_crs_epsg)
                            logger.debug(f"Loaded and reprojected vector '{basename_load}' to EPSG:{map_utils_target_crs_epsg} for overlay.")
                        else:
                            logger.warning(f"Vector file '{basename_load}' is empty.")
                    except Exception as gdf_load_err:
                        logger.error(f"Error loading/reprojecting vector '{basename_load}': {gdf_load_err}", exc_info=True)
                else:
                    logger.warning("GeoPandas is unavailable. Cannot process vector file for map overlay.")

                if loaded_vector_gdf is not None: 
                    new_overlay = _add_overlay_layer(
                        self.map_widget, selected_file_path, self.model.raw_area_df,
                        None, loaded_vector_gdf 
                    )
                else:
                    logger.warning("Skipping vector overlay on map due to loading error, empty GDF, or missing GeoPandas.")
            
            if new_overlay:
                self.base_map_overlay_layer = new_overlay 
                if hasattr(self, "show_map_overlay_btn"): self.show_map_overlay_btn.disabled = False
                logger.info(f"Map overlay '{getattr(new_overlay, 'name', 'Unnamed Overlay')}' added to map and tracked.")
                if hasattr(new_overlay, 'bounds'): 
                    self.map_widget.fit_bounds(new_overlay.bounds)
                elif loaded_vector_gdf is not None and not loaded_vector_gdf.empty: 
                    try:
                        map_bounds_vec = loaded_vector_gdf.total_bounds 
                        self.map_widget.fit_bounds([
                            [map_bounds_vec[1], map_bounds_vec[0]], 
                            [map_bounds_vec[3], map_bounds_vec[2]]  
                        ])
                        logger.debug(f"Map zoomed to vector overlay bounds: {map_bounds_vec.tolist()}")
                    except Exception as fit_err:
                        logger.warning(f"Could not automatically fit map to vector overlay bounds: {fit_err}")


            self.model.alert_type = "success"
            self.model.alert_message = f"Successfully loaded '{basename_load}'. Area data computed. Overlay status: {'Added' if new_overlay else 'Not Added'}. Proceed to Step 2."
            if hasattr(self, "stepper_step_1"): self.stepper_step_1.complete = True
            
            self.stepper_content_2.children = self._create_step_2_content()
            self.model.current_step = 2 

        except (ValueError, FileNotFoundError, AssertionError) as val_err: 
            logger.error(f"Error during file loading or initial processing: {val_err}", exc_info=True)
            self.model.alert_type = "error"
            self.model.alert_message = f"Error processing file: {val_err}"
            if hasattr(self, "stepper_step_1"): self.stepper_step_1.complete = False
        except ImportError as imp_err: 
            logger.critical(f"Import error during file load, a critical library might be missing: {imp_err}", exc_info=True)
            self.model.alert_type = "error"
            self.model.alert_message = f"Critical library missing: {imp_err}. Cannot proceed."
            if hasattr(self, "stepper_step_1"): self.stepper_step_1.complete = False
        except Exception as e_unexp: 
            logger.exception("An unexpected error occurred during file load and map display:")
            self.model.alert_type = "error"
            self.model.alert_message = f"An unexpected error occurred: {e_unexp}. Check logs for details."
            if hasattr(self, "stepper_step_1"): self.stepper_step_1.complete = False
        finally:
            self.alert_step1.type = self.model.alert_type
            self.alert_step1.children = [self.model.alert_message]
            widget.loading = False 

    def _on_stage2_submit(self, widget, event, data):
        logger.info("Stage 2 Submit button clicked. Processing class name edits and UA inputs.")
        widget.loading = True
        self.stage2_alert.reset().show() 
        self.stage2_alert.type = "info"
        self.stage2_alert.children = ["Processing class edits and user accuracies..."]

        try:
            if not hasattr(self, "class_name_widgets") or not hasattr(self, "expected_ua_sliders"):
                raise AttributeError("Stage 2 UI widgets (class_name_widgets or expected_ua_sliders) not properly initialized.")
            if self.model.raw_area_df is None or self.model.raw_area_df.empty:
                raise ValueError("Raw area data (raw_area_df from Step 1) is missing or empty. Cannot proceed.")

            edited_class_names_map = { 
                code: (name_widget.v_model.strip() if name_widget.v_model and name_widget.v_model.strip() else str(code))
                for code, name_widget in self.class_name_widgets.items()
            }
            logger.debug(f"Edited class names map: {edited_class_names_map}")

            if not self.expected_ua_sliders: 
                logger.warning("Expected UA sliders dictionary is empty, though this might be fine if not using Neyman.")


            processed_df_stage2 = self.model.raw_area_df.copy() 
            processed_df_stage2["map_edited_class"] = processed_df_stage2["map_code"].map(edited_class_names_map)
            
            if not ("map_edited_class" in processed_df_stage2.columns and not processed_df_stage2["map_edited_class"].isnull().any()):
                raise AssertionError("Mapping edited class names to DataFrame failed or resulted in null values.")
            
            self.model.final_area_df = processed_df_stage2.copy() 
            logger.info("final_area_df in model updated with edited class names.")

            self._update_target_class_options() 
            self.stepper_content_3.children = self._create_step_3_content() 

            self.stage2_alert.type = "success"
            self.stage2_alert.children = ["Class data and user accuracies submitted. Proceed to Stage 3."]
            if hasattr(self, "stepper_step_2"): self.stepper_step_2.complete = True
            if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = False 
            self.model.current_step = 3 

        except (AttributeError, ValueError, KeyError, AssertionError) as stage2_err:
            logger.error(f"Error processing Stage 2 data: {stage2_err}", exc_info=True)
            self.stage2_alert.type = "error"
            self.stage2_alert.children = [f"Error submitting class data: {stage2_err}. Check logs."]
            if hasattr(self, "stepper_step_2"): self.stepper_step_2.complete = False
        except Exception as e_unexp_s2:
            logger.exception("An unexpected error occurred during Stage 2 submit:")
            self.stage2_alert.type = "error"
            self.stage2_alert.children = [f"An unexpected error occurred: {e_unexp_s2}. Check logs."]
            if hasattr(self, "stepper_step_2"): self.stepper_step_2.complete = False
        finally:
            widget.loading = False

    def _on_stage3_calculate(self, widget, event, data):
        logger.info("Stage 3 Calculate button clicked. Initiating sample design process.")
        widget.loading = True
        if hasattr(self, "update_map_btn"): self.update_map_btn.disabled = True 
        self.stage3_output.reset().show() 
        self.stage3_output.type = "info"
        self.stage3_output.children = ["Running sample design calculation... This may take a moment."]
        
        if self.calculated_total_sample_size_display:
            self.calculated_total_sample_size_display.children = ["Calculated Total Sample Size: Calculating..."]
            self.calculated_total_sample_size_display.show()

        if self.per_class_sample_size_display: # Reset per-class display
            if self.per_class_sample_size_display_card_text:
                 self.per_class_sample_size_display_card_text.children = ["Calculating..."]
            self.per_class_sample_size_display.show()

        self.model.sample_points_df = None 

        if self.sample_points_layer is not None:
            try:
                if self.sample_points_layer in self.map_widget.layers:
                    self.map_widget.remove_layer(self.sample_points_layer)
                    logger.debug("Removed existing sample points layer from map.")
                self.sample_points_layer = None
            except Exception as e_remove_spl:
                logger.warning(f"Could not remove old sample points layer: {e_remove_spl}")

        try:
            proc.expected_user_accuracy_widgets = self.expected_ua_sliders
            proc.objective_widget = self.objective_widget
            proc.target_class_widget = self.target_class_widget
            proc.target_class_allowable_error_widget = self.target_class_allowable_error_widget
            proc.target_overall_accuracy_widget = self.target_overall_accuracy_widget
            proc.allowable_error_widget = self.allowable_error_widget
            proc.confidence_level_widget = self.confidence_level_widget
            proc.min_samples_per_class_widget = self.min_samples_per_class_widget
            proc.allocation_method_widget = self.allocation_method_widget
            proc.sample_size_output = self.stage3_output 

            if not self.model.file_path: raise ValueError("Input file path is missing in the model.")
            if self.model.final_area_df is None: raise ValueError("Final area data is missing in the model.")
            
            output_directory_path = get_output_dir(self.model.file_path) 
            if not output_directory_path: raise RuntimeError("Failed to determine or create output directory for results.")
            logger.info(f"Output directory for this run: {output_directory_path}")

            results = proc.run_sample_design( # Ensure results is assigned
                map_file_path_str_arg=self.model.file_path,
                final_area_df_arg=self.model.final_area_df.copy(), 
                output_dir_str_arg=output_directory_path,
            )

            if results and results.get("success"):
                total_samples = results.get("actual_total_samples")
                if total_samples is not None and self.calculated_total_sample_size_display:
                    self.calculated_total_sample_size_display.children = [
                        f"Calculated Total Sample Size: {total_samples}"
                    ]
                elif self.calculated_total_sample_size_display:
                    self.calculated_total_sample_size_display.children = [
                        "Calculated Total Sample Size: N/A (from results)"
                    ]
                
                returned_samples_df = results.get("generated_samples_df")
                if returned_samples_df is not None and isinstance(returned_samples_df, pd.DataFrame):
                    self.model.sample_points_df = returned_samples_df.copy()

                    # START: Calculate and display per-class samples
                    if not self.model.sample_points_df.empty:
                        if 'map_edited_class' in self.model.sample_points_df.columns:
                            try:
                                per_class_counts_df = self.model.sample_points_df.groupby('map_edited_class').size().reset_index(name='n_samples')
                                
                                display_elements = []
                                if per_class_counts_df.empty:
                                    display_elements.append(v.Html(tag="p", children=["No samples generated or classes found."]))
                                else:
                                    ul_children = []
                                    for _, row in per_class_counts_df.iterrows():
                                        ul_children.append(v.Html(tag="li", children=[f"{html.escape(str(row['map_edited_class']))}: {row['n_samples']}"]))
                                    display_elements.append(v.Html(tag="ul", style_="list-style-type: disc; padding-left: 20px;", children=ul_children))
                                
                                if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                                    self.per_class_sample_size_display_card_text.children = display_elements
                                    self.per_class_sample_size_display.show()

                            except Exception as e_per_class:
                                logger.error(f"Error calculating/displaying per-class samples: {e_per_class}", exc_info=True)
                                if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                                    self.per_class_sample_size_display_card_text.children = ["Error calculating per-class samples."]
                                    self.per_class_sample_size_display.show()
                        else: # 'map_edited_class' not in columns
                            logger.warning("'map_edited_class' not in sample_points_df, cannot show per-class counts.")
                            if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                                self.per_class_sample_size_display_card_text.children = ["Per-class data unavailable (missing class column)."]
                                self.per_class_sample_size_display.show()
                    else: # sample_points_df is empty
                        if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                            self.per_class_sample_size_display_card_text.children = ["No samples generated to show per class."]
                            self.per_class_sample_size_display.show()
                    # END: Calculate and display per-class samples
                        
                    if not self.model.sample_points_df.empty:
                        if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = True
                        if hasattr(self, "update_map_btn"): self.update_map_btn.disabled = False
                    else: 
                        if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = True 
                        logger.warning("Sample calculation successful, but no sample points were generated (final check).")
                else: 
                    self.model.sample_points_df = None
                    if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = False
                    logger.warning("Sample calculation successful, but no sample points DataFrame found in results.")
                
                if self.model.sample_points_df is None or self.model.sample_points_df.empty:
                    if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                        self.per_class_sample_size_display_card_text.children = ["Samples per class: N/A (no samples data)."]
                        self.per_class_sample_size_display.show()
            
            else: 
                error_msg_from_proc = results.get("error_message", "Calculation failed, check logs.") if results else "Calculation failed, check logs."
                if self.calculated_total_sample_size_display:
                    self.calculated_total_sample_size_display.children = [
                        "Calculated Total Sample Size: Error"
                    ]
                if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                    self.per_class_sample_size_display_card_text.children = ["Samples per class: Error in calculation."]
                    self.per_class_sample_size_display.show()
                logger.error(f"Sample design calculation reported failure. Message: {error_msg_from_proc}")
                if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = False
                if hasattr(self, "update_map_btn"): self.update_map_btn.disabled = True
                
        except Exception as e_calc_ctrl: 
            logger.exception("Error occurred in AppController during Stage 3 Calculate orchestration:")
            self.stage3_output.type = "error" 
            self.stage3_output.add_msg(f"Controller-level error during calculation: {e_calc_ctrl}. Check logs.")
            if self.calculated_total_sample_size_display:
                self.calculated_total_sample_size_display.children = ["Calculated Total Sample Size: Error"]
            if self.per_class_sample_size_display and self.per_class_sample_size_display_card_text:
                self.per_class_sample_size_display_card_text.children = ["Samples per class: Controller error."]
                self.per_class_sample_size_display.show()
            if hasattr(self, "stepper_step_3"): self.stepper_step_3.complete = False
            if hasattr(self, "update_map_btn"): self.update_map_btn.disabled = True
        finally:
            widget.loading = False

    def _on_show_map_overlay_click(self, widget, event, data):
        logger.info("Show/Update Map Overlay button clicked. Refreshing base map overlay.")
        widget.loading = True
        self.stage3_output.reset().show() 
        self.stage3_output.type = "info"
        self.stage3_output.add_msg("Refreshing map overlay...")

        if not self.model.file_path or not Path(self.model.file_path).exists():
            self.stage3_output.type = "error"
            self.stage3_output.add_msg("No valid file loaded. Cannot display or refresh overlay.")
            logger.error("Attempted to refresh map overlay with no valid file_path in model.")
            widget.loading = False
            return

        file_path_obj_refresh = Path(self.model.file_path) 
        basename_refresh = file_path_obj_refresh.name 
        raster_exts_refresh = (".tif", ".tiff", ".img", ".pix", ".rst", ".grd", ".vrt", ".hdf", ".h5", ".jpeg2000")
        vector_exts_refresh = (".shp", ".sqlite", ".gdb", ".geojson", ".json", ".gml", ".kml", ".tab", ".mif")
        is_raster_refresh = file_path_obj_refresh.suffix.lower() in raster_exts_refresh 
        is_vector_refresh = file_path_obj_refresh.suffix.lower() in vector_exts_refresh 

        try:
            if self.base_map_overlay_layer is not None:
                try:
                    if self.base_map_overlay_layer in self.map_widget.layers:
                        self.map_widget.remove_layer(self.base_map_overlay_layer)
                        logger.debug("Removed existing base map overlay layer for refresh.")
                    self.base_map_overlay_layer = None 
                except Exception as e_remove_base:
                    logger.warning(f"Could not remove existing base map overlay for refresh: {e_remove_base}")

            refreshed_overlay = None 
            if is_raster_refresh:
                if not (_rasterio_available_map and _pyproj_available_proc):
                    self.stage3_output.type = "error"
                    self.stage3_output.add_msg("Rasterio/Pyproj libraries missing. Cannot refresh raster overlay.")
                    logger.error("Cannot refresh raster overlay: Missing Rasterio or PyProj.")
                    widget.loading = False
                    return
                
                raster_bounds_refresh = None 
                try:
                    with rasterio.open(self.model.file_path) as src_r: 
                        if src_r.crs:
                            dst_b = rasterio.warp.transform_bounds(src_r.crs, f"EPSG:{map_utils_target_crs_epsg}", *src_r.bounds) 
                            raster_bounds_refresh = [[dst_b[1], dst_b[0]], [dst_b[3], dst_b[2]]]
                        else:
                            self.stage3_output.add_msg("Warning: Raster for refresh is missing CRS. Bounds may be inaccurate.", "warning")
                            logger.warning("Raster for refresh missing CRS.")
                except Exception as bounds_err_r: 
                    self.stage3_output.add_msg(f"Error calculating raster bounds for refresh: {bounds_err_r}", "error")
                    logger.error(f"Error calculating raster bounds for refresh: {bounds_err_r}", exc_info=True)

                if raster_bounds_refresh and self.model.raw_area_df is not None:
                    refreshed_overlay = _add_overlay_layer(
                        self.map_widget, self.model.file_path, self.model.raw_area_df,
                        raster_bounds_refresh, None
                    )
                else:
                    msg = "Could not generate bounds or missing legend data (raw_area_df) for raster overlay refresh."
                    self.stage3_output.add_msg(msg, "warning")
                    logger.warning(msg)
            
            elif is_vector_refresh:
                if not _geopandas_available_proc:
                    self.stage3_output.type = "error"
                    self.stage3_output.add_msg("GeoPandas library missing. Cannot refresh vector overlay.")
                    logger.error("Cannot refresh vector overlay: Missing GeoPandas.")
                    widget.loading = False
                    return

                vector_gdf_r = None 
                try:
                    gdf_loaded_r = gpd.read_file(self.model.file_path) 
                    if not gdf_loaded_r.empty:
                        if gdf_loaded_r.crs is None:
                            self.stage3_output.add_msg(f"Warning: Vector '{basename_refresh}' for refresh has no CRS.", "warning")
                            logger.warning(f"Vector '{basename_refresh}' for refresh has no CRS.")
                        vector_gdf_r = gdf_loaded_r.to_crs(epsg=map_utils_target_crs_epsg) 
                except Exception as gdf_err_r: 
                    self.stage3_output.add_msg(f"Error loading/reprojecting vector for refresh: {gdf_err_r}", "error")
                    logger.error(f"Error loading/reprojecting vector for refresh: {gdf_err_r}", exc_info=True)

                if vector_gdf_r is not None and self.model.raw_area_df is not None:
                    refreshed_overlay = _add_overlay_layer(
                        self.map_widget, self.model.file_path, self.model.raw_area_df,
                        None, vector_gdf_r
                    )
                else:
                    msg = "Could not load/process vector or missing legend data (raw_area_df) for overlay refresh."
                    self.stage3_output.add_msg(msg, "warning")
                    logger.warning(msg)
            else:
                self.stage3_output.type = "warning"
                self.stage3_output.add_msg(f"File type of '{basename_refresh}' not recognized for overlay refresh.")
                logger.warning(f"Unrecognized file type for overlay refresh: {basename_refresh}")

            if refreshed_overlay:
                self.base_map_overlay_layer = refreshed_overlay 
                self.stage3_output.type = "success"
                self.stage3_output.add_msg(f"Map overlay '{basename_refresh}' refreshed successfully.")
                logger.info(f"Map overlay '{basename_refresh}' refreshed and added to map.")
            else: 
                if self.stage3_output.type != "error": self.stage3_output.type = "warning" 
                self.stage3_output.add_msg(f"Failed to refresh map overlay for '{basename_refresh}'. Check logs if not already detailed.")
                logger.warning(f"Failed to refresh map overlay for '{basename_refresh}'.")

        except Exception as e_refresh:
            logger.exception("An unexpected error occurred while refreshing map overlay:")
            self.stage3_output.type = "error"
            self.stage3_output.add_msg(f"Unexpected error refreshing overlay: {e_refresh}. Check logs.")
        finally:
            widget.loading = False

    def _on_update_map_click(self, widget, event, data):
        logger.info("Update Map (with samples) button clicked.")
        widget.loading = True
        try:
            self._update_map_display_ipyleaflet()
        except Exception as e_map_update:
            logger.exception("Error updating map display with sample points:")
            self.stage3_output.reset().show() 
            self.stage3_output.type = "error"
            self.stage3_output.add_msg(f"Error updating sample display on map: {e_map_update}. Check logs.")
        finally:
            widget.loading = False

    def _update_map_display_ipyleaflet(self):
        logger.debug("Updating ipyleaflet map display with current sample points...")
        
        if self.sample_points_layer is not None:
            try:
                if self.sample_points_layer in self.map_widget.layers:
                    self.map_widget.remove_layer(self.sample_points_layer)
                    logger.debug("Removed previous sample points layer from map.")
                self.sample_points_layer = None 
            except Exception as e_remove_samples:
                logger.warning(f"Could not remove previous sample points layer: {e_remove_samples}")

        points_df_for_map = self.model.sample_points_df 
        if points_df_for_map is None or points_df_for_map.empty:
            msg = "No sample points available in the model to display on the map."
            if hasattr(self, "stage3_output"): 
                self.stage3_output.type = "warning"
                self.stage3_output.add_msg(msg)
            logger.warning(msg)
            return

        req_cols_for_map = ["latitude", "longitude", "map_edited_class", "map_code"] 
        if not all(col in points_df_for_map.columns for col in req_cols_for_map):
            missing_cols_str = [col for col in req_cols_for_map if col not in points_df_for_map.columns] 
            err_msg_cols = f"Sample points DataFrame is missing required columns for map display: {missing_cols_str}."
            if hasattr(self, "stage3_output"):
                self.stage3_output.type = "error"
                self.stage3_output.add_msg(err_msg_cols)
            logger.error(err_msg_cols)
            return

        try:
            unique_classes_pts = points_df_for_map["map_edited_class"].dropna().unique() 
            class_colors_pts = generate_class_color_map(unique_classes_pts) 
            default_marker_color = "#FF0000" 
        except Exception as color_map_err:
            logger.error(f"Error generating class colors for map sample points: {color_map_err}", exc_info=True)
            class_colors_pts = {} 
            default_marker_color = "#FF0000"

        map_markers = [] 
        min_lat_pts, max_lat_pts = 90.0, -90.0  
        min_lon_pts, max_lon_pts = 180.0, -180.0 
        valid_points_added_count = 0 

        for _, row_pt_data in points_df_for_map.iterrows(): 
            try:
                lat_pt = float(row_pt_data["latitude"]) 
                lon_pt = float(row_pt_data["longitude"]) 
                if not (np.isfinite(lat_pt) and np.isfinite(lon_pt) and -90 <= lat_pt <= 90 and -180 <= lon_pt <= 180):
                    logger.warning(f"Skipping sample point with invalid/out-of-bounds coords: Lat={lat_pt}, Lon={lon_pt} for class {row_pt_data.get('map_edited_class', 'N/A')}")
                    continue

                min_lat_pts = min(min_lat_pts, lat_pt)
                max_lat_pts = max(max_lat_pts, lat_pt)
                min_lon_pts = min(min_lon_pts, lon_pt)
                max_lon_pts = max(max_lon_pts, lon_pt)

                map_class_pt = row_pt_data["map_edited_class"] 
                map_code_pt = row_pt_data["map_code"] 
                marker_color = class_colors_pts.get(map_class_pt, default_marker_color) 

                popup_html_str = ( 
                    f"<b>Class:</b> {html.escape(str(map_class_pt))}<br>"
                    f"<b>Code:</b> {html.escape(str(map_code_pt))}<br>"
                    f"<b>Lat:</b> {lat_pt:.6f}<br><b>Lon:</b> {lon_pt:.6f}"
                )
                point_marker = CircleMarker( 
                    location=(lat_pt, lon_pt), radius=5, color=marker_color, weight=1,
                    fill_color=marker_color, fill_opacity=0.8,
                    popup=widgets.HTML(value=popup_html_str) 
                )
                map_markers.append(point_marker)
                valid_points_added_count += 1
            except Exception as marker_create_err:
                logger.error(f"Error creating map marker for sample point: {row_pt_data.to_dict()}. Error: {marker_create_err}", exc_info=True)
        
        if map_markers:
            self.sample_points_layer = LayerGroup(layers=map_markers, name="Sample Points")
            self.map_widget.add_layer(self.sample_points_layer)
            logger.info(f"Added {valid_points_added_count} sample points to the map as a new layer.")

            try:
                logger.debug("Invalidating map size before fitting bounds to samples.")
                self.map_widget.invalidate_size() 
            except Exception as e_invalidate:
                logger.error(f"Error calling invalidate_size on map widget: {e_invalidate}")

            if valid_points_added_count > 0 and min_lat_pts <= max_lat_pts and min_lon_pts <= max_lon_pts:
                bounds_to_fit_samples = [[min_lat_pts, min_lon_pts], [max_lat_pts, max_lon_pts]] 
                if valid_points_added_count == 1:
                    padding_val = 0.01 
                    bounds_to_fit_samples = [
                        [min_lat_pts - padding_val, min_lon_pts - padding_val],
                        [max_lat_pts + padding_val, max_lon_pts + padding_val]
                    ]
                try:
                    self.map_widget.fit_bounds(bounds_to_fit_samples)
                    logger.debug(f"Map view fitted to sample points bounds: {bounds_to_fit_samples}")
                except Exception as fit_bounds_err:
                    logger.warning(f"Could not fit map bounds to sample points: {fit_bounds_err}")
            
            if hasattr(self, "stage3_output"):
                self.stage3_output.type = "success"
                self.stage3_output.add_msg(f"Displayed {valid_points_added_count} sample points on the map.")
        else:
            msg = "No valid sample points found to display on the map after processing."
            if hasattr(self, "stage3_output"):
                self.stage3_output.type = "warning"
                self.stage3_output.add_msg(msg)
            logger.warning(msg)
