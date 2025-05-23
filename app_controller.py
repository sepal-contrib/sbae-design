# sampling_design_tool/app_controller.py

import ipyvuetify as v
import sepal_ui.sepalwidgets as sui
import traitlets
import os
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import sys
import ipywidgets as widgets
import ipyleaflet
from ipyleaflet import (
    Map, TileLayer, basemaps, LayersControl, ZoomControl, ScaleControl, Marker,
    GeoJSON, ImageOverlay, CircleMarker, LayerGroup, FullScreenControl
)
import html # For popups

# Local module imports
from .app_model import AppModel
# Import map_utils and then access its attributes
from . import map_utils # Ensures map_utils is imported and its globals are set
from .map_utils import _add_overlay_layer, generate_class_color_map, target_crs_epsg as map_utils_target_crs_epsg

from . import processing as proc # Ensures processing is imported
from .processing import (
    compute_map_area, get_output_dir, get_z_score
)

# Optional dependency flags from submodules for convenience in controller
# Access flags AFTER modules are imported and their top-level code has run.
_rasterio_available_map = getattr(map_utils, '_rasterio_available', False)
_rasterio_available_proc = getattr(proc, '_rasterio_available', False)
_geopandas_available_proc = getattr(proc, '_geopandas_available', False)
_pyproj_available_proc = getattr(proc, '_pyproj_available', False)

# For direct use of rasterio and gpd in handlers if needed
try:
    import rasterio
    import rasterio.warp
except ImportError:
    pass

try:
    import geopandas as gpd
except ImportError:
    pass

print("Loading SEPAL-UI Application components and Backend Functions (AppController)...")
# Print the state of flags after they are set in app_controller
print(f"DEBUG AppController Top Level: _rasterio_available_map = {_rasterio_available_map}")
print(f"DEBUG AppController Top Level: _pyproj_available_proc = {_pyproj_available_proc}")


class AppController:
    def __init__(self, model: AppModel):
        self.model = model
        print("Initializing AppController...")
        self.sample_points_layer = None
        self.base_map_overlay_layer = None 

        self._create_map_widget()
        self._create_stage2_widgets() 
        self._create_stage3_widgets() 
        self._create_stepper()

        self.app = v.Card(
            class_="pa-4",
            children=[
                v.CardTitle(children=["Sampling Design Tool"]),
                self.stepper
            ]
        )
        self.model.observe(self._update_stepper_value, names='current_step')
        if hasattr(self, 'objective_widget'):
            self.objective_widget.observe(self._handle_objective_change, names='v_model')
        else:
            print("WARNING (AppController Init): objective_widget not yet created, cannot observe.")
        print("DEBUG: AppController initialized.")

    def _create_map_widget(self):
        print("DEBUG: Creating ipyleaflet.Map...")
        try:
            self.map_widget = Map(
                center=[0, 0], zoom=2, scroll_wheel_zoom=True, zoom_control=False,
                layout={'height': '450px', 'border': '1px solid #ccc'}
            )
            self.map_widget.add_layer(TileLayer(
                url=basemaps.OpenStreetMap.Mapnik.url, name="OpenStreetMap", base=True,
                attribution=basemaps.OpenStreetMap.Mapnik.attribution
            ))
            self.map_widget.add_control(LayersControl(position='topright'))
            self.map_widget.add_control(ScaleControl(position='bottomleft'))
            self.map_widget.add_control(ZoomControl(position='topleft'))
            self.map_widget.add_control(FullScreenControl())
            print("DEBUG: ipyleaflet.Map created with controls.")
        except Exception as map_init_err:
            print(f"!!! Error creating ipyleaflet Map widget: {map_init_err}")
            traceback.print_exc()
            self.map_widget = widgets.HTML("<p style='color:red; font-weight:bold;'>Error initializing map widget.</p>")

    def _create_stage2_widgets(self):
        print("DEBUG: Creating Stage 2 control widget holders.")
        self.class_name_widgets = {} 
        self.expected_ua_sliders = {} 
        self.stage2_alert = sui.Alert().hide()
        self.stage2_submit_btn = sui.Btn("Submit Classes & Proceed to Stage 3", icon="mdi-check", class_="ma-2", disabled=True)

    def _create_stage3_widgets(self):
        print("DEBUG: Creating Stage 3 control widgets.")
        self.objective_widget = sui.Select(label="Calculation Objective", items=['Overall Accuracy', 'Target Class Precision'], v_model='Overall Accuracy', dense=True,outlined=True, class_="mb-2", clearable=False)
        self.target_class_widget = sui.Select(label="Target Class (for Precision Objective)", items=[], v_model=None, dense=True, clearable=True, outlined=True, disabled=True, class_="mb-2")
        self.target_class_allowable_error_widget = v.Slider(label="Target Class Allowable Error (Ej)", min=0.01, max=0.25, step=0.001, v_model=0.05, thumb_label='always', class_="mt-2 mb-2", disabled=True)
        self.target_overall_accuracy_widget = v.Slider(label="Target Overall Accuracy (OA)", min=0.01, max=0.99, step=0.01, v_model=0.90, thumb_label='always', class_="mt-2 mb-2")
        self.allowable_error_widget = v.Slider(label="Allowable Error (Overall)", min=0.001, max=0.10, step=0.001, v_model=0.05, thumb_label='always', class_="mt-2 mb-2")
        self.confidence_level_widget = v.Slider(label="Confidence Level", min=0.80, max=0.999, step=0.001, v_model=0.95, thumb_label='always', class_="mt-2 mb-2")
        self.min_samples_per_class_widget = v.Slider(label="Min Samples per Class", min=5, max=500, step=5, v_model=20, thumb_label='always', class_="mt-2 mb-2")
        self.allocation_method_widget = sui.Select(label="Allocation Method", items=['Proportional', 'Neyman', 'Equal'], v_model='Proportional', dense=True, outlined=True, class_="mb-2", clearable=False)
        
        self.stage3_calculate_btn = sui.Btn("Calculate Sample & Generate Design", icon="mdi-cogs", class_="ma-2", disabled=True)
        self.update_map_btn = sui.Btn("Show/Update Samples on Map", icon="mdi-map-marker-check", class_="ma-2", disabled=True)
        self.show_map_overlay_btn = sui.Btn("Show/Update Map Overlay", icon="mdi-layers-refresh", class_="ma-2", disabled=True)
        
        self.stage3_output = sui.Alert(class_="mt-2 pa-2", type='info').hide()

    def _create_stepper(self):
        print("DEBUG: Creating Stepper structure.")
        self.step_1_title = "1. Select File & Load"
        self.step_2_title = "2. Edit Classes & Accuracy"
        self.step_3_title = "3. Sample Design & Results"
        self.step_1 = v.StepperStep(step=1, complete=False, children=[self.step_1_title])
        self.step_2 = v.StepperStep(step=2, complete=False, children=[self.step_2_title])
        self.step_3 = v.StepperStep(step=3, complete=False, children=[self.step_3_title])
        
        try: 
            step1_content_widgets = self._create_step_1_content()
        except Exception as e_step1:
            print(f"!!! FATAL ERROR creating Step 1 content: {e_step1}"); traceback.print_exc()
            step1_content_widgets = [v.Html(tag='p', children=[f"Error creating UI Step 1: {e_step1}"])]

        self.content_1 = v.StepperContent(step=1, children=step1_content_widgets)
        self.content_2 = v.StepperContent(step=2, children=[v.ProgressCircular(indeterminate=True, class_="ma-4")])
        self.content_3 = v.StepperContent(step=3, children=[v.ProgressCircular(indeterminate=True, class_="ma-4")])
        
        self.stepper = v.Stepper(
            class_="mb-5", 
            v_model=self.model.current_step,
            children=[
                v.StepperHeader(children=[self.step_1, self.step_2, self.step_3]),
                v.StepperItems(children=[self.content_1, self.content_2, self.content_3])
            ]
        )
        print("DEBUG: Stepper structure created.")

    def _create_step_1_content(self):
        print("DEBUG: Creating Step 1 Content")
        self.file_input = sui.FileInput(
            label="Select Map File (.tif, .shp, .geojson, etc.)",
            folder=self.model.current_dir, 
            extensions=['.tif', '.tiff', '.img', '.pix', '.rst', '.grd', '.vrt', '.hdf', '.h5', '.jpeg2000',
                        '.shp', '.sqlite', '.gdb', '.geojson', '.json', '.gml', '.kml', '.tab', '.mif']
        )
        traitlets.link((self.file_input, 'file'), (self.model, 'selected_file')) 
        self.load_btn = sui.Btn("Load File, Compute Area & Display Map", icon="mdi-upload", class_="ma-2")
        self.alert_step1 = sui.Alert(
            children=[str(self.model.alert_message)], 
            type=self.model.alert_type
        ).show() 

        content = v.Layout(
            column=True, 
            children=[
                self.file_input,
                v.Layout(row=True, justify_center=True, children=[self.load_btn]), 
                self.alert_step1, 
                self.map_widget 
            ]
        )
        self.load_btn.on_event('click', self._on_load_click) 
        print("DEBUG: Step 1 Content created.")
        return [content] 

    def _create_step_2_content(self):
        print("DEBUG: Building Stage 2 Content...")
        self.class_name_widgets.clear() 
        self.expected_ua_sliders.clear()
        if self.model.raw_area_df is None or self.model.raw_area_df.empty:
            return [v.Html(tag='p', children=["Error: No area data loaded from Step 1."])]

        area_df_sorted = self.model.raw_area_df.copy().sort_values('map_code').reset_index(drop=True)
        name_items = [v.ListItemGroup(children=[v.Html(class_='pl-4 pt-2', tag='b', children=["Edit Class Names (Optional):"])])]
        ua_items = [v.ListItemGroup(children=[v.Html(class_='pl-4 pt-2', tag='b', children=["Expected User's Accuracy (for Neyman):"])])]

        for _, row in area_df_sorted.iterrows():
            map_code = row['map_code']
            initial_name = str(row.get('map_edited_class', map_code)) 
            name_widget = v.TextField(label=f"Name for Code {map_code}", v_model=initial_name, outlined=True, dense=True, class_="ma-2")
            self.class_name_widgets[map_code] = name_widget
            name_items.append(v.ListItem(children=[name_widget], class_="ma-0 pa-0"))

            slider_label = f"'{name_widget.v_model}' (Code: {map_code})" 
            acc_widget = v.Slider(label=slider_label, v_model=0.80, min=0.01, max=0.99, step=0.01, thumb_label='always', class_="ma-2")
            self.expected_ua_sliders[map_code] = acc_widget
            ua_items.append(v.ListItem(children=[acc_widget], class_="ma-0 pa-0"))

            def _make_observer(slider, text_w, code_val):
                def _observer(change): 
                    new_name = text_w.v_model.strip() if text_w.v_model else f"Code {code_val}"
                    slider.label = f"'{new_name}' (Code: {code_val})"
                return _observer
            name_widget.observe(_make_observer(acc_widget, name_widget, map_code), names='v_model')

        edit_card = v.Card(outlined=True, class_="mb-4", children=[v.List(dense=True, children=name_items)])
        ua_card = v.Card(outlined=True, class_="mb-4", children=[v.List(dense=True, children=ua_items)])
        self.stage2_submit_btn.disabled = False 
        self.stage2_alert.reset().hide() 

        try: self.stage2_submit_btn.on_event('click', None, remove=True) 
        except Exception as e: print(f"DEBUG: Error removing previous Stage 2 submit handler: {e}")
        self.stage2_submit_btn.on_event('click', self._on_stage2_submit) 
        
        content = v.Layout(
            column=True, 
            children=[
                v.Html(tag='h3', children=["Stage 2: Review Classes & Set Accuracies"]),
                self.stage2_alert, 
                edit_card,          
                ua_card,            
                self.stage2_submit_btn 
            ]
        )
        print("DEBUG: Stage 2 Content created.")
        return [content]

    def _create_step_3_content(self):
        print("DEBUG: Creating Step 3 Content...")
        if self.model.final_area_df is None or self.model.final_area_df.empty:
            return [v.Html(tag='p', children=["Error: Finalized class data is not available from Step 2."])]

        summary_out = v.Sheet(
            outlined=True, class_="pa-2 mb-4", 
            style_="max-height: 150px; overflow-y: auto; background-color: #f5f5f5;", 
            children=[
                v.Html(tag='b', children=["Final Class Summary (Map Code, Name, Area):"]),
                v.Html(tag='pre', style_="color: black;", children=[self.model.final_area_df[['map_code', 'map_edited_class', 'map_area']].round(2).to_string(index=False)])
            ]
        )
        param_layout = v.Layout(
            column=True, class_="mb-2", 
            children=[
                self.objective_widget, self.target_class_widget, self.target_class_allowable_error_widget,
                v.Divider(class_="my-2"), 
                v.Html(tag='i', children=["Standard Sampling Parameters:"]),
                self.target_overall_accuracy_widget, self.allowable_error_widget,
                self.confidence_level_widget, self.min_samples_per_class_widget, self.allocation_method_widget
            ]
        )
        self._handle_objective_change({'new': self.objective_widget.v_model}) 
        self.stage3_output.reset().hide()
        self.stage3_calculate_btn.disabled = False 
        self.update_map_btn.disabled = True
        self.show_map_overlay_btn.disabled = not (self.model.file_path and self.base_map_overlay_layer)

        try: self.stage3_calculate_btn.on_event('click', self._on_stage3_calculate, remove=True)
        except Exception as e: print(f"DEBUG: Error removing calc handler: {e}")
        self.stage3_calculate_btn.on_event('click', self._on_stage3_calculate)
        
        try: self.update_map_btn.on_event('click', self._on_update_map_click, remove=True)
        except Exception as e: print(f"DEBUG: Error removing map update handler: {e}")
        self.update_map_btn.on_event('click', self._on_update_map_click)

        try: self.show_map_overlay_btn.on_event('click', self._on_show_map_overlay_click, remove=True)
        except Exception as e: print(f"DEBUG: Error removing show map overlay handler: {e}")
        self.show_map_overlay_btn.on_event('click', self._on_show_map_overlay_click)

        content = v.Layout(
            column=True, 
            children=[
                v.Html(tag='h3', children=["Stage 3: Set Sample Design Parameters & Calculate"]),
                summary_out, 
                v.Card(outlined=True, class_="pa-4 mb-4", children=[param_layout]), 
                v.Layout(row=True, class_="d-flex justify-center flex-wrap", 
                         children=[self.stage3_calculate_btn, self.update_map_btn, self.show_map_overlay_btn]),
                self.stage3_output, 
                v.Divider(class_="my-4"), 
                v.Html(tag='h4', children=["Map Visualization"], class_="mb-2"), 
                self.map_widget 
            ]
        )
        print("DEBUG: Step 3 Content created.")
        return [content]

    def _update_stepper_value(self, change):
        if hasattr(self, 'stepper') and self.stepper:
            self.stepper.v_model = change['new']
            print(f"DEBUG: Stepper value model updated to: {self.stepper.v_model}")

    def _handle_objective_change(self, change):
        is_target_class_mode = (change.get('new') == 'Target Class Precision')
        can_enable_target_class = is_target_class_mode and \
                                  len(getattr(self.target_class_widget, 'items', [])) > 0 and \
                                  self.target_class_widget.items[0].get('value') is not None 
        
        self.target_class_widget.disabled = not can_enable_target_class
        self.target_class_allowable_error_widget.disabled = not is_target_class_mode 
        print(f"DEBUG: Objective changed to '{change.get('new')}'. Target class widgets enabled: {not self.target_class_widget.disabled}")

    def _update_target_class_options(self):
        print("DEBUG: Updating target class options...")
        target_widget = self.target_class_widget
        if self.model.final_area_df is not None and isinstance(self.model.final_area_df, pd.DataFrame) and not self.model.final_area_df.empty:
            try:
                sorted_df = self.model.final_area_df[['map_code', 'map_edited_class']].drop_duplicates().sort_values('map_edited_class')
                choices = [{'text': f"{row['map_edited_class']} (Code: {row['map_code']})", 'value': row['map_code']}
                           for _, row in sorted_df.iterrows()]
                if not choices: 
                    choices = [{'text':'No classes eligible for targeting', 'value':None}]
                target_widget.items = choices
                target_widget.v_model = None if not choices or choices[0].get('value') is None else choices[0]['value']
                print(f"DEBUG: Updated target class items (count: {len(choices)}).")
            except Exception as e:
                print(f"Error formatting target class choices: {e}")
                target_widget.items = [{'text':'Error loading classes', 'value':None}]
        else:
            target_widget.items = [{'text':'No classes available', 'value':None}]
        self._handle_objective_change({'new': self.objective_widget.v_model})

    def _on_load_click(self, widget, event, data):
        print("DEBUG: Load button clicked.")
        self.model.alert_type = 'info'
        selected_filename = os.path.basename(str(self.model.selected_file)) if self.model.selected_file else "file"
        self.model.alert_message = f"Loading {selected_filename}..."
        self.alert_step1.type = self.model.alert_type
        self.alert_step1.children = [self.model.alert_message]
        widget.loading = True

        self.model.file_path = None; self.model.raw_area_df = None; self.model.final_area_df = None; self.model.sample_points_df = None
        if hasattr(self, 'step_1'): self.step_1.complete = False
        if hasattr(self, 'step_2'): self.step_2.complete = False
        if hasattr(self, 'step_3'): self.step_3.complete = False

        layers_to_clear_on_load = []
        if self.sample_points_layer: layers_to_clear_on_load.append(self.sample_points_layer)
        if self.base_map_overlay_layer: layers_to_clear_on_load.append(self.base_map_overlay_layer)
        
        for lyr_to_clear in layers_to_clear_on_load: 
            try:
                if lyr_to_clear in self.map_widget.layers: self.map_widget.remove_layer(lyr_to_clear)
            except Exception as e: print(f"Warning: Could not remove tracked layer '{getattr(lyr_to_clear, 'name', 'Unnamed')}': {e}")
        self.sample_points_layer = None
        self.base_map_overlay_layer = None
        
        current_map_layers_list = list(self.map_widget.layers) 
        for lyr_general in current_map_layers_list: 
            if isinstance(lyr_general, (ImageOverlay, GeoJSON, LayerGroup)) and getattr(lyr_general, 'base', False) is False :
                 try: self.map_widget.remove_layer(lyr_general)
                 except Exception as e: print(f"Warning: Could not remove other layer '{getattr(lyr_general, 'name', 'Unnamed')}': {e}")

        if hasattr(self, 'update_map_btn'): self.update_map_btn.disabled = True
        if hasattr(self, 'show_map_overlay_btn'): self.show_map_overlay_btn.disabled = True
        
        try:
            selected = self.model.selected_file
            assert selected and Path(selected).exists(), "Please select a valid file first."
            file_path_obj = Path(selected)
            basename = file_path_obj.name
            raster_extensions = ('.tif', '.tiff', '.img', '.pix', '.rst', '.grd', '.vrt', '.hdf', '.h5', '.jpeg2000')
            vector_extensions = ('.shp', '.sqlite', '.gdb', '.geojson', '.json', '.gml', '.kml', '.tab', '.mif')
            is_raster = file_path_obj.suffix.lower() in raster_extensions
            is_vector = file_path_obj.suffix.lower() in vector_extensions
            
            # --- DEBUG PRINTS FOR FLAGS ---
            # These use the module-level flags defined at the top of app_controller.py
            print(f"DEBUG Controller Flags Check BEFORE is_raster (in _on_load_click): _rasterio_available_map = {_rasterio_available_map}")
            print(f"DEBUG Controller Flags Check BEFORE is_raster (in _on_load_click): _pyproj_available_proc = {_pyproj_available_proc}")
            try:
                import rasterio 
                import rasterio.warp as controller_rasterio_warp
                print("DEBUG Controller Import Check (in _on_load_click): rasterio.warp successfully imported directly.")
            except ImportError as e:
                print(f"DEBUG Controller Import Check (in _on_load_click): FAILED to import rasterio.warp directly. Error: {e}")
            # --- END DEBUG PRINTS ---

            area_data = compute_map_area(selected) 
            assert area_data is not None and isinstance(area_data, pd.DataFrame) and not area_data.empty, "Area computation failed or returned empty."
            if 'map_edited_class' not in area_data.columns: area_data['map_edited_class'] = area_data['map_code'].astype(str)
            self.model.file_path = selected
            self.model.raw_area_df = area_data

            newly_added_overlay_layer = None 
            overlay_bounds = None; vector_gdf = None 

            if is_raster:
                print(f"DEBUG _on_load_click: is_raster=True. Evaluating dependency flags for bounds calculation...")
                # Use the module-level flags here
                print(f"DEBUG _on_load_click (inside is_raster): _rasterio_available_map = {_rasterio_available_map}")
                print(f"DEBUG _on_load_click (inside is_raster): _pyproj_available_proc = {_pyproj_available_proc}")
                if _rasterio_available_map and _pyproj_available_proc: 
                    try:
                        with rasterio.open(selected) as src: 
                            if src.crs:
                                dst_bounds = rasterio.warp.transform_bounds(src.crs, f"EPSG:{map_utils_target_crs_epsg}", *src.bounds) 
                                overlay_bounds = [[dst_bounds[1], dst_bounds[0]], [dst_bounds[3], dst_bounds[2]]]
                            else: print("Warning: Raster missing CRS for bounds calculation.")
                    except Exception as bounds_err: print(f"Error calculating raster bounds: {bounds_err}")
                
                if overlay_bounds: 
                    newly_added_overlay_layer = _add_overlay_layer(self.map_widget, selected, self.model.raw_area_df, overlay_bounds, None)
                else: 
                    print("Skipping raster overlay due to missing bounds or dependencies.") 
            
            elif is_vector: 
                print(f"DEBUG _on_load_click: is_vector=True. Checking GeoPandas availability...")
                if _geopandas_available_proc: 
                    try:
                        gdf_loaded = gpd.read_file(selected) 
                        if not gdf_loaded.empty:
                            if gdf_loaded.crs is None: print("Warning: Vector has no CRS defined.")
                            vector_gdf = gdf_loaded.to_crs(epsg=map_utils_target_crs_epsg) 
                    except Exception as gdf_err: print(f"Error loading/reprojecting vector: {gdf_err}")
                
                if vector_gdf is not None:
                    newly_added_overlay_layer = _add_overlay_layer(self.map_widget, selected, self.model.raw_area_df, None, vector_gdf)
                else: print("Skipping vector overlay due to loading/processing error, empty GDF, or missing GeoPandas.")

            if newly_added_overlay_layer:
                self.base_map_overlay_layer = newly_added_overlay_layer 
                self.show_map_overlay_btn.disabled = False 
                print(f"DEBUG: Loaded map overlay '{getattr(newly_added_overlay_layer, 'name', 'Unnamed')}' tracked.")
            else:
                self.show_map_overlay_btn.disabled = True 

            self.model.alert_type = 'success'
            self.model.alert_message = f"Loaded {basename}. Overlay added status: {newly_added_overlay_layer is not None}. Proceed to Step 2."
            if hasattr(self, 'step_1'): self.step_1.complete = True
            self.content_2.children = self._create_step_2_content()
            self.model.current_step = 2
        except (AssertionError, ValueError, FileNotFoundError, ImportError) as e:
            print(f"!!! Error during file loading: {e}"); traceback.print_exc()
            self.model.alert_type = 'error'; self.model.alert_message = f"Error processing file: {e}"
            if hasattr(self, 'step_1'): self.step_1.complete = False
        except Exception as e: 
            print(f"!!! An unexpected error occurred during file load: {e}"); traceback.print_exc()
            self.model.alert_type = 'error'; self.model.alert_message = f"Unexpected error: {e}"
            if hasattr(self, 'step_1'): self.step_1.complete = False
        finally:
            self.alert_step1.type = self.model.alert_type
            self.alert_step1.children = [self.model.alert_message]
            widget.loading = False

    def _on_stage2_submit(self, widget, event, data):
        widget.loading = True
        self.stage2_alert.reset().show(); self.stage2_alert.type = 'info'
        self.stage2_alert.children = ["Processing class edits..."]
        try:
            if not hasattr(self, 'class_name_widgets') or not hasattr(self, 'expected_ua_sliders'):
                raise AttributeError("Stage 2 widgets (class names or UA sliders) not initialized properly.")
            
            edited_classes = {
                code: name_w.v_model.strip() if name_w.v_model and name_w.v_model.strip() else str(code)
                for code, name_w in self.class_name_widgets.items()
            }
            if self.model.raw_area_df is None: 
                raise ValueError("Raw area data (raw_area_df) is missing from model.")
            
            raw_df_codes = set(self.model.raw_area_df['map_code'])
            slider_codes = set(self.expected_ua_sliders.keys())
            if raw_df_codes != slider_codes:
                print(f"ERROR DEBUG: Slider keys {slider_codes} != Raw DF keys {raw_df_codes}")
                raise ValueError("Mismatch between UA sliders and loaded map codes.")
            
            processed_df = self.model.raw_area_df.copy()
            processed_df['map_edited_class'] = processed_df['map_code'].map(edited_classes)
            assert 'map_edited_class' in processed_df.columns and not processed_df['map_edited_class'].isnull().any(), "Mapping edited class names failed."
            self.model.final_area_df = processed_df.copy()
            
            self._update_target_class_options()
            self.content_3.children = self._create_step_3_content() 
            self.stage2_alert.type = 'success'; self.stage2_alert.children = ["Class data submitted. Proceed to Stage 3."]
            if hasattr(self, 'step_2'): self.step_2.complete = True
            if hasattr(self, 'step_3'): self.step_3.complete = False 
            self.model.current_step = 3
        except (AttributeError, ValueError, KeyError, AssertionError) as e:
            print(f"!!! Error processing Stage 2 data: {e}"); traceback.print_exc()
            self.stage2_alert.type = 'error'; self.stage2_alert.children = [f"Error submitting class data: {e}"]
            if hasattr(self, 'step_2'): self.step_2.complete = False
        except Exception as e:
            print(f"!!! An unexpected error occurred during Stage 2 submit: {e}"); traceback.print_exc()
            self.stage2_alert.type = 'error'; self.stage2_alert.children = [f"Unexpected error: {e}"]
            if hasattr(self, 'step_2'): self.step_2.complete = False
        finally: widget.loading = False

    def _on_stage3_calculate(self, widget, event, data):
        widget.loading = True; self.update_map_btn.disabled = True
        self.stage3_output.reset().show(); self.stage3_output.type = 'info'
        self.stage3_output.children = ["Running sample design calculation..."]
        self.model.sample_points_df = None 

        if self.sample_points_layer is not None:
            try:
                if self.sample_points_layer in self.map_widget.layers: self.map_widget.remove_layer(self.sample_points_layer)
                self.sample_points_layer = None
            except Exception as e: print(f"Warning: Could not remove old sample points layer: {e}")
        
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
            
            assert self.model.file_path, "Model is missing the input file path."
            assert self.model.final_area_df is not None, "Model is missing the final area data."
            output_directory = get_output_dir(self.model.file_path) 
            assert output_directory, "Failed to determine or create output directory."

            proc.run_sample_design( 
                map_file_path_str=self.model.file_path,
                final_area_df_input=self.model.final_area_df.copy(), 
                output_dir_str=output_directory
            )
            
            if proc.sample_points_df is not None and isinstance(proc.sample_points_df, pd.DataFrame):
                self.model.sample_points_df = proc.sample_points_df.copy() 
                if not self.model.sample_points_df.empty:
                    if hasattr(self, 'step_3'): self.step_3.complete = True
                    self.update_map_btn.disabled = False 
                else: 
                    if hasattr(self, 'step_3'): self.step_3.complete = True 
            else:
                self.model.sample_points_df = None
                if hasattr(self, 'step_3'): self.step_3.complete = False
        
        except Exception as e:
            print(f"!!! Error during Stage 3 Calculate: {e}"); traceback.print_exc()
            self.stage3_output.type = 'error'; self.stage3_output.children = [f"Error during calculation: {e}"]
            if hasattr(self, 'step_3'): self.step_3.complete = False
            self.update_map_btn.disabled = True 
        finally: widget.loading = False

    def _on_show_map_overlay_click(self, widget, event, data):
        print("DEBUG: Show/Update Map Overlay button clicked.")
        widget.loading = True
        self.stage3_output.reset().show(); self.stage3_output.type = 'info'
        self.stage3_output.add_msg("Refreshing map overlay...")

        if not self.model.file_path or not Path(self.model.file_path).exists():
            self.stage3_output.type = "error"
            self.stage3_output.add_msg("No valid file loaded to display as overlay.")
            widget.loading = False
            return

        file_path_obj = Path(self.model.file_path)
        basename = file_path_obj.name
        raster_extensions = ('.tif', '.tiff', '.img', '.pix', '.rst', '.grd', '.vrt', '.hdf', '.h5', '.jpeg2000')
        vector_extensions = ('.shp', '.sqlite', '.gdb', '.geojson', '.json', '.gml', '.kml', '.tab', '.mif')
        is_raster = file_path_obj.suffix.lower() in raster_extensions
        is_vector = file_path_obj.suffix.lower() in vector_extensions

        try:
            if self.base_map_overlay_layer is not None:
                try:
                    if self.base_map_overlay_layer in self.map_widget.layers:
                        self.map_widget.remove_layer(self.base_map_overlay_layer)
                        print("DEBUG: Removed existing base map overlay layer for refresh.")
                    self.base_map_overlay_layer = None
                except Exception as e:
                    print(f"Warning: Could not remove existing base map overlay layer for refresh: {e}")

            new_refreshed_overlay_layer = None 
            if is_raster:
                # Use the module-level flags for consistency
                if not (_rasterio_available_map and _pyproj_available_proc):
                    self.stage3_output.type = "error"; self.stage3_output.add_msg("Rasterio/Pyproj missing for raster overlay refresh."); widget.loading=False; return
                
                overlay_bounds_refresh = None 
                try:
                    with rasterio.open(self.model.file_path) as src_refresh: 
                        if src_refresh.crs:
                            dst_bounds_refresh = rasterio.warp.transform_bounds(src_refresh.crs, f"EPSG:{map_utils_target_crs_epsg}", *src_refresh.bounds) 
                            overlay_bounds_refresh = [[dst_bounds_refresh[1], dst_bounds_refresh[0]], [dst_bounds_refresh[3], dst_bounds_refresh[2]]]
                        else: self.stage3_output.add_msg("Warning: Raster missing CRS, cannot calculate accurate bounds for refresh.", "warning")
                except Exception as bounds_err_refresh: 
                    self.stage3_output.add_msg(f"Error calculating raster bounds for refresh: {bounds_err_refresh}", "error"); print(f"Error calc bounds for refresh: {bounds_err_refresh}")
                
                if overlay_bounds_refresh and self.model.raw_area_df is not None: 
                    new_refreshed_overlay_layer = _add_overlay_layer(self.map_widget, self.model.file_path, self.model.raw_area_df, overlay_bounds_refresh, None)
                else: self.stage3_output.add_msg("Could not generate bounds or missing legend data for raster overlay refresh.", "warning")
            
            elif is_vector:
                if not _geopandas_available_proc: 
                    self.stage3_output.type = "error"; self.stage3_output.add_msg("GeoPandas missing for vector overlay refresh."); widget.loading=False; return
                
                vector_gdf_refresh = None 
                try:
                    gdf_loaded_refresh = gpd.read_file(self.model.file_path) 
                    if not gdf_loaded_refresh.empty:
                        if gdf_loaded_refresh.crs is None: self.stage3_output.add_msg("Warning: Vector has no CRS defined for refresh.", "warning")
                        vector_gdf_refresh = gdf_loaded_refresh.to_crs(epsg=map_utils_target_crs_epsg)
                except Exception as gdf_err_refresh: 
                    self.stage3_output.add_msg(f"Error loading/reprojecting vector for refresh: {gdf_err_refresh}", "error"); print(f"Error loading vector for refresh: {gdf_err_refresh}")

                if vector_gdf_refresh is not None and self.model.raw_area_df is not None: 
                    new_refreshed_overlay_layer = _add_overlay_layer(self.map_widget, self.model.file_path, self.model.raw_area_df, None, vector_gdf_refresh)
                else: self.stage3_output.add_msg("Could not load/process vector or missing legend data for overlay refresh.", "warning")
            
            else:
                self.stage3_output.type = "warning"; self.stage3_output.add_msg(f"File type of '{basename}' not recognized for overlay refresh.")

            if new_refreshed_overlay_layer:
                self.base_map_overlay_layer = new_refreshed_overlay_layer
                self.stage3_output.type = "success"
                self.stage3_output.add_msg(f"Map overlay '{basename}' refreshed successfully.")
            else:
                if self.stage3_output.type != 'error': self.stage3_output.type = "warning" 
                self.stage3_output.add_msg(f"Failed to refresh map overlay for '{basename}'. Check logs.")

        except Exception as e:
            print(f"!!! Error refreshing map overlay: {e}"); traceback.print_exc()
            self.stage3_output.type = "error"; self.stage3_output.add_msg(f"Error refreshing overlay: {e}")
        finally:
            widget.loading = False

    def _on_update_map_click(self, widget, event, data):
        widget.loading = True
        try: 
            self._update_map_display_ipyleaflet()
        except Exception as e:
            print(f"!!! Error updating map display with samples: {e}"); traceback.print_exc()
            self.stage3_output.type = "error"; self.stage3_output.add_msg(f"Error updating sample display: {e}")
        finally: 
            widget.loading = False

    def _update_map_display_ipyleaflet(self):
        print("DEBUG: Updating ipyleaflet map display with sample points...")
        if self.sample_points_layer is not None:
            try:
                if self.sample_points_layer in self.map_widget.layers: self.map_widget.remove_layer(self.sample_points_layer)
                self.sample_points_layer = None
            except Exception as e: print(f"Warning: Could not remove previous sample points layer: {e}")

        points_df_local = self.model.sample_points_df
        if points_df_local is None or points_df_local.empty:
            self.stage3_output.type = "warning"; self.stage3_output.add_msg("No sample points available to display.")
            return

        required_cols = ['latitude', 'longitude', 'map_edited_class', 'map_code']
        if not all(col in points_df_local.columns for col in required_cols):
            missing_cols_samples = [col for col in required_cols if col not in points_df_local.columns] 
            self.stage3_output.type = "error"; self.stage3_output.add_msg(f"Sample points DataFrame is missing required columns: {missing_cols_samples}.")
            return

        try:
            unique_classes_in_points = points_df_local['map_edited_class'].unique()
            class_colors_samples = generate_class_color_map(unique_classes_in_points) 
            default_color_samples = '#FF0000' 
        except Exception as color_err_samples: 
            print(f"Error generating class colors for map samples: {color_err_samples}"); class_colors_samples = {}; default_color_samples = '#FF0000'

        markers_list = []; min_lat_samples, max_lat_samples = 90.0, -90.0; min_lon_samples, max_lon_samples = 180.0, -180.0 
        points_added_count = 0
        for _, row_sample in points_df_local.iterrows(): 
            try:
                lat_sample = float(row_sample['latitude']); lon_sample = float(row_sample['longitude']) 
                if not (np.isfinite(lat_sample) and np.isfinite(lon_sample) and -90 <= lat_sample <= 90 and -180 <= lon_sample <= 180):
                    print(f"Warning: Skipping sample point with invalid coords: lat={lat_sample}, lon={lon_sample}"); continue
                
                min_lat_samples = min(min_lat_samples, lat_sample); max_lat_samples = max(max_lat_samples, lat_sample)
                min_lon_samples = min(min_lon_samples, lon_sample); max_lon_samples = max(max_lon_samples, lon_sample)
                
                map_class_sample = row_sample['map_edited_class']; map_code_sample = row_sample['map_code'] 
                color_sample = class_colors_samples.get(map_class_sample, default_color_samples) 
                
                popup_content_html_sample = (f"<b>Class:</b> {html.escape(str(map_class_sample))}<br>" 
                                         f"<b>Code:</b> {html.escape(str(map_code_sample))}<br>"
                                         f"<b>Lat:</b> {lat_sample:.6f}<br><b>Lon:</b> {lon_sample:.6f}")
                marker_sample = CircleMarker(location=(lat_sample, lon_sample), radius=5, color=color_sample, weight=1, 
                                      fill_color=color_sample, fill_opacity=0.8, popup=widgets.HTML(value=popup_content_html_sample))
                markers_list.append(marker_sample)
                points_added_count +=1
            except Exception as marker_err_sample: 
                print(f"Error creating marker for sample point: {marker_err_sample}")
        
        if markers_list:
            self.sample_points_layer = LayerGroup(layers=markers_list, name='Sample Points')
            self.map_widget.add_layer(self.sample_points_layer)
            if min_lat_samples <= max_lat_samples and min_lon_samples <= max_lon_samples: 
                bounds_for_samples = [[min_lat_samples, min_lon_samples], [max_lat_samples, max_lon_samples]] 
                if points_added_count == 1: 
                    padding_for_samples = 0.01 
                    bounds_for_samples = [[min_lat_samples-padding_for_samples, min_lon_samples-padding_for_samples], [max_lat_samples+padding_for_samples, max_lon_samples+padding_for_samples]]
                try: 
                    self.map_widget.fit_bounds(bounds_for_samples)
                except Exception as bounds_err_samples: 
                    print(f"Warning: Could not fit map bounds to samples: {bounds_err_samples}")
            self.stage3_output.type = "success"; self.stage3_output.add_msg(f"Displayed {points_added_count} sample points on the map.")
        else:
            self.stage3_output.type = "warning"; self.stage3_output.add_msg("No valid sample points found to display on the map.")

