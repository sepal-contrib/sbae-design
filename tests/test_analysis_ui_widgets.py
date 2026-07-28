"""UI-widget tests for the analysis tab: current-table card and download menu."""

import ipyvuetify as v
import numpy as np
import pandas as pd
import rasterio
import solara
from rasterio.transform import from_origin

from component.analysis.service import AnalysisService
from component.model import app_state
from component.model.state_manager import AppState
from component.widget import analysis_tab
from component.widget.analysis_results import _ConfusionMatrix
from component.widget.analysis_tab import (
    AnalysisPanel,
    CurrentTableDisplay,
    _ReferenceUploadDialog,
)
from component.widget.custom_widgets import DownloadMenu, Section


def _html_text(rc) -> str:
    return " ".join(str(c) for w in rc.find(v.Html).widgets for c in (w.children or []))


def test_current_table_display_shows_name_and_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    _, rc = solara.render(
        CurrentTableDisplay(
            title="Reference table", df=df, name="ref.csv", on_clear=lambda: None
        ),
        handle_error=False,
    )

    text = _html_text(rc)
    assert "Reference table" in text
    assert "ref.csv" in text
    assert "3 rows" in text
    assert "2 columns" in text
    # a single clear button is present
    rc.find(v.Btn).assert_single()


def test_current_table_display_empty_renders_nothing():
    _, rc = solara.render(
        CurrentTableDisplay(title="Reference table", df=pd.DataFrame()),
        handle_error=False,
    )

    rc.find(v.Btn).assert_empty()


def test_download_menu_single_button_with_one_item_per_file():
    items = [
        ("Error matrix", "c,v\n1,2\n", "cm.csv"),
        ("Area estimates", "c,v\n1,2\n", "area.csv"),
        ("Accuracy table", "c,v\n1,2\n", "acc.csv"),
    ]

    _, rc = solara.render(DownloadMenu(items), handle_error=False)

    # Exactly ONE activator button (not one-per-file).
    buttons = rc.find(v.Btn).widgets
    assert len(buttons) == 1
    assert any("Download" in str(c) for c in (buttons[0].children or []))
    # One menu row per file.
    assert len(rc.find(v.ListItem).widgets) == 3


def test_download_menu_skips_items_without_data():
    items = [
        ("Has data", "c,v\n1,2\n", "a.csv"),
        ("No data", "", "b.csv"),
        ("Also none", None, "c.csv"),
    ]

    _, rc = solara.render(DownloadMenu(items), handle_error=False)

    assert len(rc.find(v.ListItem).widgets) == 1


def test_download_menu_empty_renders_nothing():
    _, rc = solara.render(DownloadMenu([]), handle_error=False)

    rc.find(v.Btn).assert_empty()


def test_reference_upload_dialog_has_title_and_close():
    ref_path = solara.reactive(None)

    _, rc = solara.render(
        _ReferenceUploadDialog(ref_path, on_close=lambda: None), handle_error=False
    )

    title = " ".join(
        str(c) for w in rc.find(v.CardTitle).widgets for c in (w.children or [])
    )
    assert "Upload reference data" in title
    labels = " ".join(
        str(c) for b in rc.find(v.Btn).widgets for c in (b.children or [])
    )
    assert "Close" in labels


def test_analysis_panel_shows_small_upload_button_when_no_reference():
    app_state.clear_analysis_data()
    app_state.area_data.value = None

    _, rc = solara.render(AnalysisPanel(), handle_error=False)

    upload_btns = [
        b
        for b in rc.find(v.Btn).widgets
        if any("Upload reference data" in str(c) for c in (b.children or []))
    ]
    assert len(upload_btns) == 1
    assert upload_btns[0].small is True


def test_analysis_panel_has_intro_help_button():
    app_state.clear_analysis_data()
    app_state.area_data.value = None

    _, rc = solara.render(AnalysisPanel(), handle_error=False)

    icons = [str(c) for i in rc.find(v.Icon).widgets for c in (i.children or [])]
    assert "mdi-help-circle-outline" in icons


def test_analysis_panel_shows_hint_alert_when_empty():
    app_state.clear_analysis_data()
    app_state.area_data.value = None

    _, rc = solara.render(AnalysisPanel(), handle_error=False)

    alert_texts = " ".join(
        str(c) for a in rc.find(v.Alert).widgets for c in (a.children or [])
    )
    assert "reference table" in alert_texts.lower()


def test_section_uses_theme_aware_classes_not_hardcoded_colors():
    _, rc = solara.render(
        Section("Summary", "mdi-progress-check", "A description"),
        handle_error=False,
    )

    htmls = rc.find(v.Html).widgets
    title = next(w for w in htmls if "Summary" in (w.children or []))
    assert "subtitle-2" in (title.class_ or "")
    # Description inherits the theme text color rather than a hardcoded gray.
    desc = next(w for w in htmls if "A description" in (w.children or []))
    assert "body-2" in (desc.class_ or "")
    assert "text--secondary" not in (desc.class_ or "")
    assert "#" not in (desc.style_ or "")
    assert any(
        "mdi-progress-check" in (i.children or []) for i in rc.find(v.Icon).widgets
    )


def test_analysis_results_use_sections_not_cards():
    results = {
        "confusion_matrix": {
            "columns": [1, 2],
            "index": [1, 2],
            "data": [[5, 1], [2, 4]],
        }
    }

    _, rc = solara.render(_ConfusionMatrix(results), handle_error=False)

    # No card in the right-side panel — a themed Section header instead.
    rc.find(v.Card).assert_empty()
    texts = {c for w in rc.find(v.Html).widgets for c in (w.children or [])}
    assert "Error matrix" in texts


def test_section_without_description_renders_only_title_text():
    _, rc = solara.render(
        Section("Generate Points", "mdi-map-marker-multiple"), handle_error=False
    )

    text_spans = [
        w
        for w in rc.find(v.Html).widgets
        if any(isinstance(c, str) for c in (w.children or []))
    ]
    assert len(text_spans) == 1
    assert "Generate Points" in (text_spans[0].children or [])


def test_column_mapping_hides_map_role_for_map_source(monkeypatch):
    from component.widget import analysis_tab

    st = AppState()
    st.analysis_area_source.value = "map"
    st.analysis_column_mapping.value = {}
    monkeypatch.setattr(analysis_tab, "app_state", st)
    _, rc = solara.render(
        analysis_tab._ColumnMappingCard(["a", "b"], area_source="map"),
        handle_error=False,
    )
    # Select labels are a v.Select widget trait (rendered client-side by
    # Vuetify), not v.Html children, so inspect the Select widgets directly.
    labels = " ".join(str(s.label) for s in rc.find(v.Select).widgets)
    assert "Reference class" in labels
    assert "Map / predicted" not in labels  # map role hidden when the map derives it


def test_derive_map_source_surfaces_error_without_xy_mapping(tmp_path):
    """derive_map_source must not raise when x/y columns aren't mapped.

    The Calculate action runs this off the UI thread; a raised exception would
    abort the task. Instead the error is surfaced via ``app_state.add_error``
    and the reference table is left untouched (no partial map_code column).
    """
    raster_path = tmp_path / "classification.tif"
    raster_path.write_bytes(b"")  # never opened: the x/y check raises first

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"x": [1, 2], "y": [3, 4], "ref_code": [1, 2]}
    )
    st.analysis_column_mapping.value = {}  # x/y NOT mapped yet
    st.analysis_classification_path.value = str(raster_path)

    result = analysis_tab.derive_map_source(st, None)  # must not raise

    assert result is None
    assert any(
        "x/y column mapping" in msg for msg in st.error_messages.value
    ), st.error_messages.value
    # reference table untouched -- no map_code column was added
    assert "map_code" not in st.analysis_reference_df.value.columns


def test_analysis_panel_accepts_sbae_map():
    import inspect

    from component.widget.analysis_tab import AnalysisPanel

    fn = getattr(AnalysisPanel, "f", AnalysisPanel)
    assert "sbae_map" in inspect.signature(fn).parameters


class _FakeSbaeMap:
    """Records add_class_raster/add_sample_points calls instead of a real map."""

    def __init__(self):
        self.class_raster_calls = []
        self.sample_points_calls = []
        self.reference_points_calls = []

    def add_class_raster(self, path, class_colors, layer_name, key):
        self.class_raster_calls.append(
            {
                "path": path,
                "class_colors": class_colors,
                "layer_name": layer_name,
                "key": key,
            }
        )

    def add_sample_points(self, points_df, class_colors=None):
        self.sample_points_calls.append((points_df, class_colors))

    def add_reference_points(self, points_df, **kwargs):
        self.reference_points_calls.append((points_df, kwargs))


def _write_2x2_class_raster(raster_path):
    """4x4 raster of four 2x2 class blocks (1 top-left, 2 top-right, 3, 4)."""
    data = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(data, 1)


def test_derive_map_source_renders_raster_layer(tmp_path):
    """derive_map_source samples the raster, fills map_code, and adds the layer.

    Also derives class colors from the raster (standalone mode). Real rasterio
    round-trip (same fixture as ``test_derive_from_classification``) with a fake
    ``sbae_map`` -- exercises the actual success path, no rendering.
    """
    raster_path = tmp_path / "clas.tif"
    _write_2x2_class_raster(raster_path)

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"lon": [0.5, 2.5], "lat": [3.5, 0.5], "ref_code": [1, 2]}
    )
    st.analysis_column_mapping.value = {"x": "lon", "y": "lat", "ref": "ref_code"}
    st.analysis_classification_path.value = str(raster_path)
    # Standalone mode: class_colors starts EMPTY, as it would with no
    # design-step upload. derive_map_source must derive it from the raster.
    assert st.class_colors.value == {}

    fake_map = _FakeSbaeMap()
    dropped = analysis_tab.derive_map_source(st, fake_map)

    assert dropped == 0
    assert len(fake_map.class_raster_calls) == 1
    call = fake_map.class_raster_calls[0]
    assert call["path"] == str(raster_path)
    assert call["class_colors"] == st.class_colors.value
    assert call["key"] == "clas_an"
    # The map layer must never fall back to a continuous colormap: with no
    # design-step upload, class_colors starts empty and derive_map_source must
    # derive it from the raster (one entry per class present in the raster).
    assert call["class_colors"], "class_colors must not be empty (near-black map)"
    assert set(call["class_colors"]) == {1, 2, 3, 4}
    # map_code filled on the reference + mapping updated to the derived column
    assert st.analysis_column_mapping.value["map"] == "map_code"
    assert st.analysis_reference_df.value["map_code"].tolist() == [1, 4]
    # Reference points are drawn by the AnalysisPanel's render thread (on their
    # own "ref_pts" layer), not by the derivation.
    assert fake_map.sample_points_calls == []


def test_derive_map_source_without_sbae_map(tmp_path):
    """No sbae_map -> derivation still fills map_code; no AttributeError."""
    raster_path = tmp_path / "clas.tif"
    _write_2x2_class_raster(raster_path)

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"lon": [0.5, 2.5], "lat": [3.5, 0.5], "ref_code": [1, 2]}
    )
    st.analysis_column_mapping.value = {"x": "lon", "y": "lat", "ref": "ref_code"}
    st.analysis_classification_path.value = str(raster_path)

    dropped = analysis_tab.derive_map_source(st, None)  # sbae_map=None

    assert dropped == 0
    assert st.analysis_reference_df.value["map_code"].tolist() == [1, 4]
    assert not st.error_messages.value


# ---- explicit Calculate flow: signature, freshness, button, source labels ----


def test_inputs_signature_changes_when_an_input_changes():
    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame({"m": [1], "r": [1]})
    st.analysis_reference_name.value = "ref.csv"
    sig1 = AnalysisService.inputs_signature(st)
    st.analysis_confidence_level.value = 90.0
    assert AnalysisService.inputs_signature(st) != sig1


def test_inputs_signature_stable_when_nothing_changes():
    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame({"m": [1], "r": [1]})
    assert AnalysisService.inputs_signature(st) == AnalysisService.inputs_signature(st)


def test_results_are_fresh_tracks_input_edits():
    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame({"m": [1], "r": [1]})
    st.set_analysis_results(
        {"overall_accuracy": 0.9}, signature=AnalysisService.inputs_signature(st)
    )
    assert analysis_tab._results_are_fresh(st) is True
    # any later edit invalidates the stored results -> dashboard hidden
    st.analysis_confidence_level.value = 90.0
    assert analysis_tab._results_are_fresh(st) is False


def test_set_analysis_results_none_clears_signature():
    st = AppState()
    st.set_analysis_results({"x": 1}, signature=("sig",))
    st.set_analysis_results(None)
    assert st.analysis_results.value is None
    assert st.analysis_results_signature.value is None


def test_run_calculation_design_source_sets_fresh_results():
    st = AppState()
    st.area_data.value = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [600000.0, 400000.0],
            "map_edited_class": ["Forest", "NonForest"],
        }
    )
    rows = [(1, 1)] * 45 + [(1, 2)] * 5 + [(2, 1)] * 10 + [(2, 2)] * 40
    st.analysis_reference_df.value = pd.DataFrame(rows, columns=["mapc", "refc"])
    st.analysis_column_mapping.value = {"map": "mapc", "ref": "refc"}
    st.analysis_area_source.value = "design"

    analysis_tab.run_calculation(st, None)

    assert st.analysis_results.value is not None
    assert analysis_tab._results_are_fresh(st) is True


def test_run_calculation_without_inputs_reports_error_and_stays_blank():
    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame()  # nothing loaded
    analysis_tab.run_calculation(st, None)
    assert st.analysis_results.value is None
    assert st.error_messages.value  # a validation error surfaced


def test_area_source_labels_are_bijective():
    assert set(analysis_tab._AREA_SOURCE_LABELS) == {"design", "upload", "map"}
    # order presented to the user: design map, upload a map, area CSV
    assert analysis_tab._AREA_SOURCE_ORDER == ["design", "map", "upload"]
    for key, label in analysis_tab._AREA_SOURCE_LABELS.items():
        assert analysis_tab._AREA_SOURCE_BY_LABEL[label] == key


def test_area_source_select_shows_friendly_labels_not_raw_keys(monkeypatch):
    st = AppState()
    st.analysis_area_source.value = "map"
    monkeypatch.setattr(analysis_tab, "app_state", st)
    _, rc = solara.render(analysis_tab._AnalysisControls(), handle_error=False)
    src = next(s for s in rc.find(v.Select).widgets if "source" in str(s.label).lower())
    # current selection renders as the friendly label, never the raw key
    assert src.v_model == "Upload a classification map"
    items = [str(x) for x in (src.items or [])]
    assert "Upload a classification map" in items
    assert "map" not in items and "design" not in items


def test_design_source_shows_classification_card(monkeypatch, tmp_path):
    st = AppState()
    st.analysis_area_source.value = "design"
    st.file_path.value = str(tmp_path / "my_design.tif")
    st.area_data.value = pd.DataFrame(
        {"map_code": [1, 2, 3], "map_area": [1.0, 2.0, 3.0]}
    )
    monkeypatch.setattr(analysis_tab, "app_state", st)
    _, rc = solara.render(analysis_tab._AnalysisControls(), handle_error=False)
    text = _html_text(rc)
    assert "Design map:" in text
    assert "my_design.tif" in text
    assert "3 classes" in text


def test_classification_map_upload_shows_card_when_path_set(monkeypatch, tmp_path):
    raster = tmp_path / "clas.tif"
    raster.write_bytes(b"")
    st = AppState()
    st.analysis_classification_path.value = str(raster)
    monkeypatch.setattr(analysis_tab, "app_state", st)
    _, rc = solara.render(analysis_tab._ClassificationMapUpload(), handle_error=False)
    text = _html_text(rc)
    assert "Classification map:" in text
    assert "clas.tif" in text
    rc.find(v.Btn).assert_single()  # the clear button


def test_analysis_panel_shows_calculate_button_when_reference_loaded(monkeypatch):
    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame({"mapc": [1, 2], "refc": [1, 2]})
    st.analysis_reference_name.value = "ref.csv"
    st.analysis_column_mapping.value = {"map": "mapc", "ref": "refc"}
    st.analysis_area_source.value = "design"
    st.area_data.value = pd.DataFrame({"map_code": [1, 2], "map_area": [1.0, 2.0]})
    monkeypatch.setattr(analysis_tab, "app_state", st)
    _, rc = solara.render(analysis_tab.AnalysisPanel(), handle_error=False)
    labels = " ".join(
        str(c) for b in rc.find(v.Btn).widgets for c in (b.children or [])
    )
    assert "Calculate" in labels
