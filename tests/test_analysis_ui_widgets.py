"""UI-widget tests for the analysis tab: current-table card and download menu."""

import asyncio

import ipyvuetify as v
import numpy as np
import pandas as pd
import rasterio
import solara
from rasterio.transform import from_origin

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
    # Description is theme-aware muted text, not a hardcoded gray.
    desc = next(w for w in htmls if "A description" in (w.children or []))
    assert "text--secondary" in (desc.class_ or "")
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


def _walk_widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()) or ():
        yield from _walk_widgets(child)


def _run_with_task_loop(coro_factory):
    """Run ``coro_factory()`` on a real event loop, then restore loop state.

    ``asyncio.run`` unconditionally clears the process' "current" event loop
    when it tears down, which sticks for the rest of the test session and
    breaks later tests that rely on ``asyncio.get_event_loop()``'s legacy
    auto-create fallback (e.g. solara's task runner). Save whatever loop was
    current beforehand and restore it afterward so this test stays isolated.
    """
    try:
        previous_loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous_loop = None
    try:
        return asyncio.run(coro_factory())
    finally:
        asyncio.set_event_loop(previous_loop)


def test_classification_map_upload_survives_derivation_error(monkeypatch, tmp_path):
    """A raster picked before x/y mapping must not crash the whole widget tree.

    Regression test for a bug where ``derive_from_classification`` raising
    inside the ``use_task`` derivation (e.g. because x/y columns aren't
    mapped yet) propagated out of render: with reacton's default
    ``handle_error=True`` the whole widget tree -- clear button included --
    gets replaced by a raw traceback ``ipywidgets.HTML``, wedging the page.
    Verified empirically against the pre-fix code: the tree collapsed to a
    single traceback widget and the clear button vanished. Post-fix the
    error is caught, surfaced via ``status`` + ``app_state.add_error``, and
    the normal widget tree (clear button included) is preserved.
    """
    raster_path = tmp_path / "classification.tif"
    raster_path.write_bytes(b"")  # never opened: the x/y check raises first

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"x": [1, 2], "y": [3, 4], "ref_code": [1, 2]}
    )
    st.analysis_column_mapping.value = {}  # x/y NOT mapped yet
    st.analysis_classification_path.value = str(raster_path)
    monkeypatch.setattr(analysis_tab, "app_state", st)

    async def _runner():
        element = analysis_tab._ClassificationMapUpload.widget()
        # Let the use_task's background thread run the derivation and the
        # resulting re-render settle (see pysepal's export-test pattern).
        for _ in range(20):
            await asyncio.sleep(0.05)
        return element

    element = _run_with_task_loop(_runner)

    widgets = list(_walk_widgets(element))
    # No raw traceback widget replaced the tree.
    assert not [w for w in widgets if type(w).__name__ == "HTML"]
    # The clear (mdi-close) button is still there -- the session isn't stuck.
    assert [w for w in widgets if isinstance(w, v.Btn)]
    # The error is surfaced through the app's error channel.
    assert any(
        "x/y column mapping" in msg for msg in st.error_messages.value
    ), st.error_messages.value


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

    def add_class_raster(self, path, class_colors, layer_name, key):
        self.class_raster_calls.append(
            {
                "path": path,
                "class_colors": class_colors,
                "layer_name": layer_name,
                "key": key,
            }
        )

    def add_sample_points(self, points_df):
        self.sample_points_calls.append(points_df)


def test_classification_map_upload_renders_layers_on_success(monkeypatch, tmp_path):
    """A successful derivation adds the classification raster + ref points to sbae_map.

    Drives the real ``use_task`` derivation (real rasterio round-trip, same
    fixture as ``test_derive_from_classification.py``) with a fake ``sbae_map``
    standing in for ``SbaeMap``, so this exercises the actual success-path
    wiring rather than asserting it by inspection.
    """
    data = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    raster_path = tmp_path / "clas.tif"
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

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"lon": [0.5, 2.5], "lat": [3.5, 0.5], "ref_code": [1, 2]}
    )
    st.analysis_column_mapping.value = {"x": "lon", "y": "lat", "ref": "ref_code"}
    st.analysis_classification_path.value = str(raster_path)
    # Standalone mode: class_colors starts EMPTY, as it would with no
    # design-step upload. run_derivation must derive it from the raster
    # (see test assertions below) instead of leaving it empty.
    assert st.class_colors.value == {}
    monkeypatch.setattr(analysis_tab, "app_state", st)

    fake_map = _FakeSbaeMap()

    async def _runner():
        element = analysis_tab._ClassificationMapUpload.widget(sbae_map=fake_map)
        # Let the use_task's background thread run the derivation and the
        # resulting re-render settle (see pysepal's export-test pattern).
        for _ in range(20):
            await asyncio.sleep(0.05)
        return element

    _run_with_task_loop(_runner)

    assert len(fake_map.class_raster_calls) == 1
    call = fake_map.class_raster_calls[0]
    assert call["path"] == str(raster_path)
    assert call["class_colors"] == st.class_colors.value
    assert call["key"] == "clas_an"
    # The map layer must never fall back to a continuous colormap: with no
    # design-step upload, class_colors starts empty and run_derivation must
    # derive it from the raster (one entry per class present in the raster).
    assert call["class_colors"], "class_colors must not be empty (near-black map)"
    assert set(call["class_colors"]) == {1, 2, 3, 4}

    assert len(fake_map.sample_points_calls) == 1
    points_df = fake_map.sample_points_calls[0]
    assert set(points_df.columns) >= {"latitude", "longitude", "map_code"}
    assert points_df["map_code"].tolist() == [1, 4]
    assert points_df["longitude"].tolist() == [0.5, 2.5]
    assert points_df["latitude"].tolist() == [3.5, 0.5]


def test_classification_map_upload_skips_layers_without_sbae_map(monkeypatch, tmp_path):
    """No sbae_map -> derivation still succeeds; no AttributeError from a None map."""
    data = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    raster_path = tmp_path / "clas.tif"
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

    st = AppState()
    st.analysis_reference_df.value = pd.DataFrame(
        {"lon": [0.5, 2.5], "lat": [3.5, 0.5], "ref_code": [1, 2]}
    )
    st.analysis_column_mapping.value = {"x": "lon", "y": "lat", "ref": "ref_code"}
    st.analysis_classification_path.value = str(raster_path)
    monkeypatch.setattr(analysis_tab, "app_state", st)

    async def _runner():
        element = analysis_tab._ClassificationMapUpload.widget()  # sbae_map=None
        for _ in range(20):
            await asyncio.sleep(0.05)
        return element

    _run_with_task_loop(_runner)

    assert st.analysis_reference_df.value["map_code"].tolist() == [1, 4]
    assert not st.error_messages.value
