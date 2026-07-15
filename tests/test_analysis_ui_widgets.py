"""UI-widget tests for the analysis tab: current-table card and download menu."""

import ipyvuetify as v
import pandas as pd
import solara

from component.model import app_state
from component.model.state_manager import AppState
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
