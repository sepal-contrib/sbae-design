"""Tests for the upload modal styling (title, no nested cards, right Close)."""

import ipyvuetify as v
import solara

from component.model import app_state
from component.tile.upload import FilePreview, UploadTile
from component.widget.aoi_upload_selector import UploadDialogCard


def test_upload_dialog_card_has_title_and_right_aligned_close():
    _, rc = solara.render(
        UploadDialogCard(sbae_map=None, on_close=lambda: None), handle_error=False
    )

    title_text = " ".join(
        str(c) for w in rc.find(v.CardTitle).widgets for c in (w.children or [])
    )
    assert "Upload" in title_text

    labels = " ".join(
        str(c) for b in rc.find(v.Btn).widgets for c in (b.children or [])
    )
    assert "Close" in labels

    # Close sits in a CardActions row with a Spacer before it → right-aligned.
    assert rc.find(v.CardActions).widgets
    assert rc.find(v.Spacer).widgets


def test_upload_section_has_no_card_of_its_own():
    # The dialog provides the single card; the upload section must not add its
    # own solara.Card (that produced the nested-card look).
    app_state.uploaded_file_info.value = None
    app_state.file_path.value = None

    _, rc = solara.render(UploadTile(None), handle_error=False)

    rc.find(v.Card).assert_empty()


def test_file_preview_is_not_a_colored_alert():
    info = {
        "file_type": "raster",
        "size_mb": 67.8,
        "feature_count": 3_530_071_680,
        "crs": "EPSG:4326",
    }

    _, rc = solara.render(FilePreview(info), handle_error=False)

    # No colored info alert (the previous blue box) and no nested card.
    rc.find(v.Alert).assert_empty()
    rc.find(v.Card).assert_empty()
    text = " ".join(str(c) for w in rc.find(v.Html).widgets for c in (w.children or []))
    assert "File selected" in text
    assert "Raster" in text
    assert "EPSG:4326" in text
