"""Click handlers that write several reactives render once, not once per write.

A click runs outside any render batch, so each write used to trigger a render
pass of its own and the browser painted every half-updated state in between:
removing the classification map took the Design tab apart in visible stages.
"""

import threading

import ipyvuetify as v
import pandas as pd
import pytest
import solara

import component.tile.upload as upload
from component.message import msg
from component.model import app_state
from component.tile.class_editor import class_editor_table
from component.tile.upload import SampleMapButton
from component.widget.analysis_tab import AnalysisPanel
from component.widget.custom_widgets import use_batch
from component.widget.sample_configuration import SampleConfiguration


def _renders_during(rc, action):
    """Count the render passes ``action`` triggers from this thread.

    Workers report back through reactives too, and their renders land whenever
    they finish; counting only this thread keeps the number deterministic.
    """
    here = threading.current_thread()
    passes = []
    render = rc.render

    def counting(*args, **kwargs):
        if threading.current_thread() is here:
            passes.append(1)
        return render(*args, **kwargs)

    rc.render = counting
    try:
        action()
    finally:
        del rc.render
    return len(passes)


def _button(rc, label):
    (found,) = [b for b in rc.find(v.Btn).widgets if label in (b.children or [])]
    return found


@pytest.fixture
def mount(render_in_app):
    """Render components side by side, and leave ``app_state`` clean afterwards."""
    contexts = []

    def render(*factories):
        def tree():
            for factory in factories:
                factory()

        _, rc = render_in_app(tree)
        contexts.append(rc)
        return rc

    app_state.clear_all_sampling_data()
    app_state.sampling_method.value = "stratified"
    yield render
    for rc in contexts:
        rc.close()
    app_state.clear_all_sampling_data()


def _load_design():
    """The state a loaded classification map leaves behind."""
    app_state.uploaded_file_info.value = {"file_type": "raster", "size_mb": 1.0}
    app_state.file_path.value = "/tmp/classification.tif"
    app_state.area_data.value = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [600_000, 400_000],
            "map_edited_class": ["Forest", "Non forest"],
        }
    )
    app_state.class_colors.value = {1: "#00ff00", 2: "#ff0000"}
    app_state.expected_user_accuracies.value = {1: 0.9, 2: 0.9}
    app_state.eua_modes.value = {1: "high", 2: "high"}


# --- use_batch itself ---------------------------------------------------------


@pytest.fixture
def counters():
    return [solara.reactive(0) for _ in range(3)]


def _bump(counters):
    for counter in counters:
        counter.value += 1


def test_unbatched_click_renders_once_per_write(counters):
    """The control: without a batch the three writes really are three renders."""

    @solara.component
    def Demo():
        solara.Text(" ".join(str(c.value) for c in counters))
        solara.Button("go", on_click=lambda: _bump(counters))

    _, rc = solara.render(Demo(), handle_error=False)

    assert _renders_during(rc, _button(rc, "go").click) == 3


def test_batch_decorator_folds_the_writes_into_one_render(counters):
    @solara.component
    def Demo():
        batch = use_batch()

        @batch
        def go():
            _bump(counters)

        solara.Text(" ".join(str(c.value) for c in counters))
        solara.Button("go", on_click=go)

    _, rc = solara.render(Demo(), handle_error=False)

    assert _renders_during(rc, _button(rc, "go").click) == 1
    assert [c.value for c in counters] == [1, 1, 1]


def test_each_batch_block_renders_as_it_closes(counters):
    """What lets a handler show progress before slow work: see load_sample_map."""
    seen = []

    @solara.component
    def Demo():
        batch = use_batch()

        def go():
            with batch:
                _bump(counters)
            seen.append(rc.find(v.Html).widget.children)
            with batch:
                _bump(counters)

        solara.Text(" ".join(str(c.value) for c in counters))
        solara.Button("go", on_click=go)

    _, rc = solara.render(Demo(), handle_error=False)

    assert _renders_during(rc, _button(rc, "go").click) == 2
    assert seen == [["1 1 1"]]


def test_batch_lets_the_error_through_and_still_renders(counters):
    @solara.component
    def Demo():
        batch = use_batch()

        @batch
        def go():
            _bump(counters)
            raise RuntimeError("boom")

        solara.Text(" ".join(str(c.value) for c in counters))
        solara.Button("go", on_click=go)

    _, rc = solara.render(Demo(), handle_error=False)

    with pytest.raises(RuntimeError, match="boom"):
        _button(rc, "go").click()
    assert rc.find(v.Html).widget.children == ["1 1 1"]


# --- the app's handlers -------------------------------------------------------


def test_removing_the_map_is_one_render_pass(mount):
    _load_design()
    rc = mount(SampleConfiguration)
    # the design was computed, so the charts that make the teardown costly exist
    assert app_state.sample_results.value

    passes = _renders_during(rc, rc.find(v.Btn, color="error").widget.click)

    assert app_state.file_path.value is None
    assert app_state.sample_results.value is None
    assert passes == 1


def test_sample_map_shows_its_spinner_before_reading_the_raster(mount, monkeypatch):
    """Batching the whole handler would hold the spinner back until the work is done."""
    loading = solara.reactive(False)
    rc = mount(SampleConfiguration, lambda: SampleMapButton(is_loading=loading))
    sample_map = _button(rc, msg("upload.sample_map"))
    spinner_while_reading = []
    load_classification_source = upload.load_classification_source

    def reading(*args, **kwargs):
        spinner_while_reading.append(sample_map.loading)
        return load_classification_source(*args, **kwargs)

    monkeypatch.setattr(upload, "load_classification_source", reading)

    passes = _renders_during(rc, sample_map.click)

    assert spinner_while_reading == [True]
    assert passes == 2
    assert not app_state.area_data.value.empty
    assert sample_map.loading is False


def test_changing_a_class_accuracy_mode_is_one_render_pass(mount):
    _load_design()
    rc = mount(SampleConfiguration, class_editor_table)
    low = [
        b
        for b in rc.find(v.Btn).widgets
        if msg("design.class_editor.mode_low") in (b.children or [])
    ]

    passes = _renders_during(rc, low[0].click)

    assert app_state.eua_modes.value[1] == "low"
    assert passes == 1


def test_loading_the_example_analysis_data_is_one_render_pass(mount):
    rc = mount(AnalysisPanel)
    _button(rc, msg("analysis.reference.upload_button")).click()  # opens the dialog

    passes = _renders_during(
        rc, _button(rc, msg("analysis.reference.example_button")).click
    )

    assert not app_state.analysis_reference_df.value.empty
    assert passes == 1
