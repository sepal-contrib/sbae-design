"""Widgets whose translation needs more than a catalog lookup.

The map legend is composed off the UI thread and translated at render time, so
it breaks in ways a plain string swap would not.
"""

import ipyvuetify as v
import pytest
import solara
from pysepal.solara import get_current_locale_state

from component.message import get_translator
from component.model import app_state
from component.widget import map as map_mod
from component.widget.map import PointsLegend


@pytest.fixture(autouse=True)
def _reset_locale():
    """Keep a locale switch from leaking into the rest of the suite."""
    state = get_current_locale_state()
    yield
    state.set_locale("en")


def _legend_labels(rc):
    # LegendComponent is a component_vue function, so it renders as a dynamic
    # VuetifyTemplate subclass that cannot be matched by class.
    widget = next(
        w for w in rc.find(v.VuetifyTemplate).widgets if hasattr(w, "legend_data")
    )
    return [entry["label"] for entry in widget.legend_data["items"]]


def test_points_legend_translates_the_composed_keys():
    """The overlay turns composer keys into text, so labels come from the catalog.

    ``_compose_points_legend`` runs in worker threads and can only emit keys;
    if the overlay stopped translating them the map would show "correct" and
    "incorrect" verbatim.
    """
    app_state.points_legend.value = map_mod._compose_points_legend(True, True, True)

    _, rc = solara.render(PointsLegend(), handle_error=False)

    ms = get_translator()
    assert _legend_labels(rc) == [
        ms.map.legend.sample,
        ms.map.legend.correct,
        ms.map.legend.incorrect,
    ]


def test_points_legend_relabels_on_a_language_change():
    """A legend already on the map must follow the language selector."""
    app_state.points_legend.value = map_mod._compose_points_legend(False, True, False)

    _, rc = solara.render(PointsLegend(), handle_error=False)
    assert _legend_labels(rc) == [get_translator().map.legend.reference]

    get_current_locale_state().set_locale("es")

    assert _legend_labels(rc) == [get_translator("es").map.legend.reference]
