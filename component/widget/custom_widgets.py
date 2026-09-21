"""SBAE UI Components Module.

Contains reusable Solara components for the SBAE application.
"""

import contextlib

import reacton.core
import solara
import solara.lab
from solara.alias import rv

from component.message import msg


class _Batch(contextlib.ContextDecorator):
    """What ``use_batch`` returns: a reusable, re-entrant render batch."""

    def __init__(self, render_context):
        self._render_context = render_context

    def __enter__(self):
        self._render_context.__enter__()
        return self

    def __exit__(self, *exc_info):
        self._render_context.__exit__(*exc_info)


def use_batch():
    """Return a batch that folds the reactive writes made inside it into one render.

    Outside a render, reacton renders synchronously on every state change. It
    only batches trait syncs coming from the browser; a click arrives as a custom
    message, so a handler that writes N reactives renders N times and the browser
    paints each half-updated state in between.

    Decorate a handler with ``@batch``, or wrap part of one in ``with batch:``.
    The render happens when the batch closes, so a handler that must show
    progress before slow work opens one batch for the progress state and another
    for the results. Call this while rendering, before any early return.
    """
    return _Batch(reacton.core.get_render_context())


@solara.component
def Section(
    title: str | None = None,
    icon: str | None = None,
    description: str | None = None,
):
    """A theme-aware section header (icon + title + optional description).

    Mirrors pysepal's RightPanel section styling with Vuetify theme classes and
    CSS variables (``subtitle-2``, ``--v-divider-base``) instead of hardcoded
    colors, so it adapts to light/dark themes. Render it before the section's
    content.

    Args:
        title: Section title (shown with theme typography).
        icon: Optional leading MDI icon name.
        description: Optional theme-aware description line.
    """
    if title or icon:
        with solara.Row(
            style=(
                "align-items: center; gap: 8px; padding: 8px 0; margin-bottom: 12px; "
                "border-bottom: 1px solid var(--v-divider-base, rgba(0, 0, 0, 0.12));"
            )
        ):
            if icon:
                solara.v.Icon(small=True, children=[icon])
            if title:
                solara.Text(title, classes=["subtitle-2", "font-weight-medium"])
    if description:
        solara.Text(
            description,
            classes=["body-2"],
            style="padding-left: 16px; margin-bottom: 12px; display: block;",
        )


@solara.component
def DownloadMenu(
    items,
    label: str | None = None,
    icon_name: str = "mdi-download",
    mime_type: str = "text/csv",
):
    """A single dropdown button listing one download per file.

    Replaces rows of individual download buttons with one "Download" button
    that opens a menu with one row per file.

    Args:
        items: Iterable of ``(label, data, filename)`` tuples, or
            ``(label, data, filename, mime_type)`` to override the MIME type per
            file. Entries whose ``data`` is falsy (empty/None) are skipped;
            ``data`` may be ``str`` or ``bytes``.
        label: Text shown on the activator button; defaults to "Download".
        icon_name: Icon shown on the activator button.
        mime_type: Default MIME type applied when an item does not specify one.
    """
    valid = [item for item in items if item[1]]
    if not valid:
        return

    activator = solara.Button(
        label or msg("common.download"),
        icon_name=icon_name,
        outlined=True,
        color="primary",
    )
    with solara.lab.Menu(activator=activator):
        with solara.v.List(dense=True):
            for lbl, data, filename, *rest in valid:
                payload = data.encode() if isinstance(data, str) else data
                with solara.FileDownload(
                    data=payload,
                    filename=filename,
                    mime_type=rest[0] if rest else mime_type,
                ):
                    with solara.v.ListItem():
                        solara.v.ListItemIcon(
                            children=[
                                solara.v.Icon(children=["mdi-file-download-outline"])
                            ]
                        )
                        solara.v.ListItemTitle(children=[lbl])


def error_display(error_message: str, error_type: str = "error") -> None:
    """Display error messages with appropriate styling.

    Args:
        error_message: Error message to display
        error_type: Type of error (error, warning, info)
    """
    with rv.Alert(type=error_type, text=True):
        with solara.Row(gap="4px", style="align-items: center;"):
            solara.Text(f"{error_type.title()}:", style="font-weight: bold;")
            solara.Text(error_message)


def info_panel(title: str, content: str, collapsible: bool = True) -> None:
    """Display informational content in an expandable panel.

    Args:
        title: Panel title
        content: Panel content (markdown)
        collapsible: Whether panel can be collapsed
    """
    if collapsible:
        with solara.Details(title):
            solara.Markdown(content)
    else:
        with solara.Card(title):
            solara.Markdown(content)
