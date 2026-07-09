"""The categorical colormap must show every sampleable class (incl. 0 and >255)."""

from component.widget.map import _build_class_colormap


def test_colormap_colors_class_zero():
    cm = _build_class_colormap({0: "#ff0000", 5: "#00ff00"})
    assert cm[0] == (255, 0, 0, 255)  # class 0 is a real class, not background
    assert cm[5] == (0, 255, 0, 255)


def test_colormap_colors_codes_above_255():
    cm = _build_class_colormap({300: "#0000ff", 1024: "#ffffff"})
    assert cm[300] == (0, 0, 255, 255)
    assert cm[1024] == (255, 255, 255, 255)


def test_colormap_leaves_unknown_values_transparent():
    cm = _build_class_colormap({5: "#00ff00"})
    assert cm[7] == (0, 0, 0, 0)
    assert cm[5] == (0, 255, 0, 255)
