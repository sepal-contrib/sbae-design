"""palette_for_codes: a stable categorical palette for class codes."""

from component.scripts.geospatial import palette_for_codes


def test_palette_for_codes_stable_regardless_of_input_order():
    assert palette_for_codes([3, 1, 2]) == palette_for_codes([1, 2, 3])


def test_palette_for_codes_distinct_within_base_length():
    p = palette_for_codes([1, 2, 3])
    assert len({p[1], p[2], p[3]}) == 3
    assert all(c.startswith("#") for c in p.values())


def test_palette_for_codes_cycles_when_exhausted():
    p = palette_for_codes(list(range(20)))
    assert len(p) == 20
    # the base palette has 9 colours, so the 10th sorted code wraps to the 1st
    assert p[0] == p[9]
