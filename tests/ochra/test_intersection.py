import pytest

import ochra as ox

def test_segment_intersection():
    assert ox.intersect_segment_segment(
        ox.LineSegment((0, 0), (2, 2)),
        ox.LineSegment((1, 1), (3, 3))
    ) == ox.LineSegment((1, 1), (2, 2))

    assert ox.intersect_segment_segment(
        ox.LineSegment((0, 0), (2, 2)),
        ox.LineSegment((2, 0), (0, 2))
    ) == ox.Point.mk((1, 1))

    assert ox.intersect_segment_segment(
        ox.LineSegment((0, 0), (2, 0)),
        ox.LineSegment((0, 2), (2, 2))
    ) == None
