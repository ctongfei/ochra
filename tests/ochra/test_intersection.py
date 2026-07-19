import ochra as ox


def test_segment_intersection():
    assert ox.intersect_segment_segment(
        ox.LineSegment((0, 0), (2, 2)), ox.LineSegment((1, 1), (3, 3))
    ) == ox.LineSegment((1, 1), (2, 2))

    assert ox.intersect_segment_segment(ox.LineSegment((0, 0), (2, 2)), ox.LineSegment((2, 0), (0, 2))) == ox.Point.mk(
        (1, 1)
    )

    assert ox.intersect_segment_segment(ox.LineSegment((0, 0), (2, 0)), ox.LineSegment((0, 2), (2, 2))) is None


def test_collinear_segment_overlap_uses_line_parameterization():
    cases = [
        (((0, 0), (4, 4)), ((2, 2), (6, 6)), ox.LineSegment((2, 2), (4, 4))),
        (((4, 4), (0, 0)), ((2, 2), (6, 6)), ox.LineSegment((4, 4), (2, 2))),
        (((1, -2), (1, 4)), ((1, 0), (1, 8)), ox.LineSegment((1, 0), (1, 4))),
        (((0, 0), (6, 3)), ((2, 1), (4, 2)), ox.LineSegment((2, 1), (4, 2))),
    ]
    for (p0, p1), (q0, q1), expected in cases:
        assert ox.intersect_segment_segment(ox.LineSegment(p0, p1), ox.LineSegment(q0, q1)) == expected


def test_collinear_segment_touching_and_disjoint_cases():
    touching = ox.intersect_segment_segment(ox.LineSegment((0, 0), (2, 1)), ox.LineSegment((2, 1), (4, 2)))
    disjoint = ox.intersect_segment_segment(ox.LineSegment((0, 0), (2, 1)), ox.LineSegment((4, 2), (6, 3)))

    assert touching == ox.Point.mk((2, 1))
    assert disjoint is None
