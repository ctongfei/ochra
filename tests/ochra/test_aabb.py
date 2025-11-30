import pytest

import ochra as ox
from .test_common import finite_parametric_test_cases


def check_aabb(e: ox.Parametric):
    aabb = e.aabb()
    assert aabb is not None
    points = e.approx_as_polyline().points.points
    ok = [p in aabb for p in points]
    assert all(ok)


@pytest.mark.parametrize("elem_type", finite_parametric_test_cases.keys())
def test_aabb(elem_type: type):
    """
    Checks that the axis-aligned bounding box contains all the points on the curve.
    """
    for e in finite_parametric_test_cases[elem_type]:
        check_aabb(e)
