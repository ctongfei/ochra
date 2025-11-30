import ochra as ox
from ochra.geometry import τ


global_viewport = ox.AxisAlignedRectangle((0, 0), (100, 100))


finite_parametric_test_cases: dict[type, list] = {
    ox.LineSegment: [
        ox.LineSegment((1, 1), (99, 99)),
        ox.LineSegment((0, 80), (80, 10)),
        ox.LineSegment((0, 10), (100, 80)),
    ],
    ox.Circle: [
        ox.Circle(10, (20, 20)),
        ox.Circle(40, (50, 50)),
    ],
    ox.Ellipse: [
        ox.Ellipse.from_foci_and_major_axis((10, 10), (40, 40), 55),
    ],
    ox.QuadraticBezierCurve: [
        ox.QuadraticBezierCurve((0, 0), (50, 100), (100, 0)),
        ox.QuadraticBezierCurve((50, 80), (10, 10), (80, 50)),
    ],
    ox.CubicBezierCurve: [
        ox.CubicBezierCurve((0, 0), (20, 80), (80, 20), (100, 100)),
    ],
    ox.Polyline: [
        ox.Polyline([(0, 0), (50, 100), (100, 0)]),
        ox.Polyline([(50, 80), (10, 10), (80, 50)]),
    ],
    ox.Polygon: [
        ox.Polygon.regular(3, center=(30, 30), circumradius=20),
        ox.Polygon.regular(5, center=(50, 50), circumradius=50),
        ox.Polygon.regular_star(5, 2, center=(50, 50), circumradius=50),
    ],
}


infinite_parametric_test_cases: dict[type, list] = {
    ox.Line: [
        ox.Line((1, 1, 1)),
        ox.Line((1, 0, 1)),
        ox.Line((0, 1, 1)),
    ],
    ox.Ray: [
        ox.Ray((0, 0), 0),
        ox.Ray((0, 0), τ / 8),
        ox.Ray((0, 0), τ / 3),
    ],
    ox.Parabola: [],
    ox.Hyperbola: [],
}


implicit_parametric_test_cases: dict[type, list] = {
    ox.Line: [
        ox.Line((1, 1, 1)),
        ox.Line((1, 0, 1)),
        ox.Line((0, 1, 1)),
    ],
    ox.Circle: [
        ox.Circle(10, (50, 50)),
    ],
    ox.Ellipse: [
        ox.Ellipse.standard(10, 5),
    ],
    ox.Parabola: [],
    ox.Hyperbola: [],
}
