from ochra.plane import Point
from ochra.marker import Marker
from ochra.rect import AxisAlignedRectangle
from ochra.group import Group
from ochra.poly import Polygon, Polyline
from ochra.conic import Circle


def circle(size: float = 5.0, **kwargs):
    return Marker(
        [Circle(radius=size, **kwargs)],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )


def polygon(n: int, size: float = 5.0, **kwargs):
    return Marker(
        [Polygon.regular(n, circumradius=size, **kwargs)],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )


def arrow_triangle(size: float = 5.0, **kwargs):
    return Marker(
        [Polygon.regular(3, circumradius=size, **kwargs).translate(-size, 0)],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )


def arrow_stealth(size: float = 5.0, **kwargs):
    triangle = Polygon.regular(3, circumradius=size, **kwargs)
    return Marker(
        [Polygon(
            [
                triangle.vertices[0],
                triangle.vertices[1],
                Point.origin,
                triangle.vertices[2]
            ],
            **kwargs
        ).translate(-size, 0)],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )


def arrow_line(size: float = 5.0, **kwargs):
    triangle = Polygon.regular(3, circumradius=size, **kwargs)
    return Marker(
        [Polyline(
            [
                triangle.vertices[1],
                triangle.vertices[0],
                triangle.vertices[2]
            ],
            **kwargs
        ).translate(-size, 0)],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )