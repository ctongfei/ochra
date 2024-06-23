import math
from dataclasses import replace

from ochra.plane import Point
from ochra.marker import Marker
from ochra.rect import AxisAlignedRectangle
from ochra.poly import Polygon, Polyline
from ochra.conic import Circle
from ochra.style import Fill
from ochra.style.stroke import LineJoin, Stroke
from ochra.util.functions import deg_to_rad


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

