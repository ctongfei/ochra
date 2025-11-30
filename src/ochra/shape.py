from collections.abc import Sequence

from ochra import LineSegment
from ochra.core import Rectangle, Polygon, Group, Element
from ochra.geometry import τ, Vector, Point, PointI
from ochra.style import Style


def asterisk(
        n: int,
        size: int = 4,
        angle: float = τ / 4,
        center: PointI = Point.origin,
        styles: Sequence[Style] = ()
) -> Element:
    """
    Draws an $n$-pointed asterisk. The asterisk is inscribed in a circle of diameter `size`.
    """
    return Group(
        [
            LineSegment((0, 0), (Vector.unit(angle + i * τ / n) * size / 2))
            for i in range(n)
        ],
        styles=styles,
    ).translate(center.x, center.y)


def cross(
        size: int = 4,
        center: PointI = Point.origin,
        styles: Sequence[Style] = ()
) -> Element:
    return asterisk(4, size, angle = τ / 8, center=center, styles=styles)


def plus(
        size: int = 4,
        center: PointI = Point.origin,
        styles: Sequence[Style] = ()
) -> Element:
    return asterisk(4, size, angle = 0, center=center, styles=styles)


def rounded_rectangle(
        bottom_left: PointI,
        top_right: PointI,
        radius: float,
        styles: Sequence[Style] = ()
) -> Element:
    raise NotImplementedError


def star():
    pass

