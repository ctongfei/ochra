import math
from dataclasses import replace

from ochra.geometry import Point
from ochra.core import Polygon, Polyline, AxisAlignedRectangle, Marker
from ochra.style import Stroke, Fill, LineJoin


class Arrowhead:

    def __init__(self, marker: Marker):
        self.marker = marker

    def as_start_marker(self):
        raise NotImplementedError

    def as_end_marker(self):
        raise NotImplementedError


def _arrow_triangle_base(size: float, angle: float, **kwargs):
    return Polygon(
        [
            Point(size, 0),
            Point.polar(size, math.tau / 2 - angle),
            Point.polar(size, math.tau / 2 + angle)
        ],
        **kwargs
    ).translate(-size, 0)  # center at arrow tip


def arrow_triangle(size: float = 5.0, angle: float = math.degrees(30), **kwargs):
    stroke = replace(kwargs.pop("stroke", Stroke()))
    stroke.line_join = LineJoin.miter
    # https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-miterlimit
    stroke.miter_limit = math.ceil(1.0 / math.sin(angle / 2))
    fill = replace(kwargs.pop("fill", Fill()))
    fill.color = stroke.color
    return Marker(
        [_arrow_triangle_base(size, angle, fill=fill, stroke=stroke, **kwargs)],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )


def arrow_stealth(size: float = 5.0, angle: float = math.degrees(30), **kwargs):
    stroke = replace(kwargs.pop("stroke", Stroke(width=1)))
    fill = replace(kwargs.pop("fill", Fill()))
    stroke.line_join = LineJoin.miter
    # https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-miterlimit
    stroke.miter_limit = math.ceil(1.0 / math.sin(angle / 4))
    miter_length = 0.1 * stroke.width / math.sin(angle / 4)
    fill.color = stroke.color
    triangle = _arrow_triangle_base(size - miter_length, angle, **kwargs)
    return Marker(
        [Polygon(
            [
                triangle.vertices[0],
                triangle.vertices[1],
                Point(-size, 0),
                triangle.vertices[2]
            ],
            stroke=stroke,
            fill=fill,
            **kwargs
        )],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )


def arrow_line(size: float = 5.0, angle: float = math.degrees(30), **kwargs):
    triangle = _arrow_triangle_base(size, angle, **kwargs)
    stroke = replace(kwargs.pop("stroke", Stroke()))
    stroke.line_join = LineJoin.miter
    stroke.miter_limit = math.ceil(1.0 / math.sin(angle / 2))
    return Marker(
        [Polyline(
            [
                triangle.vertices[1],
                triangle.vertices[0],
                triangle.vertices[2]
            ],
            stroke=stroke,
            **kwargs
        )],
        viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
    )
