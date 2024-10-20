from collections.abc import Collection
from enum import Enum
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

from ochra.core import Group, Element, Canvas, Circle, LineSegment, AxisAlignedRectangle, Polygon
from ochra.style import Fill, Color
from ochra.geometry import Point, PointI, Transformation, Vector


class MarkerOrientation(Enum):
    auto = "auto"
    auto_start_reverse = "auto-start-reverse"


class MarkerUnits(Enum):
    user_space_on_use = "userSpaceOnUse"
    stroke_width = "strokeWidth"


class Marker:

    all_named_markers: dict[str, 'Marker'] = {}
    all_named_symbols: dict[str, 'Marker'] = {}

    def __init__(self,
                 elements: Collection[Element],
                 viewport: "AxisAlignedRectangle",
                 units: MarkerUnits = MarkerUnits.user_space_on_use,
                 orientation: MarkerOrientation | float = MarkerOrientation.auto,
                 name: Optional[str] = None
                 ):
        self.elements = elements
        self.viewport = viewport
        self.units = units
        self.orientation = orientation
        self.name = name if name is not None else hex(id(self))[2:]

    @property
    def canvas(self) -> "Canvas":
        return Canvas(self.elements, self.viewport)

    @classmethod
    def bullet(cls, size: float = 0.75, **kwargs):
        if "fill" not in kwargs:
            if "stroke" in kwargs:
                kwargs['fill'] = Fill(color=kwargs['stroke'].color)
            else:
                kwargs['fill'] = Fill(color=Color(0, 0, 0))
        return cls(
            [Circle(radius=size, **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def circle(cls, size: float = 2.0, **kwargs):
        return cls(
            [Circle(radius=size, **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def tick(cls, size: float = 2, angle: float = 0.0, **kwargs):
        return cls(
            [LineSegment((0, 0), (Vector.unit(angle) * size).to_point(), **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
        )

    @classmethod
    def x_mark(cls, size: float = 2, **kwargs):
        return cls(
            [
                LineSegment((-size, -size), (size, size), **kwargs),
                LineSegment((-size, size), (size, -size), **kwargs),
            ],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def plus_mark(cls, size: float = 2.5, **kwargs):
        return cls(
            [
                LineSegment((-size, 0), (size, 0), **kwargs),
                LineSegment((0, -size), (0, size), **kwargs),
            ],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
        )

    @classmethod
    def polygon(cls, n: int, size: float = 2.0, angle: float = 0, **kwargs):
        return cls(
            [Polygon.regular(n, circumradius=size, **kwargs).rotate(angle)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def register_as_symbol(cls, m: 'Marker'):
        Marker.all_named_symbols[f"symbol-{m.name}"] = m

    @classmethod
    def register_as_marker(cls, m: 'Marker'):
        Marker.all_named_markers[m.name] = m


@dataclass
class MarkerConfig:
    start: Optional[Marker] = None
    mid: Optional[Marker] = None
    end: Optional[Marker] = None



class Mark(Element):  # should probably be Annotation

    def __init__(self, point: PointI, marker: Marker):
        self.point = Point.mk(point)
        self.marker = marker
        Marker.register_as_symbol(marker)

    def aabb(self) -> 'Optional[AxisAlignedRectangle]':
        return Group(self.marker.elements).aabb().translate(self.point.x, self.point.y)

    def transform(self, f: Transformation) -> 'Element':
        # Only transforms the location of the marker, not the marker itself
        return Mark(f(self.point), marker=self.marker)


