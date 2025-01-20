import math
from collections.abc import Collection
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from ochra.core import Group, Element, Canvas, Circle, LineSegment, AxisAlignedRectangle, Polygon
from ochra.style import Fill, Color
from ochra.geometry import Point, PointI, AffineTransformation, Vector, τ


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
    def bullet(cls, size: float = 2, **kwargs):
        if "fill" not in kwargs:
            if "stroke" in kwargs:
                kwargs['fill'] = Fill(color=kwargs['stroke'].color)
            else:
                kwargs['fill'] = Fill(color=Color(0, 0, 0))
        return cls(
            [Circle(radius=size / 2, **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
        )

    @classmethod
    def circle(cls, size: float = 3, **kwargs):
        return cls(
            [Circle(radius=size / 2, **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
        )

    @classmethod
    def asterisk(cls, n: int, size: float = 4, angle: float = τ / 4, **kwargs):
        return cls(
            [
                LineSegment((0, 0), (Vector.unit(angle + τ * i / n) * (size / 2)).to_point(), **kwargs)
                for i in range(n)
            ],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
        )

    @classmethod
    def tick(cls, size: float = 4, angle: float = 0.0, **kwargs):
        return Marker.asterisk(1, size, angle, **kwargs)

    @classmethod
    def x_mark(cls, size: float = 4, **kwargs):
        return Marker.asterisk(4, size, angle=τ / 8, **kwargs)

    @classmethod
    def plus_mark(cls, size: float = 4, **kwargs):
        return Marker.asterisk(4, size, angle=0, **kwargs)

    @classmethod
    def polygon(cls, n: int, size: float = 4, angle: float = τ / 4, **kwargs):
        width = kwargs.get("stroke", {}).get("width", 1.0)
        θ = τ / 2 - τ / n
        size -= 0.5 * width / math.sin(θ / 2)
        return cls(
            [Polygon.regular(n, circumradius=size / 2, **kwargs).rotate(angle)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
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


class Mark(Element):  # TODO: should probably be Annotation

    def __init__(self, point: PointI, marker: Marker):
        self.point = Point.mk(point)
        self.marker = marker
        Marker.register_as_symbol(marker)

    def aabb(self) -> 'Optional[AxisAlignedRectangle]':
        return Group(self.marker.elements).aabb().translate(self.point.x, self.point.y)

    def transform(self, f: AffineTransformation) -> 'Element':
        # Only transforms the location of the marker, not the marker itself
        return Mark(f(self.point), marker=self.marker)
