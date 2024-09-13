from dataclasses import dataclass
from enum import Enum
from typing import Collection, Dict, Optional, TYPE_CHECKING

from ochra.element import Element

if TYPE_CHECKING:
    from ochra.canvas import Canvas
    from ochra.rect import AxisAlignedRectangle


class MarkerOrientation(Enum):
    auto = "auto"
    auto_start_reverse = "auto-start-reverse"


class MarkerUnits(Enum):
    user_space_on_use = "userSpaceOnUse"
    stroke_width = "strokeWidth"


class Marker:

    all_named_markers: Dict[str, 'Marker'] = {}
    all_named_symbols: Dict[str, 'Marker'] = {}

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
        from ochra.canvas import Canvas
        return Canvas(self.elements, self.viewport)

    @classmethod
    def circle(cls, size: float = 2.0, **kwargs):
        from ochra.conic import Circle
        from ochra.rect import AxisAlignedRectangle
        return cls(
            [Circle(radius=size, **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def x_mark(cls, size: float = 2, **kwargs):
        from ochra.rect import AxisAlignedRectangle
        from ochra.segment import LineSegment
        return cls(
            [
                LineSegment((-size, -size), (size, size), **kwargs),
                LineSegment((-size, size), (size, -size), **kwargs),
            ],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def plus_mark(cls, size: float = 2.5, **kwargs):
        from ochra.rect import AxisAlignedRectangle
        from ochra.segment import LineSegment
        return cls(
            [
                LineSegment((-size, 0), (size, 0), **kwargs),
                LineSegment((0, -size), (0, size), **kwargs),
            ],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)),
        )

    @classmethod
    def polygon(cls, n: int, size: float = 2.0, angle: float = 0, **kwargs):
        from ochra.poly import Polygon
        from ochra.rect import AxisAlignedRectangle
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

