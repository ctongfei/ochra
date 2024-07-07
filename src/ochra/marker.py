from dataclasses import dataclass
from enum import Enum
from typing import Collection, Optional, Dict

from ochra.element import Element


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
        Marker.all_named_markers[self.name] = self

    @property
    def canvas(self) -> "Canvas":
        from ochra.canvas import Canvas
        return Canvas(self.elements, self.viewport)

    @classmethod
    def circle(cls, size: float = 4.0, **kwargs):
        from ochra.conic import Circle
        from ochra.rect import AxisAlignedRectangle
        return cls(
            [Circle(radius=size, **kwargs)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )

    @classmethod
    def polygon(cls, n: int, size: float = 5.0, angle: float = 0, **kwargs):
        from ochra.poly import Polygon
        from ochra.rect import AxisAlignedRectangle
        return cls(
            [Polygon.regular(n, circumradius=size, **kwargs).rotate(angle)],
            viewport=AxisAlignedRectangle((-size, -size), (size, size)).scale(2, 2),
        )


@dataclass
class MarkerConfig:
    start: Optional[Marker] = None
    mid: Optional[Marker] = None
    end: Optional[Marker] = None

