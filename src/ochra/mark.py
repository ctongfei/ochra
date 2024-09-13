from typing import Optional, TYPE_CHECKING

from ochra.element import Element
from ochra.marker import Marker
from ochra.plane import Point, PointI, Transformation

if TYPE_CHECKING:
    from ochra.rect import AxisAlignedRectangle


class Mark(Element):

    def __init__(self, point: PointI, marker: Marker):
        self.point = Point.mk(point)
        self.marker = marker
        Marker.register_as_symbol(marker)

    def axis_aligned_bbox(self) -> 'Optional[AxisAlignedRectangle]':
        return self.marker.viewport.translate(self.point.x, self.point.y)

    def transform(self, f: Transformation) -> 'Element':
        # Only transforms the location of the marker, not the marker itself
        return Mark(f(self.point), marker=self.marker)
