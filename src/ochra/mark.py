from ochra import Element
from ochra.marker import Marker
from ochra.plane import Point, Transformation, PointI


class Mark(Element):

    def __init__(self, point: PointI, marker: Marker):
        self.point = Point.mk(point)
        self.marker = marker
        Marker.register_as_symbol(marker)

    def transform(self, f: Transformation) -> 'Element':
        # Only transforms the location of the marker, not the marker itself
        return Mark(f(self.point), marker=self.marker)
