from ochra import Element
from ochra.marker import Marker
from ochra.plane import Point, Transformation, PointI


class Mark(Element):

    def __init__(self, point: PointI, marker: Marker):
        self.point = Point.mk(point)
        self.marker = marker
        Marker.all_named_symbols[marker.name] = marker

    def transform(self, f: Transformation) -> 'Element':
        return Mark(f(self.point), marker=self.marker)
