from ochra import Element
from ochra.parameterizable import Parameterizable0
from ochra.plane import Point, Transformation, PointI
from ochra.style.stroke import Stroke


class Dot(Element, Parameterizable0):

    def __init__(self, point: PointI, stroke: Stroke = Stroke()):
        self.point = Point.mk(point)
        self.stroke = stroke

    def at(self):
        return self.point

    def transform(self, f: Transformation) -> 'Element':
        return Dot(f(self.point), stroke=self.stroke)
