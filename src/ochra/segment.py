import math

from ochra.marker import MarkerConfig
from ochra.element import Element
from ochra.line import Line
from ochra.parameterizable import Parameterizable1
from ochra.plane import Point, Transformation, Vector, PointI
from ochra.util.functions import lerp_point, dist
from ochra.style.stroke import Stroke


class LineSegment(Parameterizable1):

    def __init__(self,
                 p0: PointI,
                 p1: PointI,
                 stroke: Stroke = Stroke(),
                 markers: MarkerConfig = MarkerConfig()
                 ):
        self.p0 = Point.mk(p0)
        self.p1 = Point.mk(p1)
        self.stroke = stroke
        self.markers = markers

    @property
    def midpoint(self):
        return self.at(0.5)

    @property
    def center(self):
        return self.midpoint

    @property
    def slope(self):
        return (self.p1.y - self.p0.y) / (self.p1.x - self.p0.x)

    @property
    def angle(self):
        return math.atan2(self.p1.y - self.p0.y, self.p1.x - self.p0.x)

    @property
    def length(self):
        return dist(self.p0, self.p1)

    def extend_as_line(self) -> Line:
        return Line.from_two_points(self.p0, self.p1)

    def as_vector(self) -> Vector:
        return Vector(self.p1.x - self.p0.x, self.p1.y - self.p0.y)

    def at(self, t: float):
        return lerp_point(self.p0, self.p1, t)

    def transform(self, f: Transformation) -> 'LineSegment':
        return LineSegment(f(self.p0), f(self.p1), stroke=self.stroke, markers=self.markers)
