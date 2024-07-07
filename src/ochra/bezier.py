from ochra.element import Element
from ochra.parametric import Parametric
from ochra.plane import Transformation, Point, PointI
from ochra.style.stroke import Stroke
from ochra.util.functions import lerp_point


class QuadraticBezierCurve(Parametric):

    def __init__(self, p0: PointI, p1: PointI, p2: PointI, stroke: Stroke = Stroke()):
        self.p0 = Point.mk(p0)
        self.p1 = Point.mk(p1)
        self.p2 = Point.mk(p2)
        self.stroke = stroke

    def at(self, t: float):
        f = lambda x, y: lerp_point(x, y, t)
        return f(
            f(self.p0, self.p1),
            f(self.p1, self.p2)
        )

    def transform(self, f: Transformation) -> 'QuadraticBezierCurve':
        return QuadraticBezierCurve(f(self.p0), f(self.p1), f(self.p2), stroke=self.stroke)


class CubicBezierCurve(Parametric):

    def __init__(self, p0: PointI, p1: PointI, p2: PointI, p3: PointI, stroke: Stroke = Stroke()):
        self.p0 = Point.mk(p0)
        self.p1 = Point.mk(p1)
        self.p2 = Point.mk(p2)
        self.p3 = Point.mk(p3)
        self.stroke = stroke

    def at(self, t: float):
        f = lambda x, y: lerp_point(x, y, t)
        return f(
            f(
                f(self.p0, self.p1),
                f(self.p1, self.p2)
            ),
            f(
                f(self.p1, self.p2),
                f(self.p2, self.p3)
            )
        )

    def transform(self, f: Transformation) -> 'CubicBezierCurve':
        return CubicBezierCurve(f(self.p0), f(self.p1), f(self.p2), f(self.p3), stroke=self.stroke)
