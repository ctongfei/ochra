import math
from typing import Tuple
import numpy as np

from ochra.element import Element
from ochra.parameterizable import Parameterizable1
from ochra.plane import Point, Transformation, PointI
from ochra.style import Fill
from ochra.style.stroke import Stroke
from ochra.util.functions import lerp_point, dist, lerp


class Conic(Parameterizable1):
    """
    Represents any conic section in the plane.
    """

    def __init__(self, coef: np.ndarray | Tuple[float, float, float, float, float, float], stroke: Stroke = Stroke()):
        if isinstance(coef, tuple):
            a, b, c, d, e, f = coef
            coef = np.array([
                [a, b/2, d/2],
                [b/2, c, e/2],
                [d/2, e/2, f],
            ])
        self.proj_coef = coef
        assert self.proj_coef.shape == (3, 3)
        self.stroke = stroke

    def materialize(self) -> Element:
        d3 = np.linalg.det(self.proj_coef)
        d2 = np.linalg.det(self.proj_coef[:2, :2])
        if d3 == 0.0:  # degenerate
            if d2 < 0.0:
                # two intersecting lines, degenerate hyperbola
                pass
            elif d2 == 0:
                # two parallel lines, degenerate parabola
                pass
            else:
                # a dot, degenerate ellipse
                pass
        else:  # non-degenerate
            if d2 < 0.0:
                # hyperbola
                pass
            elif d2 == 0:
                # parabola
                pass
            else:
                # ellipse
                pass


class Ellipse(Conic, Parameterizable1):

    def __init__(self, focus0: PointI, focus1: PointI, major_axis: float, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        focus0 = Point.mk(focus0)
        focus1 = Point.mk(focus1)

        from math import sin, cos, atan2
        center = lerp_point(focus0, focus1, 0.5)
        a = major_axis / 2
        b = dist(focus0, focus1) / 2
        θ = atan2(focus1.y - focus0.y, focus1.x - focus0.x)
        A = a * a * sin(θ) * sin(θ) + b * b * cos(θ) * cos(θ)
        B = 2 * (b * b - a * a) * sin(θ) * cos(θ)
        C = a * a * cos(θ) * cos(θ) + b * b * sin(θ) * sin(θ)
        D = -2 * A * center.x - B * center.y
        E = -B * center.x - 2 * C * center.y
        F = A * center.x * center.x + B * center.x * center.y + C * center.y * center.y - a * a * b * b
        super().__init__((A, B, C, D, E, F), stroke=stroke)
        self.focus0 = focus0
        self.focus1 = focus1
        self.center = center
        self.major_axis = major_axis
        self.fill = fill

    @property
    def c(self):
        return dist(self.focus0, self.center)

    @property
    def semi_major_axis(self):
        return self.major_axis / 2

    @property
    def a(self):
        return self.semi_major_axis

    @property
    def semi_minor_axis(self):
        return math.sqrt(self.semi_major_axis ** 2 - self.c ** 2)

    @property
    def b(self):
        return self.semi_minor_axis

    @property
    def eccentricity(self):
        return self.c / self.a

    @property
    def major_axis_angle(self):
        return math.atan2(self.focus1.y - self.focus0.y, self.focus1.x - self.focus0.x)

    def arc_between(self, start: float, end: float):
        return Arc(self, start, end)

    def at(self, t: float):
        θ = t * math.tau
        φ = self.major_axis_angle
        x = self.center.x + self.a * math.cos(θ) * math.cos(φ) - self.b * math.sin(θ) * math.sin(φ)
        y = self.center.y + self.b * math.sin(θ) * math.cos(φ) + self.a * math.cos(θ) * math.sin(φ)
        return Point(x, y)

    @classmethod
    def standard(cls, a: float, b: float, **kwargs):
        if a >= b:
            c = math.sqrt(a ** 2 - b ** 2)
            return cls(Point(-c, 0), Point(c, 0), 2 * a, **kwargs)
        else:
            c = math.sqrt(b ** 2 - a ** 2)
            return cls(Point(0, -c), Point(0, c), 2 * b, **kwargs)


class Circle(Ellipse, Parameterizable1):

    def __init__(self, radius: float, center: PointI = (0, 0), stroke: Stroke = Stroke(), fill: Fill = Fill()):
        center = Point.mk(center)
        super().__init__(center, center, 2 * radius, stroke=stroke, fill=fill)
        self.center = center
        self.radius = radius

    def at(self, t: float):
        x, y = self.center
        θ = t * math.tau  # [0, 1] -> [0, τ]
        return x + self.radius * math.cos(θ), y + self.radius * math.sin(θ)

    def transform(self, f: Transformation) -> 'Circle':
        # TODO: wrong! transforms into an ellipse
        return Circle(self.radius, center=f(self.center), stroke=self.stroke, fill=self.fill)

    @classmethod
    def from_center_and_radius(cls, center: PointI, radius: float):
        center = Point.mk(center)
        return cls(center, radius)


class Arc(Parameterizable1):
    def __init__(self, ellipse: Ellipse, start: float, end: float, stroke: Stroke = Stroke()):
        assert 0 <= start < end <= 1
        self.ellipse = ellipse
        self.start = start
        self.end = end
        self.stroke = stroke

    def at(self, t: float):
        return self.ellipse.at(lerp(self.start, self.end, t))
