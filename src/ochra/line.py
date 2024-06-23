from typing import Union, Tuple
import math

import numpy as np

from ochra.parameterizable import Parameterizable1
from ochra.plane import Point, Vector, Transformation, PointI
from ochra.util.functions import logit
from ochra.style.stroke import Stroke
from ochra.util.property_utils import classproperty


class Line(Parameterizable1):

    def __init__(self,
                 coef: np.ndarray | Tuple[float, float, float],
                 stroke: Stroke = Stroke(),
                 **kwargs
                 ):
        if isinstance(coef, tuple):
            coef = np.array(coef)
        self.proj_coef = coef  # [a, b, c], projective coefficients
        assert self.proj_coef.shape == (3,)
        self.stroke = stroke

    @property
    def _a(self) -> float:
        return self.proj_coef[0].item()

    @property
    def _b(self) -> float:
        return self.proj_coef[1].item()

    @property
    def _c(self) -> float:
        return self.proj_coef[2].item()

    @property
    def normal_vector(self):
        return Vector(self._a, self._b)

    @property
    def angle(self) -> float:
        return -math.atan2(self._a, self._b)

    @property
    def slope(self) -> float:
        return -self._a / self._b

    @property
    def y_intercept(self) -> float:
        return +self._c / self._b

    def x_intercept(self) -> float:
        return +self._c / self._a

    def at(self, t: float):
        s = logit(t) * 1000  # maps (0, 1) to (-∞, +∞)
        h = math.hypot(self._a, self._b)
        p = -self._c / h
        θ = math.atan2(self._b, self._a)
        x = p * math.cos(θ) - s * math.sin(θ)
        y = p * math.sin(θ) + s * math.cos(θ)
        return Point(x, y)

    def transform(self, f: Transformation) -> 'Line':
        v = f.inverse().matrix.T.dot(self.proj_coef)
        return Line(v, stroke=self.stroke)

    @classproperty
    def y_axis(cls):
        return HorizontalLine(0)

    @classproperty
    def x_axis(cls):
        return VerticalLine(0)

    @classmethod
    def from_two_points(cls, p0: PointI, p1: PointI, **kwargs):
        p0 = Point.mk(p0)
        p1 = Point.mk(p1)
        return cls(
            (
                p0.y - p1.y,
                p1.x - p0.x,
                p0.x * p1.y - p0.y * p1.x,
            ),
            **kwargs
        )

    @classmethod
    def from_intercepts(cls, x_intercept: float, y_intercept: float, **kwargs):
        return cls(
            (
                y_intercept,
                x_intercept,
                -x_intercept * y_intercept,
            ),
            **kwargs
        )

    @classmethod
    def from_slope_and_y_intercept(cls, slope: float, y_intercept: float, **kwargs):
        return cls(
            (
                slope,
                -1,
                y_intercept,
            ),
            **kwargs
        )


class HorizontalLine(Line):
    def __init__(self, y: float):
        super().__init__((0, 1, -y))

    @classproperty
    def x_axis(cls):
        return cls(0)


class VerticalLine(Line):
    def __init__(self, x: float):
        super().__init__((1, 0, -x))

    @classproperty
    def y_axis(cls):
        return cls(0)

