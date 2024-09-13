from typing import Optional, TYPE_CHECKING
import math

import numpy as np

from ochra.parametric import Parametric
from ochra.plane import LineI, Point, PointI, Transformation, Vector
from ochra.style.stroke import Stroke
from ochra.util.functions import logit
from ochra.util.property_utils import classproperty

if TYPE_CHECKING:
    from ochra.rect import AxisAlignedRectangle


class Line(Parametric):

    def __new__(cls,
                coef: np.ndarray | LineI,
                stroke: Stroke = Stroke(),
                **kwargs
                ):
        self = super().__new__(cls)
        if isinstance(coef, tuple):
            coef = np.array(coef)
        self.coef = coef  # [a, b, c], projective coefficients
        assert self.coef.shape == (3,)
        self.stroke = stroke
        return self

    @property
    def _a(self) -> float:
        return self.coef[0].item()

    @property
    def _b(self) -> float:
        return self.coef[1].item()

    @property
    def _c(self) -> float:
        return self.coef[2].item()

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

    def axis_aligned_bbox(self) -> 'Optional[AxisAlignedRectangle]':
        from ochra.exception import BoundingBoxIndeterminateException
        raise BoundingBoxIndeterminateException(self)

    def at(self, t: float):
        s = logit(t) * 1000  # maps (0, 1) to (-∞, +∞)
        h = math.hypot(self._a, self._b)
        p = -self._c / h
        θ = math.atan2(self._b, self._a)
        x = p * math.cos(θ) - s * math.sin(θ)
        y = p * math.sin(θ) + s * math.cos(θ)
        return Point(x, y)

    def transform(self, f: Transformation) -> 'Line':
        v = f.inverse().matrix.T.dot(self.coef)
        return Line(v, stroke=self.stroke)

    def closest_to(self, p: PointI) -> Point:
        """
        Returns the point on the line that is closest to the given point.
        """
        a, b, c = self.coef
        p = Point.mk(p)
        x = (b * (b * p.x - a * p.y) - a * c) / (a ** 2 + b ** 2)
        y = (a * (-b * p.x + a * p.y) - b * c) / (a ** 2 + b ** 2)
        return Point(x, y)

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

