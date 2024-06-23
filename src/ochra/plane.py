import math
from dataclasses import dataclass
import numpy as np
from typing import Tuple, TypeAlias

from ochra.util.property_utils import classproperty


@dataclass
class Vector:
    """
    Represents a 2D vector.
    Vectors are elements in the 2D vector space.
    """
    x: float
    y: float

    def __add__(self, v: 'Vector') -> 'Vector':
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v: 'Vector') -> 'Vector':
        return Vector(self.x - v.x, self.y - v.y)

    def __mul__(self, s: float) -> 'Vector':
        return Vector(self.x * s, self.y * s)

    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y)

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass
class Point:
    """
    Represents a 2D point.
    Points are elements in the 2D affine space.
    """
    x: float
    y: float

    def __add__(self, v: Vector):
        return Point(self.x + v.x, self.y + v.y)

    def __sub__(self, p: 'Point | Vector') -> 'Vector | Point':
        if isinstance(p, Point):
            return Vector(self.x - p.x, self.y - p.y)
        return Point(self.x - p.x, self.y - p.y)

    def scale(self, sx: float, sy: float) -> 'Point':
        return Point(self.x * sx, self.y * sy)

    def as_vector(self) -> Vector:
        return Vector(self.x, self.y)

    @classproperty
    def origin(cls):
        return cls(0, 0)

    @classmethod
    def polar(cls, r: float, θ: float):
        return cls(r * math.cos(θ), r * math.sin(θ))

    @classmethod
    def mk(cls, p: 'PointI'):
        if isinstance(p, Point):
            return p
        return cls(p[0], p[1])


PointI: TypeAlias = Point | Tuple[float, float]


class Transformation:
    """
    Represents a 2D affine transformation.
    """
    def __init__(self, matrix: np.ndarray):
        """
        :param matrix: 3x3 matrix representing affine transformation.
        """
        assert matrix.shape == (3, 3)
        assert matrix[2, 0] == 0
        assert matrix[2, 1] == 0
        assert matrix[2, 2] == 1
        self.matrix = matrix

    def __matmul__(self, other):
        return Transformation(self.matrix @ other.matrix)

    def __call__(self, point: Point):
        x, y, w = self.matrix @ np.array([point.x, point.y, 1])
        return Point(x / w, y / w)

    def inverse(self):
        inverse = np.linalg.inv(self.matrix)
        return Transformation(inverse)

    @classproperty
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def translate(cls, d: Vector):
        return Transformation(np.array([
            [1, 0, d.x],
            [0, 1, d.y],
            [0, 0, 1]
        ]))

    @classmethod
    def rotate(cls, θ: float, center: Point = Point.origin):
        c, s = np.cos(θ), np.sin(θ)
        r = cls(np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ]))
        if center == Point.origin:
            return r
        else:
            return cls.translate(center.as_vector()) @ r @ cls.translate(-center.as_vector())

    @classmethod
    def scale(cls, s: Vector):
        return cls(np.array([
            [s.x, 0, 0],
            [0, s.y, 0],
            [0, 0, 1]
        ]))
