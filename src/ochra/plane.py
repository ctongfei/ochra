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

    @classmethod
    def mk(cls, v: 'VectorI'):
        if isinstance(v, Vector):
            return v
        else:
            return cls(v[0], v[1])


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

    def to_proj_point(self) -> 'ProjPoint':
        return ProjPoint(self.x, self.y, 1)

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
VectorI: TypeAlias = Vector | Tuple[float, float]
LineI: TypeAlias = Tuple[float, float, float]  # a x + b y + c = 0


@dataclass
class ProjPoint:
    x: float
    y: float
    z: float

    def to_point(self):
        return Point(self.x / self.z, self.y / self.z)


class Transformation:
    """
    Represents a 2D affine transformation.
    """
    def __new__(cls, matrix: np.ndarray):
        """
        :param matrix: 3x3 matrix representing affine transformation.
        """
        self = super().__new__(cls)
        assert matrix.shape == (3, 3)
        assert matrix[2, 0] == 0
        assert matrix[2, 1] == 0
        assert matrix[2, 2] == 1
        self.matrix = matrix
        return self

    def __matmul__(self, other):
        return Transformation(self.matrix @ other.matrix)

    def __call__(self, point: Point):
        x, y, w = self.matrix @ np.array([point.x, point.y, 1])
        # TODO: may result in an infinity-point on the affine plane
        return Point(x / w, y / w)

    def inverse(self):
        inverse = np.linalg.inv(self.matrix)
        return Transformation(inverse)

    def decompose(self) -> Tuple["Transformation", "Transformation", "Transformation"]:
        """
        Decomposes into translation, rotation, and scaling.
        """
        w = self.matrix[2, 2]
        m = self.matrix / w
        translation = np.eye(3)
        translation[:2, 2] = m[:2, 2]
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        sx = math.hypot(a, b)
        sy = math.hypot(c, d)
        rotation = np.eye(3)
        rotation[:2, 0] = m[:2, 0] / sx
        rotation[:2, 1] = m[:2, 1] / sy
        return Transformation(translation), Transformation(rotation), Transformation.scale((sx, sy))

    def is_identity(self):
        return np.allclose(self.matrix, np.eye(3))

    def __repr__(self):
        return f"Transformation({self.matrix})"

    @classproperty
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def translate(cls, d: VectorI):
        d = Vector.mk(d)
        return Transformation(np.array([
            [1, 0, d.x],
            [0, 1, d.y],
            [0, 0, 1]
        ]))

    @classmethod
    def rotate(cls, θ: float, center: Point = Point.origin):
        c, s = np.cos(θ), np.sin(θ)
        rot = cls(np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ]))
        if center == Point.origin:
            return rot
        else:
            v = center.as_vector()
            return cls.translate(v) @ rot @ cls.translate(-v)

    @classmethod
    def scale(cls, s: VectorI, center: Point = Point.origin):
        s = Vector.mk(s)
        sc = cls(np.array([
            [s.x, 0, 0],
            [0, s.y, 0],
            [0, 0, 1]
        ]))
        if center == Point.origin:
            return sc
        else:
            v = center.as_vector()
            return cls.translate(v) @ sc @ cls.translate(-v)

    @classmethod
    def reflect(cls, line: LineI):
        a, b, c = line
        d = math.hypot(a, b)
        a, b, c = a / d, b / d, c / d
        refl = cls(np.array([
            [1 - 2 * a ** 2, -2 * a * b, -2 * a * c],
            [-2 * a * b, 1 - 2 * b ** 2, -2 * b * c],
            [0, 0, 1]
        ]))
        return refl
