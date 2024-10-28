import math
from typing import Tuple, TypeAlias, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ochra.util import Global, classproperty
if TYPE_CHECKING:
    from ochra.core import Line


τ = 6.283185307179586
"""The true circle constant, the ratio of a circle's circumference to its radius."""

Scalar: TypeAlias = float | jax.Array
"""A scalar value, which can be a float or a Jax array of dimensionality 0."""


@jdc.pytree_dataclass
class Vector:
    """
    Represents a 2D vector.
    Vectors are elements in the 2D vector space.
    """
    vec: jax.Array

    def __post_init__(self):
        assert self.vec.shape == (2,)  # 2D vector

    @property
    def x(self) -> Scalar:
        return self.vec[0]

    @property
    def y(self) -> Scalar:
        return self.vec[1]

    def __add__(self, v: 'Vector') -> 'Vector':
        return Vector(self.vec + v.vec)

    def __sub__(self, v: 'Vector') -> 'Vector':
        return Vector(self.vec - v.vec)

    def __mul__(self, s: float) -> 'Vector':
        return Vector(self.vec * s)

    def __neg__(self) -> 'Vector':
        return Vector(-self.vec)

    def __abs__(self) -> Scalar:
        return jnp.linalg.norm(self.vec)

    def dot(self, v: 'Vector') -> Scalar:
        return jnp.dot(self.vec, v.vec)

    def norm(self) -> Scalar:
        return abs(self)

    def rotate(self, θ: Scalar) -> 'Vector':
        rot = jnp.array([
            [math.cos(θ), -math.sin(θ)],
            [math.sin(θ), math.cos(θ)]
        ])
        return Vector(jnp.dot(rot, self.vec))

    def normalize(self) -> 'Vector':
        return Vector(self.vec / abs(self))

    def to_point(self) -> 'Point':
        return Point(self.vec)

    @classmethod
    def mk(cls, x: 'VectorI | float', y: float = None):
        if (isinstance(x, float) or isinstance(x, jax.Array)) and y is not None:
            return cls(jnp.array([x, y]))
        assert y is None
        if isinstance(x, Vector):
            return x
        else:
            return cls(jnp.array([float(x[0]), float(x[1])]))

    @classmethod
    def unit(cls, θ: Scalar):
        return cls.mk((jnp.cos(θ), jnp.sin(θ)))


@jdc.pytree_dataclass
class Point:
    """
    Represents a 2D point.
    Points are elements in the 2D affine space.
    """
    loc: jax.Array

    @property
    def x(self) -> Scalar:
        return self.loc[0]

    @property
    def y(self) -> Scalar:
        return self.loc[1]

#   def __post_init__(self):
#        assert self.loc.shape == (2,)

    def __add__(self, v: Vector):
        return Point(self.loc + v.vec)

    def __sub__(self, p: 'Point | Vector') -> 'Vector | Point':
        if isinstance(p, Point):
            return Vector(self.loc - p.loc)
        return Point(self.loc - p.vec)

    def __eq__(self, other):
        return jnp.allclose(self.loc, other.loc, atol=Global.approx_eps).item()

    def scale(self, sx: float, sy: float) -> 'Point':
        return Point(self.loc * jnp.array([sx, sy]))

    def translate(self, dx: float, dy: float) -> 'Point':
        return self + Vector.mk((dx, dy))

    def to_vector(self) -> Vector:
        return Vector(self.loc)

    def to_proj_point(self) -> 'ProjPoint':
        return ProjPoint(jnp.concat([self.loc, jnp.array([1])]))

    def __str__(self):
        return f"({self.x.item()}, {self.y.item()})"

    def __repr__(self):
        return self.__str__()

    @classproperty
    def origin(cls):
        return cls(jnp.array([0.0, 0.0]))

    @classmethod
    def polar(cls, r: Scalar, θ: Scalar):
        return cls(jnp.array([r * jnp.cos(θ), r * jnp.sin(θ)]))

    @classmethod
    def mk(cls, x: 'PointI | float', y: float = None):
        if (isinstance(x, float) or (isinstance(x, jax.Array) and x.ndim == 0)) and y is not None:
            return cls(jnp.array([x, y]))
        assert y is None
        if isinstance(x, Point):
            return x
        return cls(jnp.array([float(x[0]), float(x[1])]))


PointI: TypeAlias = Point | tuple[float, float] | np.ndarray | jax.Array
VectorI: TypeAlias = Vector | tuple[float, float] | np.ndarray | jax.Array
LineI: TypeAlias = tuple[float, float, float] | np.ndarray | jax.Array  # a x + b y + c = 0
ConicI: TypeAlias = tuple[float, float, float, float, float, float] | np.ndarray | jax.Array  # a x^2 + b x y + c y^2 + d x + e y + f = 0


@jdc.pytree_dataclass
class ProjPoint:
    """A point in the projective plane RP^2."""
    loc: jax.Array  # [3]

    def __post_init__(self):
        assert self.loc.shape == (3,)
        # TODO: assert not all zero
        assert not jnp.allclose(self.loc, 0, atol=Global.approx_eps)

    @property
    def x(self) -> Scalar:
        return self.loc[0]

    @property
    def y(self) -> Scalar:
        return self.loc[1]

    @property
    def z(self) -> Scalar:
        return self.loc[2]

    def to_point(self):
        assert self.z != 0
        return Point(self.loc[:2] / self.z)

    def is_infinity_point(self):
        return self.z == 0

    def __str__(self):
        return f"({self.x} : {self.y} : {self.z})"

    def __eq__(self, other):
        for i in range(3):
            if other.loc[i] != 0:
                t = self.loc[i] / other.loc[i]
                if jnp.allclose(self.loc, t * other.loc, atol=Global.approx_eps):
                    return True
        return False


class Transformation:
    """
    Represents a 2D affine transformation.
    """
    def __init__(self, matrix: jax.Array):
        """
        :param matrix: 3x3 matrix representing affine transformation.
        """
        assert matrix.shape == (3, 3)
        self.matrix = matrix

    def __matmul__(self, other):
        if isinstance(other, CompositeTransformation):
            return CompositeTransformation([self] + other.transformations)
        else:
            return CompositeTransformation([self, other])

    def __call__(self, point: Point):
        pp = self.matrix @ jnp.concat([point.loc, jnp.array([1])])
        # TODO: may result in an infinity-point on the affine plane
        return Point(pp[:2] / pp[2])

    def apply_batch(self, points: jax.Array) -> jax.Array:
        proj_points = jnp.concat([points, jnp.ones((points.shape[0], 1))], axis=1)
        transformed = jnp.dot(self.matrix, proj_points.T).T
        return transformed[:, :2] / transformed[:, 2][:, None]

    def inverse(self):
        inverse = jnp.linalg.inv(self.matrix)
        return Transformation(inverse)

    def angle_if_rotation(self) -> float:
        a, b, c, d = self.matrix[:2, :2].flatten()
        return math.atan2(b, a)

    def scale_if_scaling(self) -> Tuple[float, float]:
        a, b, c, d = self.matrix[:2, :2].flatten()
        return math.hypot(a, b), math.hypot(c, d)

    def vector_if_translation(self) -> Vector:
        return Vector(self.matrix[:2, 2])

    def decompose(self) -> Tuple["Transformation", "Transformation", "Transformation"]:
        """
        Decomposes into translation, rotation, and scaling.
        """
        w = self.matrix[2, 2]
        m = self.matrix / w
        translation = jnp.eye(3).at[:2, 2].set(m[:2, 2])
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        sx = math.hypot(a, b)
        sy = math.hypot(c, d)
        rotation = jnp.eye(3).at[:2, 0].set(m[:2, 0] / sx).at[:2, 1].set(m[:2, 1] / sy)
        return Transformation(translation), Transformation(
            rotation), Scaling((sx, sy))

    def is_identity(self):
        return jnp.allclose(self.matrix, jnp.eye(3))

    def __repr__(self):
        return f"Transformation({self.matrix})"

    @classproperty
    def identity(cls):
        return cls(jnp.eye(3))


class CompositeTransformation(Transformation):
    def __init__(self, transformations: list[Transformation]):
        matrix = jnp.eye(3)
        for t in transformations:
            matrix = matrix @ t.matrix
        super().__init__(matrix)
        self.transformations = transformations

    def __matmul__(self, other):
        if isinstance(other, CompositeTransformation):
            return CompositeTransformation(self.transformations + other.transformations)
        else:
            return CompositeTransformation(self.transformations + [other])


class Translation(Transformation):
    def __init__(self, d: VectorI):
        d = Vector.mk(d)
        super().__init__(jnp.array([
            [1, 0, d.x],
            [0, 1, d.y],
            [0, 0, 1]
        ]))
        self.vec = d

    def inverse(self):
        return Translation(-self.vec)

    def __matmul__(self, other):
        if isinstance(other, Translation):
            return Translation(self.vec + other.vec)
        return super().__matmul__(other)


class Rotation(Transformation):
    def __init__(self, θ: Scalar):
        super().__init__(jnp.array([
            [math.cos(θ), -math.sin(θ), 0],
            [math.sin(θ), math.cos(θ), 0],
            [0, 0, 1]
        ]))
        self.angle = θ

    def __matmul__(self, other):
        if isinstance(other, Rotation):
            return Rotation(self.angle + other.angle)
        return super().__matmul__(other)

    def inverse(self):
        return Rotation(-self.angle)

    @classmethod
    def centered(cls, θ: Scalar, center: PointI):
        if center == Point.origin:
            return cls(θ)
        else:
            v = center.to_vector()
            return CompositeTransformation([Translation(v), cls(θ), Translation(-v)])


class Scaling(Transformation):
    def __init__(self, s: VectorI):
        s = Vector.mk(s)
        super().__init__(jnp.diag(jnp.concat([s.vec, jnp.array([1])])))
        self.scale = s

    def __matmul__(self, other):
        if isinstance(other, Scaling):
            return Scaling(Vector(self.scale.vec * other.scale.vec))
        return super().__matmul__(other)

    @classmethod
    def centered(cls, s: VectorI, center: Point = Point.origin):
        s = Vector.mk(s)
        sc = cls(s)
        if center == Point.origin:
            return sc
        else:
            v = center.to_vector()
            return Translation(v) @ sc @ Translation(-v)


class Reflection(Transformation):
    def __init__(self, line: 'LineI | Line'):
        if isinstance(line, Line):
            a, b, c = line.coef
        else:
            a, b, c = line
        d = math.hypot(a, b)
        a, b, c = a / d, b / d, c / d
        super().__init__(jnp.array([
            [1 - 2 * a ** 2, -2 * a * b, -2 * a * c],
            [-2 * a * b, 1 - 2 * b ** 2, -2 * b * c],
            [0, 0, 1]
        ]))
        self.line = line
