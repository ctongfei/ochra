import math
from gettext import npgettext
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ochra.util import Global, classproperty
if TYPE_CHECKING:
    from ochra.core import Line


τ = 6.283185307179586
"""The true circle constant, the ratio of a circle's circumference to its radius."""

type Scalar = float | jax.Array
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

    def __mul__(self, s: Scalar) -> 'Vector':
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
    def mk(cls, x: 'VectorI | Scalar', y: Scalar | None = None):
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

    def __sub__(self, p: 'Point') -> Vector:
        return Vector(self.loc - p.loc)

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
        return f"Point({self.x.item()}, {self.y.item()})"

    def __repr__(self):
        return self.__str__()

    @classproperty
    def origin(cls):
        return cls(jnp.array([0.0, 0.0]))

    @classmethod
    def polar(cls, r: Scalar, θ: Scalar):
        return cls(jnp.array([r * jnp.cos(θ), r * jnp.sin(θ)]))

    @classmethod
    def mk(cls, x: 'PointI | Scalar', y: Scalar | None = None):
        if (isinstance(x, float) or (isinstance(x, jax.Array) and x.ndim == 0)) and y is not None:
            return cls(jnp.array([x, y]))
        assert y is None
        if isinstance(x, Point):
            return x
        return cls(jnp.array([float(x[0]), float(x[1])]))


type PointI = Point | tuple[float, float] | jax.Array
type VectorI = Vector | tuple[float, float] | jax.Array
type LineI = tuple[float, float, float] | jax.Array  # a x + b y + c = 0
type ConicI = tuple[float, float, float, float, float, float] | jax.Array  # a x^2 + b x y + c y^2 + d x + e y + f = 0


@jdc.pytree_dataclass
class ProjPoint:
    """A point in the projective plane RP^2."""
    loc: jax.Array  # [3]

    def __post_init__(self):
        assert self.loc.shape == (3,)
        # TODO: assert not all zero

    @property
    def x(self) -> Scalar:
        return self.loc[0]

    @property
    def y(self) -> Scalar:
        return self.loc[1]

    @property
    def z(self) -> Scalar:
        return self.loc[2]

    def to_point(self) -> Point | None:  # None if infinity point
        if self.z == 0:
            return None
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


# TODO: create hierarchy of transformations
# Euclidean < Affine < Projective
class Transformation:
    """
    Represents a 2D projective transformation.
    """
    def __init__(self, matrix: jax.Array):
        """
        :param matrix: 3x3 matrix representing any projective transformation.
        """
        assert matrix.shape == (3, 3)
        z = matrix[2, 2]
        assert z != 0.0
        assert jnp.linalg.det(matrix) != 0.0  # nonsingular
        self.matrix = matrix / z

    def __matmul__(self, other) -> 'Transformation':
        if isinstance(other, CompositeTransformation):
            return CompositeTransformation([self] + other.transformations)
        else:
            return CompositeTransformation([self, other])

    def __call__(self, point: Point) -> Point | None:
        return self.apply_proj(point.to_proj_point()).to_point()

    def apply_proj(self, pp: ProjPoint) -> ProjPoint:
        return ProjPoint(self.matrix @ pp.loc)

    def apply_batch(self, points: jax.Array) -> jax.Array:
        proj_points = jnp.concat([points, jnp.ones((points.shape[0], 1))], axis=1)
        transformed = jnp.dot(self.matrix, proj_points.T).T
        return transformed[:, :2] / transformed[:, 2][:, None]

    def inverse(self):
        inverse = jnp.linalg.inv(self.matrix)
        return Transformation(inverse)

    def decompose(self) -> tuple['Translation', 'Rotation', 'ShearX', 'Scaling', 'Elation']:
        """
        Decomposes an affine transformation into translation, rotation, and scaling.
        """
        w = self.matrix[2, 2]
        m = self.matrix / w
        tr = Translation(m[:2, 2])
        m0 = m[:2, :2]
        sx = jnp.linalg.norm(m0[:, 0])
        θ = jnp.atan2(m0[1, 0], m0[0, 0])
        msy = jnp.cos(θ) * m0[0, 1] + jnp.sin(θ) * m0[1, 1]
        if not jnp.isclose(jnp.sin(θ), 0.0, atol=Global.approx_eps):
            sy = (msy * jnp.cos(θ) - m0[0, 1]) / jnp.sin(θ)
        else:
            sy = (m0[1, 1] - msy * jnp.sin(θ)) / jnp.cos(θ)
        m = msy / sy
        el = Elation(self.matrix[:2, 2])
        return tr, Rotation(θ), ShearX(m), Scaling((sx.item(), sy.item())), el

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
        center = Point.mk(center)
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


class ShearX(Transformation):
    def __init__(self, m: Scalar):
        super().__init__(jnp.array([
            [1, m, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]))
        self.factor = m
        self.angle = jnp.arctan(m)

    @classmethod
    def from_angle(cls, θ: Scalar):
        return cls(jnp.tan(θ))


class ShearY(Transformation):
    def __init__(self, m: Scalar):
        super().__init__(jnp.array([
            [1, 0, 0],
            [m, 1, 0],
            [0, 0, 1]
        ]))
        self.factor = m
        self.angle = jnp.arctan(m)

    @classmethod
    def from_angle(cls, θ: Scalar):
        return cls(jnp.tan(θ))


class Elation(Transformation):
    def __init__(self, vec: PointI):
        vec = Point.mk(vec)
        super().__init__(jnp.array([
            [1, 0, 0],
            [0, 1, 0],
            [vec.x, vec.y, 1]
        ]))
        self.vec = vec


class Reflection(Transformation):
    def __init__(self, line: 'LineI | Line'):
        from ochra.core import Line
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
