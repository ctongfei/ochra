import math
from typing import TYPE_CHECKING, overload

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

    @property
    def angle(self) -> Scalar:
        return jnp.arctan2(self.y, self.x)

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

    def normalize(self) -> 'Vector':
        return Vector(self.vec / abs(self))

    def rotate(self, θ: Scalar) -> 'Vector':
        rot = jnp.array([
            [math.cos(θ), -math.sin(θ)],
            [math.sin(θ), math.cos(θ)]
        ])
        return Vector(jnp.dot(rot, self.vec))

    def to_point(self) -> 'Point':
        return Point(self.vec)

    def __hash__(self):
        return hash((self.x.item(), self.y.item()))

    @classmethod
    def mk(cls, x: 'VectorI | Scalar', y: Scalar | None = None) -> 'Vector':
        if (isinstance(x, float) or isinstance(x, jax.Array)) and y is not None:
            return cls(jnp.array([x, y]))
        assert y is None
        if isinstance(x, Vector):
            return x
        else:
            assert isinstance(x, jax.Array | tuple)
            return cls(jnp.array([float(x[0]), float(x[1])]))

    @classmethod
    def unit(cls, θ: Scalar):
        return cls.mk(jnp.array([jnp.cos(θ), jnp.sin(θ)]))


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

    def __add__(self, v: Vector) -> 'Point':
        return Point(self.loc + v.vec)

    @overload
    def __sub__(self, p: 'Point') -> Vector: ...

    @overload
    def __sub__(self, p: Vector) -> 'Point': ...

    def __sub__(self, p: 'Point | Vector') -> 'Point | Vector':
        if isinstance(p, Point):
            return Vector(self.loc - p.loc)
        else:
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

    def __hash__(self):
        return hash((self.x.item(), self.y.item()))

    def __str__(self):
        return f"Point({float(self.x)}, {float(self.y)})"

    def __repr__(self):
        return self.__str__()

    @classproperty
    def origin(cls):
        return Point(jnp.array([0.0, 0.0]))

    @classmethod
    def polar(cls, r: Scalar, θ: Scalar):
        return cls(jnp.array([r * jnp.cos(θ), r * jnp.sin(θ)]))

    @classmethod
    def mk(cls, x: 'PointI | Scalar', y: Scalar | None = None) -> 'Point':
        if (isinstance(x, float) or (isinstance(x, jax.Array) and x.ndim == 0)) and y is not None:
            return cls(jnp.array([x, y]))
        assert y is None
        if isinstance(x, Point):
            return x
        else:
            assert isinstance(x, jax.Array | tuple)
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

    # TODO: hash

    def __str__(self):
        return f"({self.x} : {self.y} : {self.z})"

    def __eq__(self, other):
        for i in range(3):
            if other.loc[i] != 0:
                t = self.loc[i] / other.loc[i]
                if jnp.allclose(self.loc, t * other.loc, atol=Global.approx_eps):
                    return True
        return False


class ProjectiveTransformation:
    """
    Represents a 2D projective transformation.
    """
    def __init__(self, matrix: jax.Array):
        assert matrix.shape == (3, 3)
        assert not jnp.allclose(matrix[2, 2], 0.0, atol=Global.approx_eps)
        assert not jnp.allclose(jnp.linalg.det(matrix), 0.0, atol=Global.approx_eps)  # nonsingular
        self.matrix = matrix


    def __matmul__(self: 'ProjectiveTransformation', other: 'ProjectiveTransformation') -> 'ProjectiveTransformation':
        return ProjectiveTransformation(self.matrix @ other.matrix)

    @overload
    def __call__(self, p: ProjPoint) -> ProjPoint: ...
    @overload
    def __call__(self, p: Point) -> Point | None: ...

    def __call__(self, p: Point | ProjPoint) -> Point | None | ProjPoint:
        projective = isinstance(p, ProjPoint)
        if not projective:
            p = p.to_proj_point()
        pp = ProjPoint(jnp.dot(self.matrix, p.loc))
        if not projective:
            return pp.to_point()
        else:
            return pp

    def apply_batch(self, points: jax.Array) -> jax.Array:
        projective = points.shape[1] == 3
        if not projective:
            points = jnp.concat([points, jnp.ones((points.shape[0], 1))], axis=1)
        transformed = jnp.dot(self.matrix, points.T).T
        if not projective:
            return transformed[:, :2] / transformed[:, 2][:, None]
        else:
            return transformed

    def inverse(self) -> 'ProjectiveTransformation':
        inverse = jnp.linalg.inv(self.matrix)
        return ProjectiveTransformation(inverse)

    def is_identity(self):
        return jnp.allclose(self.matrix, jnp.eye(3))

    @classmethod
    def identity(cls):
        return cls(jnp.eye(3))


class AffineTransformation(ProjectiveTransformation):
    """
    Represents a 2D affine transformation.
    """
    def __init__(self, matrix: jax.Array):
        super().__init__(matrix)
        assert self.matrix[2, 0] == 0.0
        assert self.matrix[2, 1] == 0.0

    @overload
    def __matmul__(self: 'AffineTransformation', other: 'AffineTransformation') -> 'AffineTransformation': ...
    @overload
    def __matmul__(self: 'AffineTransformation', other: 'ProjectiveTransformation') -> 'ProjectiveTransformation': ...

    def __matmul__(self: 'AffineTransformation', other: 'ProjectiveTransformation') -> 'ProjectiveTransformation':
        if isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)

    def inverse(self):
        inverse = jnp.linalg.inv(self.matrix)
        return AffineTransformation(inverse)

    def decompose(self) -> tuple['Translation', 'Rotation', 'ShearX', 'Scaling']:
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
        return tr, Rotation(θ), ShearX(m), Scaling((sx.item(), sy.item()))

    def __repr__(self):
        return f"Transformation({self.matrix})"

    @classmethod
    def identity(cls):
        return cls(jnp.eye(3))


class RigidTransformation(AffineTransformation):
    pass


class Translation(RigidTransformation):
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

    @overload
    def __matmul__(self: 'Translation', other: 'Translation') -> 'Translation': ...
    @overload
    def __matmul__(self: 'Translation', other: AffineTransformation) -> AffineTransformation: ...
    @overload
    def __matmul__(self: 'Translation', other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self: 'Translation', other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, Translation):
            return Translation(self.vec + other.vec)
        elif isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)


class Rotation(RigidTransformation):
    def __init__(self, θ: Scalar):
        super().__init__(jnp.array([
            [math.cos(θ), -math.sin(θ), 0],
            [math.sin(θ), math.cos(θ), 0],
            [0, 0, 1]
        ]))
        self.angle = θ

    @overload
    def __matmul__(self: 'Rotation', other: 'Rotation') -> 'Rotation': ...
    @overload
    def __matmul__(self: 'Rotation', other: AffineTransformation) -> AffineTransformation: ...
    @overload
    def __matmul__(self: 'Rotation', other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self: 'Rotation', other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, Rotation):
            return Rotation(self.angle + other.angle)
        elif isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)

    def inverse(self):
        return Rotation(-self.angle)

    @classmethod
    def centered(cls, θ: Scalar, center: PointI = Point.origin):
        center = Point.mk(center)
        if center == Point.origin:
            return cls(θ)
        else:
            v = center.to_vector()
            return Translation(v) @ cls(θ) @ Translation(-v)


class Scaling(AffineTransformation):
    def __init__(self, s: VectorI):
        s = Vector.mk(s)
        super().__init__(jnp.diag(jnp.concat([s.vec, jnp.array([1])])))
        self.scale = s

    @classmethod
    def centered(cls, s: VectorI, center: Point = Point.origin):
        s = Vector.mk(s)
        sc = cls(s)
        if center == Point.origin:
            return sc
        else:
            v = center.to_vector()
            return Translation(v) @ sc @ Translation(-v)


class ShearX(AffineTransformation):
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


class ShearY(AffineTransformation):
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


class Elation(ProjectiveTransformation):
    def __init__(self, vec: PointI):
        vec = Point.mk(vec)
        super().__init__(jnp.array([
            [1, 0, 0],
            [0, 1, 0],
            [vec.x, vec.y, 1]
        ]))
        self.vec = vec


class Reflection(RigidTransformation):
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
