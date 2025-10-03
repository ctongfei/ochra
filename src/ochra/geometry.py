"""
Defines the geometric primitives used in Ochra.
"""
from __future__ import annotations
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float

from ochra.util import Global, classproperty, f2s

if TYPE_CHECKING:
    from ochra.core import Line


τ = math.tau
r"""The [true circle constant](https://en.wikipedia.org/wiki/Tau_(mathematics)) $\tau = 2\pi$, 
the ratio of a circle's circumference to its radius."""

type Scalar = float | Float[jax.Array, ""]
"""A scalar value, which can be a float or a Jax array of dimensionality 0."""


@jdc.pytree_dataclass
class Vector:
    r"""
    Represents a 2D vector $(x, y) \in \mathbb{R}^2$.
    Vectors are elements in the 2D vector space.
    """

    vec: Float[jax.Array, "2"]

    @property
    def x(self) -> Scalar:
        """Returns the x-coordinate of the vector."""
        return self.vec[0]

    @property
    def y(self) -> Scalar:
        """Returns the y-coordinate of the vector."""
        return self.vec[1]

    @property
    def angle(self) -> Scalar:
        """Returns the angle of the vector with the positive x-axis."""
        return jnp.arctan2(self.y, self.x)

    def __add__(self, v: Vector) -> Vector:
        """Returns the vector sum."""
        return Vector(self.vec + v.vec)

    def __sub__(self, v: Vector) -> Vector:
        """Returns the vector difference."""
        return Vector(self.vec - v.vec)

    def __mul__(self, s: Scalar) -> Vector:
        """Returns the vector scaled by a scalar."""
        return Vector(self.vec * s)

    def __neg__(self) -> Vector:
        """Returns the vector with the opposite direction."""
        return Vector(-self.vec)

    def __abs__(self) -> Scalar:
        """Returns the norm of the vector."""
        return jnp.linalg.norm(self.vec)

    def dot(self, v: Vector) -> Scalar:
        """Returns the dot product of this vector and another vector."""
        return jnp.dot(self.vec, v.vec)

    def norm(self) -> Scalar:
        """Returns the norm of the vector."""
        return abs(self)

    def normalize(self) -> Vector:
        """Returns the unit vector of length 1 in the same direction as this vector."""
        return Vector(self.vec / abs(self))

    def rotate(self, θ: Scalar) -> Vector:
        """Returns the vector rotated by an angle."""
        rot = jnp.array([[math.cos(θ), -math.sin(θ)], [math.sin(θ), math.cos(θ)]])
        return Vector(jnp.dot(rot, self.vec))

    def to_point(self) -> Point:
        """Converts the vector to a point."""
        return Point(self.vec)

    @classmethod
    def mk(cls, x: VectorI | Scalar, y: Scalar | None = None) -> Vector:
        """
        Creates a vector from a tuple, a list, a :py:class:`jax.Array` or another vector.
        This should be preferred over the constructor for readability.
        """
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
        r"""Returns the unit vector $(\cos \theta, \sin \theta)$ of length 1 in the direction of the given angle $\theta$."""
        return cls.mk(jnp.array([jnp.cos(θ), jnp.sin(θ)]))


@jdc.pytree_dataclass
class Point:
    r"""
    Represents a 2D point $(x, y) \in \mathbb{A}^2$.
    Points are elements in the 2D affine space.
    """

    loc: Float[jax.Array, "2"]

    @property
    def x(self) -> Scalar:
        return self.loc[0]

    @property
    def y(self) -> Scalar:
        return self.loc[1]

    def __add__(self, v: Vector) -> Point:
        return Point(self.loc + v.vec)

    @overload
    def __sub__(self, p: Point) -> Vector: ...

    @overload
    def __sub__(self, p: Vector) -> Point: ...

    def __sub__(self, p: Point | Vector) -> Point | Vector:
        if isinstance(p, Point):
            return Vector(self.loc - p.loc)
        else:
            return Point(self.loc - p.vec)

    def __eq__(self, other):
        return jnp.allclose(self.loc, other.loc, atol=Global.approx_eps).item()

    def scale(self, sx: Scalar, sy: Scalar) -> Point:
        return Point(self.loc * jnp.array([sx, sy]))

    def translate(self, dx: Scalar, dy: Scalar) -> Point:
        return self + Vector.mk((dx, dy))

    def to_vector(self) -> Vector:
        return Vector(self.loc)

    def to_proj_point(self) -> ProjPoint:
        return ProjPoint(jnp.concat([self.loc, jnp.array([1])]))

    def __str__(self):
        return f"Point({f2s(self.x)}, {f2s(self.y)})"

    def __repr__(self):
        return self.__str__()

    @classproperty
    def origin(cls) -> Point:
        """Returns the origin point $(0, 0)$."""
        return cls(jnp.array([0.0, 0.0]))  # type: ignore

    @classmethod
    def polar(cls, r: Scalar, θ: Scalar):
        r"""Creates a point $(r \cos \theta, r \sin \theta)$ from polar coordinates $(r, \theta)$."""
        return cls(jnp.array([r * jnp.cos(θ), r * jnp.sin(θ)]))

    @classmethod
    def mk(cls, x: PointI | Scalar, y: Scalar | None = None) -> Point:
        """
        Creates a point from a tuple, a list, a `jax.Array` or another point.
        This should be preferred over the constructor for readability.
        """
        if (
            isinstance(x, float) or (isinstance(x, int)) or (isinstance(x, jax.Array) and x.ndim == 0)
        ) and y is not None:
            return cls(jnp.array([x, y]))
        assert y is None
        if isinstance(x, Point):
            assert isinstance(x.loc, jax.Array)
            return x
        else:
            assert isinstance(x, jax.Array | tuple)
            return cls(jnp.array([float(x[0]), float(x[1])]))


@jdc.pytree_dataclass
class ProjPoint:
    r"""Represents a 2D point in the projective plane $\mathbb{P}^2$."""

    loc: Float[jax.Array, "3"]

    @property
    def x(self) -> Scalar:
        return self.loc[0]

    @property
    def y(self) -> Scalar:
        return self.loc[1]

    @property
    def z(self) -> Scalar:
        return self.loc[2]

    def to_point_safe(self) -> Point | None:  # None if infinity point
        """Converts the projective point to an affine point. Returns `None` if the point is an infinity point."""
        if self.is_infinity_point():
            return None
        return self.to_point()

    def to_point(self) -> Point:
        return Point(self.loc[:2] / self.z)

    def is_infinity_point(self):
        """Returns True if the point is an infinity point."""
        max_component = jnp.max(jnp.abs(self.loc))
        w = self.loc[2] / max_component
        return jnp.isclose(w, 0.0, atol=Global.approx_eps)

    def __str__(self):
        return f"({self.x} : {self.y} : {self.z})"

    def __eq__(self, other):
        for i in range(3):
            if other.loc[i] != 0:
                t = self.loc[i] / other.loc[i]
                if jnp.allclose(self.loc, t * other.loc, atol=Global.approx_eps):
                    return True
        return False


type PointI = Point | tuple[float, float] | Float[jax.Array, "2"]
type VectorI = Vector | tuple[Scalar, Scalar] | Float[jax.Array, "2"]

type LineI = tuple[float, float, float] | Float[jax.Array, "3"]
# a x + b y + c = 0

type ConicI = tuple[float, float, float, float, float, float] | Float[jax.Array, "3 3"]
# a x^2 + b x y + c y^2 + d x + e y + f = 0


@jdc.pytree_dataclass
class VectorSequence(Sequence[Vector]):
    """
    Represents a sequence of vectors in the plane as a `jax.Array`.
    This enables vectorized operations on the vectors.
    """

    vectors: Float[jax.Array, "n 2"]

    @overload
    def __getitem__(self, index: int) -> Vector: ...
    @overload
    def __getitem__(self, index: slice) -> VectorSequence: ...

    def __getitem__(self, index: int | slice) -> Vector | VectorSequence:
        if isinstance(index, slice):
            return VectorSequence(self.vectors[index, :])
        return Vector(self.vectors[index, :])

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield Vector(self.vectors[i, :])

    @classmethod
    def mk(cls, vectors: Sequence[VectorI] | jax.Array | VectorSequence):
        if isinstance(vectors, VectorSequence):
            return vectors
        elif isinstance(vectors, jax.Array):
            assert vectors.shape[1] == 2, "Vectors must be 2-dimensional."
            return cls(vectors)
        vecs = [Vector.mk(v).vec for v in vectors]
        return cls(jnp.stack(vecs))


@jdc.pytree_dataclass
class PointSequence(Sequence[Point]):
    """
    Represents a sequence of points in the plane as a `jax.Array`.
    This enables vectorized operations on the points.
    """

    points: Float[jax.Array, "n 2"]

    @overload
    def __getitem__(self, index: int) -> Point: ...
    @overload
    def __getitem__(self, index: slice) -> PointSequence: ...

    def __getitem__(self, index: int | slice) -> Point | PointSequence:
        if isinstance(index, slice):
            return PointSequence(self.points[index, :])
        return Point(self.points[index, :])

    def __len__(self) -> int:
        return self.points.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield Point(self.points[i, :])

    def to_proj(self) -> ProjPointSequence:
        return ProjPointSequence(jnp.concat([self.points, jnp.ones((len(self), 1))], axis=1))

    def transform(self, f: ProjectiveTransformation) -> PointSequence:
        return f.apply_batch(self)

    @classmethod
    def mk(cls, points: Sequence[PointI] | jax.Array | PointSequence):
        if isinstance(points, PointSequence):
            return points
        elif isinstance(points, jax.Array):
            assert points.shape[1] == 2, "Points must be 2-dimensional."
            return cls(points)
        vecs = [Point.mk(p).loc for p in points]
        return cls(jnp.stack(vecs))


type VectorSequenceI = Sequence[VectorI] | jax.Array | VectorSequence
type PointSequenceI = Sequence[PointI] | jax.Array | PointSequence


@jdc.pytree_dataclass
class ProjPointSequence(Sequence[ProjPoint]):
    """
    Represents a sequence of projective points in the plane as a `jax.Array`.
    This enables vectorized operations on the points.
    """

    points: Float[jax.Array, "n 3"]

    @overload
    def __getitem__(self, index: int) -> ProjPoint: ...
    @overload
    def __getitem__(self, index: slice) -> ProjPointSequence: ...

    def __getitem__(self, index: int | slice) -> ProjPoint | ProjPointSequence:
        if isinstance(index, slice):
            return ProjPointSequence(self.points[index, :])
        return ProjPoint(self.points[index, :])

    def __len__(self) -> int:
        return self.points.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield ProjPoint(self.points[i, :])

    def to_affine(self) -> PointSequence:
        return PointSequence(self.points[:, :2] / self.points[:, 2][:, None])

    def transform(self, f: ProjectiveTransformation) -> ProjPointSequence:
        return f.apply_batch(self)


class ProjectiveTransformation:
    """
    Represents a 2D projective transformation.
    """

    def __init__(self, matrix: Float[jax.Array, "3 3"]):
        assert matrix.shape == (3, 3)
        assert not jnp.allclose(matrix[2, 2], 0.0, atol=Global.approx_eps)
        assert not jnp.allclose(jnp.linalg.det(matrix), 0.0, atol=Global.approx_eps)  # nonsingular
        self.matrix = matrix

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        return ProjectiveTransformation(self.matrix @ other.matrix)

    @overload
    def __call__(self, p: Point) -> Point | None: ...
    @overload
    def __call__(self, p: ProjPoint) -> ProjPoint: ...

    def __call__(self, p: Point | ProjPoint) -> Point | ProjPoint | None:
        projective = isinstance(p, ProjPoint)
        if not projective:
            p = p.to_proj_point()
        pp = ProjPoint(jnp.dot(self.matrix, p.loc))
        if not projective:
            return pp.to_point_safe()
        else:
            return pp

    def unsafe_apply(self, p: Point) -> Point:
        pp = ProjPoint(jnp.dot(self.matrix, p.to_proj_point().loc))
        return pp.to_point()

    @overload
    def apply_batch(self, ps: PointSequence) -> PointSequence: ...
    @overload
    def apply_batch(self, ps: ProjPointSequence) -> ProjPointSequence: ...

    def apply_batch(self, ps: PointSequence | ProjPointSequence) -> PointSequence | ProjPointSequence:
        projective = isinstance(ps, ProjPointSequence)
        if not projective:
            points = jnp.concat([ps.points, jnp.ones((len(ps), 1))], axis=1)
        else:
            points = ps.points
        transformed = jnp.dot(self.matrix, points.T).T
        if not projective:
            return PointSequence(transformed[:, :2] / transformed[:, 2][:, None])
        else:
            return ProjPointSequence(transformed)

    def inverse(self) -> ProjectiveTransformation:
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

    def __init__(self, matrix: Float[jax.Array, "3 3"]):
        super().__init__(matrix)
        assert self.matrix[2, 0] == 0.0
        assert self.matrix[2, 1] == 0.0

    @overload
    def __call__(self, p: Point) -> Point: ...
    @overload
    def __call__(self, p: ProjPoint) -> ProjPoint: ...

    def __call__(self, p: Point | ProjPoint) -> Point | ProjPoint:
        projective = isinstance(p, ProjPoint)
        if not projective:
            p = p.to_proj_point()
        pp = ProjPoint(jnp.dot(self.matrix, p.loc))
        if not projective:
            return pp.to_point()
        else:
            return pp

    @overload
    def __matmul__(self, other: AffineTransformation) -> AffineTransformation: ...
    @overload
    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)

    def inverse(self):
        inverse = jnp.linalg.inv(self.matrix)
        return AffineTransformation(inverse)

    def decompose(self) -> tuple["Translation", "Rotation", "ShearX", "Scaling"]:
        """
        Decomposes an affine transformation into translation, rotation, shear and scaling.
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
    @overload
    def __matmul__[Tr: (RigidTransformation, AffineTransformation, RigidTransformation)](self, other: Tr) -> Tr: ...
    @overload
    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, RigidTransformation):
            return RigidTransformation(self.matrix @ other.matrix)
        elif isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)

    def decompose_rigid(self) -> tuple["Translation", "Rotation"]:
        return Translation(self.matrix[:2, 2]), Rotation(jnp.arctan2(self.matrix[1, 0], self.matrix[0, 0]))


class Translation(RigidTransformation):
    def __init__(self, d: VectorI):
        d = Vector.mk(d)
        super().__init__(jnp.array([[1, 0, d.x], [0, 1, d.y], [0, 0, 1]]))
        self.vec = d

    def inverse(self):
        return Translation(-self.vec)

    @overload
    def __matmul__[Tr: ("Translation", RigidTransformation, AffineTransformation)](self, other: Tr) -> Tr: ...
    @overload
    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, Translation):
            return Translation(self.vec + other.vec)
        elif isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)


class Rotation(RigidTransformation):
    def __init__(self, θ: Scalar):
        super().__init__(jnp.array([[math.cos(θ), -math.sin(θ), 0], [math.sin(θ), math.cos(θ), 0], [0, 0, 1]]))
        self.angle = θ

    @overload
    def __matmul__[Tr: ("Rotation", RigidTransformation, AffineTransformation)](self, other: Tr) -> Tr: ...
    @overload
    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, Rotation):
            return Rotation(self.angle + other.angle)
        elif isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)

    def inverse(self):
        return Rotation(-self.angle)

    @classmethod
    def centered(cls, θ: Scalar, center: PointI = Point.origin) -> RigidTransformation:
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

    @overload
    def __matmul__[Tr: ("Scaling", AffineTransformation)](self, other: Tr) -> Tr: ...
    @overload
    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        if isinstance(other, Scaling):
            return Scaling(Vector(self.scale.vec * other.scale.vec))
        elif isinstance(other, AffineTransformation):
            return AffineTransformation(self.matrix @ other.matrix)
        return ProjectiveTransformation(self.matrix @ other.matrix)

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
        super().__init__(jnp.array([[1, m, 0], [0, 1, 0], [0, 0, 1]]))
        self.factor = m
        self.angle = jnp.arctan(m)

    @classmethod
    def from_angle(cls, θ: Scalar):
        return cls(jnp.tan(θ))


class ShearY(AffineTransformation):
    def __init__(self, m: Scalar):
        super().__init__(jnp.array([[1, 0, 0], [m, 1, 0], [0, 0, 1]]))
        self.factor = m
        self.angle = jnp.arctan(m)

    @classmethod
    def from_angle(cls, θ: Scalar):
        return cls(jnp.tan(θ))


class Elation(ProjectiveTransformation):
    def __init__(self, vec: VectorI):
        vec = Vector.mk(vec)
        super().__init__(jnp.array([[1, 0, 0], [0, 1, 0], [vec.x, vec.y, 1]]))
        self.vec = vec


class Reflection(RigidTransformation):
    def __init__(self, line: "LineI | Line"):
        from ochra.core import Line

        if isinstance(line, Line):
            a, b, c = line.coef
        else:
            a, b, c = line
        d = math.hypot(a, b)
        a, b, c = a / d, b / d, c / d
        super().__init__(
            jnp.array([[1 - 2 * a**2, -2 * a * b, -2 * a * c], [-2 * a * b, 1 - 2 * b**2, -2 * b * c], [0, 0, 1]])
        )
        self.line = line
