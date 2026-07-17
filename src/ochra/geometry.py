"""
Defines the geometric primitives used in Ochra.
"""
from __future__ import annotations
import math
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, overload

import jax
import jax.core
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


def _cos(x: Scalar) -> Scalar:
    return jnp.cos(x) if isinstance(x, jax.Array | jax.core.Tracer) else math.cos(x)


def _sin(x: Scalar) -> Scalar:
    return jnp.sin(x) if isinstance(x, jax.Array | jax.core.Tracer) else math.sin(x)


def _assert_if_concrete(condition: bool | jax.Array, message: str) -> None:
    """Checks value-dependent invariants eagerly without forcing JAX tracers to concrete values."""
    if not isinstance(condition, jax.core.Tracer):
        assert bool(condition), message


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

    def __rmul__(self, s: Scalar) -> Vector:
        """Returns the vector scaled by a scalar."""
        return self * s

    def __truediv__(self, s: Scalar) -> Vector:
        """Returns the vector divided by a scalar."""
        return Vector(self.vec / s)

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
        cos_θ, sin_θ = _cos(θ), _sin(θ)
        rot = jnp.array([[cos_θ, -sin_θ], [sin_θ, cos_θ]])
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
        if y is not None:
            return cls(jnp.asarray([x, y], dtype=float))
        if isinstance(x, Vector):
            return x
        vector = jnp.asarray(x, dtype=float)
        assert vector.shape == (2,), "Vectors must be 2-dimensional."
        return cls(vector)

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
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
        if y is not None:
            return cls(jnp.asarray([x, y], dtype=float))
        if isinstance(x, Point):
            return x
        point = jnp.asarray(x, dtype=float)
        assert point.shape == (2,), "Points must be 2-dimensional."
        return cls(point)


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjPoint):
            return False
        for i in range(3):
            if other.loc[i] != 0:
                t = self.loc[i] / other.loc[i]
                if jnp.allclose(self.loc, t * other.loc, atol=Global.approx_eps):
                    return True
        return False


type PointI = Point | Sequence[Scalar] | Float[jax.Array, "2"]
type VectorI = Vector | Sequence[Scalar] | Float[jax.Array, "2"]

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
            assert vectors.ndim == 2 and vectors.shape[1] == 2, "Vectors must be 2-dimensional."
            return cls(vectors)
        vecs = [Vector.mk(v).vec for v in vectors]
        if len(vecs) == 0:
            return cls(jnp.empty((0, 2)))
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
            assert points.ndim == 2 and points.shape[1] == 2, "Points must be 2-dimensional."
            return cls(points)
        vecs = [Point.mk(p).loc for p in points]
        if len(vecs) == 0:
            return cls(jnp.empty((0, 2)))
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
        matrix = jnp.asarray(matrix, dtype=float)
        assert matrix.shape == (3, 3)
        _assert_if_concrete(jnp.all(jnp.isfinite(matrix)), "Transformation matrix must be finite.")
        _assert_if_concrete(jnp.linalg.det(matrix) != 0.0, "Transformation matrix must be nonsingular.")
        self.matrix = matrix

    @overload
    def __matmul__(self: Translation, other: Translation) -> Translation: ...
    @overload
    def __matmul__(self: Rotation, other: Rotation) -> Rotation: ...
    @overload
    def __matmul__(self: UniformScaling, other: UniformScaling) -> UniformScaling: ...
    @overload
    def __matmul__(self: Scaling, other: Scaling) -> Scaling: ...
    @overload
    def __matmul__(self: ShearX, other: ShearX) -> ShearX: ...
    @overload
    def __matmul__(self: ShearY, other: ShearY) -> ShearY: ...
    @overload
    def __matmul__(self: Elation, other: Elation) -> Elation: ...
    @overload
    def __matmul__(self: RigidTransformation, other: RigidTransformation) -> RigidTransformation: ...
    @overload
    def __matmul__(self: SimilarTransformation, other: SimilarTransformation) -> SimilarTransformation: ...
    @overload
    def __matmul__(self: AffineTransformation, other: AffineTransformation) -> AffineTransformation: ...
    @overload
    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation: ...

    def __matmul__(self, other: ProjectiveTransformation) -> ProjectiveTransformation:
        return _compose_transformations(self, other)

    @overload
    def __call__(self, p: Point) -> Point | None: ...
    @overload
    def __call__(self, p: ProjPoint) -> ProjPoint: ...

    def __call__(self, p: Point | ProjPoint) -> Point | ProjPoint | None:
        projective = isinstance(p, ProjPoint)
        if not projective:
            assert isinstance(p, Point)
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

    @overload
    def inverse(self: Translation) -> Translation: ...
    @overload
    def inverse(self: Rotation) -> Rotation: ...
    @overload
    def inverse(self: UniformScaling) -> UniformScaling: ...
    @overload
    def inverse(self: Scaling) -> Scaling: ...
    @overload
    def inverse(self: ShearX) -> ShearX: ...
    @overload
    def inverse(self: ShearY) -> ShearY: ...
    @overload
    def inverse(self: Elation) -> Elation: ...
    @overload
    def inverse(self: Reflection) -> Reflection: ...
    @overload
    def inverse(self: RigidTransformation) -> RigidTransformation: ...
    @overload
    def inverse(self: SimilarTransformation) -> SimilarTransformation: ...
    @overload
    def inverse(self: AffineTransformation) -> AffineTransformation: ...
    @overload
    def inverse(self) -> ProjectiveTransformation: ...

    def inverse(self) -> ProjectiveTransformation:
        return _inverse_transformation(self)

    def is_identity(self):
        return jnp.allclose(self.matrix, jnp.eye(3))

    @classmethod
    def identity(cls):
        if issubclass(cls, Translation):
            return Translation((0.0, 0.0))
        if issubclass(cls, Rotation):
            return Rotation(0.0)
        if issubclass(cls, UniformScaling):
            return UniformScaling(1.0)
        if issubclass(cls, Scaling):
            return Scaling((1.0, 1.0))
        if issubclass(cls, ShearX):
            return ShearX(0.0)
        if issubclass(cls, ShearY):
            return ShearY(0.0)
        if issubclass(cls, Elation):
            return Elation((0.0, 0.0))
        if issubclass(cls, Reflection):
            return RigidTransformation(jnp.eye(3))
        return cls(jnp.eye(3))


class AffineTransformation(ProjectiveTransformation):
    r"""
    Represents a 2D affine transformation.
    An affine transformation is a projective transformation that preserves parallelism.
    """

    def __init__(self, matrix: Float[jax.Array, "3 3"]):
        super().__init__(matrix)
        _assert_if_concrete(
            jnp.allclose(self.matrix[2, :2], 0.0, atol=Global.approx_eps),
            "Affine transformations must preserve the line at infinity.",
        )

    @overload
    def __call__(self, p: Point) -> Point: ...
    @overload
    def __call__(self, p: ProjPoint) -> ProjPoint: ...

    def __call__(self, p: Point | ProjPoint) -> Point | ProjPoint:
        projective = isinstance(p, ProjPoint)
        if not projective:
            assert isinstance(p, Point)
            p = p.to_proj_point()
        pp = ProjPoint(jnp.dot(self.matrix, p.loc))
        if not projective:
            return pp.to_point()
        else:
            return pp

    def decompose(self) -> tuple["Translation", "Rotation", "ShearX", "Scaling"]:
        """
        Decomposes this transformation as translation, rotation, x-shear, and axis scaling.

        This is a QR decomposition of the linear part. A reflection is represented by a
        negative y scale. All returned parameters remain compatible with JAX transformations.
        """
        w = self.matrix[2, 2]
        m = self.matrix / w
        tr = Translation(m[:2, 2])
        m0 = m[:2, :2]
        sx = jnp.linalg.norm(m0[:, 0])
        θ = jnp.atan2(m0[1, 0], m0[0, 0])
        sy = jnp.linalg.det(m0) / sx
        shear = jnp.dot(m0[:, 0], m0[:, 1]) / (sx * sy)
        return tr, Rotation(θ), ShearX(shear), Scaling((sx, sy))

    def decompose_svd(self) -> tuple["Translation", "Rotation", "Scaling", "Rotation"]:
        """
        Decomposes this transformation as translation, rotation, axis scaling, and rotation.

        This is the singular value decomposition of the linear part. If the transformation
        includes a reflection, the second scale component is negative so both orthogonal
        factors remain rotations.
        """
        matrix = self.matrix / self.matrix[2, 2]
        transl = Translation(matrix[:2, 2])
        u, sv, vh = jnp.linalg.svd(matrix[:2, :2])

        det_u = jnp.linalg.det(u)
        det_vh = jnp.linalg.det(vh)
        u_rot = u @ jnp.diag(jnp.array([1.0, det_u]))
        vh_rot = jnp.diag(jnp.array([1.0, det_vh])) @ vh
        signed_scales = sv * jnp.array([1.0, det_u * det_vh])

        rot_l = Rotation(jnp.atan2(u_rot[1, 0], u_rot[0, 0]))
        scaling = Scaling(signed_scales)
        rot_r = Rotation(jnp.atan2(vh_rot[1, 0], vh_rot[0, 0]))
        return transl, rot_l, scaling, rot_r

    def __repr__(self):
        return f"Transformation({self.matrix})"

class SimilarTransformation(AffineTransformation):
    def __init__(self, matrix: Float[jax.Array, "3 3"]):
        super().__init__(matrix)
        linear = self.matrix[:2, :2] / self.matrix[2, 2]
        gram = linear.T @ linear
        _assert_if_concrete(
            jnp.allclose(gram, jnp.eye(2) * gram[0, 0], atol=Global.approx_eps),
            "Similar transformations must scale uniformly.",
        )

    def decompose_similar(self) -> tuple[Translation, Rotation, UniformScaling]:
        """Decomposes a similar transformation into the composition of translation, rotation and scaling."""
        tr, rot, _, sc = super().decompose()
        return tr, rot, UniformScaling(sc.scale.x)


class RigidTransformation(SimilarTransformation):
    def __init__(self, matrix: Float[jax.Array, "3 3"]):
        super().__init__(matrix)
        linear = self.matrix[:2, :2] / self.matrix[2, 2]
        _assert_if_concrete(
            jnp.allclose(linear.T @ linear, jnp.eye(2), atol=Global.approx_eps),
            "Rigid transformations must preserve distances.",
        )

    def decompose_rigid(self) -> tuple[Translation, Rotation]:
        matrix = self.matrix / self.matrix[2, 2]
        angle = jnp.atan2(matrix[1, 0], matrix[0, 0])
        return Translation(matrix[:2, 2]), Rotation(angle)


class Translation(RigidTransformation):
    def __init__(self, d: VectorI):
        d = Vector.mk(d)
        super().__init__(jnp.array([[1, 0, d.x], [0, 1, d.y], [0, 0, 1]]))
        self.vec = d


class Rotation(RigidTransformation):
    def __init__(self, θ: Scalar):
        cos_θ, sin_θ = _cos(θ), _sin(θ)
        super().__init__(jnp.array([[cos_θ, -sin_θ, 0], [sin_θ, cos_θ, 0], [0, 0, 1]]))
        self.angle = θ

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

    @classmethod
    def centered(cls, s: VectorI, center: PointI = Point.origin) -> AffineTransformation:
        s = Vector.mk(s)
        center = Point.mk(center)
        if s.x == s.y:
            return UniformScaling.centered_uniform(s.x, center)
        sc = cls(s)
        if center == Point.origin:
            return sc
        else:
            v = center.to_vector()
            return Translation(v) @ sc @ Translation(-v)


class UniformScaling(Scaling, SimilarTransformation):
    def __init__(self, s: Scalar):
        self.uniform_scale = s
        super().__init__(Vector.mk((s, s)))

    @classmethod
    def centered_uniform(cls, s: Scalar, center: PointI = Point.origin):
        center = Point.mk(center)
        if center == Point.origin:
            return cls(s)
        else:
            v = center.to_vector()
            return Translation(v) @ cls(s) @ Translation(-v)


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
        d = jnp.hypot(a, b)
        a, b, c = a / d, b / d, c / d
        self.coef = jnp.array([a, b, c])
        AffineTransformation.__init__(self,
            jnp.array([[1 - 2 * a**2, -2 * a * b, -2 * a * c], [-2 * a * b, 1 - 2 * b**2, -2 * b * c], [0, 0, 1]])
        )

    @cached_property
    def symmetry_line(self):
        from ochra.core import Line

        return Line.mk(self.coef)

def _compose_transformations(
    left: ProjectiveTransformation, right: ProjectiveTransformation
) -> ProjectiveTransformation:
    if isinstance(left, Translation) and isinstance(right, Translation):
        return Translation(left.vec + right.vec)
    if isinstance(left, Rotation) and isinstance(right, Rotation):
        return Rotation(left.angle + right.angle)
    if isinstance(left, UniformScaling) and isinstance(right, UniformScaling):
        return UniformScaling(left.uniform_scale * right.uniform_scale)
    if isinstance(left, Scaling) and isinstance(right, Scaling):
        return Scaling(Vector(left.scale.vec * right.scale.vec))
    if isinstance(left, ShearX) and isinstance(right, ShearX):
        return ShearX(left.factor + right.factor)
    if isinstance(left, ShearY) and isinstance(right, ShearY):
        return ShearY(left.factor + right.factor)
    if isinstance(left, Elation) and isinstance(right, Elation):
        return Elation(left.vec + right.vec)

    matrix = left.matrix @ right.matrix
    if isinstance(left, RigidTransformation) and isinstance(right, RigidTransformation):
        return RigidTransformation(matrix)
    if isinstance(left, SimilarTransformation) and isinstance(right, SimilarTransformation):
        return SimilarTransformation(matrix)
    if isinstance(left, AffineTransformation) and isinstance(right, AffineTransformation):
        return AffineTransformation(matrix)
    return ProjectiveTransformation(matrix)


def _inverse_transformation(transformation: ProjectiveTransformation) -> ProjectiveTransformation:
    if isinstance(transformation, Translation):
        return Translation(-transformation.vec)
    if isinstance(transformation, Rotation):
        return Rotation(-transformation.angle)
    if isinstance(transformation, UniformScaling):
        return UniformScaling(1 / transformation.uniform_scale)
    if isinstance(transformation, Scaling):
        return Scaling(Vector(1 / transformation.scale.vec))
    if isinstance(transformation, ShearX):
        return ShearX(-transformation.factor)
    if isinstance(transformation, ShearY):
        return ShearY(-transformation.factor)
    if isinstance(transformation, Elation):
        return Elation(-transformation.vec)
    if isinstance(transformation, Reflection):
        return Reflection(transformation.coef)

    inverse = jnp.linalg.inv(transformation.matrix)
    if isinstance(transformation, RigidTransformation):
        return RigidTransformation(inverse)
    if isinstance(transformation, SimilarTransformation):
        return SimilarTransformation(inverse)
    if isinstance(transformation, AffineTransformation):
        return AffineTransformation(inverse)
    return ProjectiveTransformation(inverse)
