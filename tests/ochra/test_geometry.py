import numpy as np
import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis.strategies import floats, lists
import pytest

from ochra import Global
from ochra.geometry import (
    AffineTransformation,
    Elation,
    Point,
    PointSequence,
    ProjectiveTransformation,
    ProjPoint,
    RigidTransformation,
    Rotation,
    Scaling,
    ShearX,
    ShearY,
    SimilarTransformation,
    Translation,
    UniformScaling,
    Vector,
    VectorSequence,
)


def test_proj_point():
    assert ProjPoint(jnp.array((1, 2, 3))) == ProjPoint(jnp.array((2, 4, 6)))
    assert ProjPoint(jnp.array((1, 2, 3))) != ProjPoint(jnp.array((2, 4, 7)))
    assert ProjPoint(jnp.array((1, 2, 0))) == ProjPoint(jnp.array((2, 4, 0)))
    assert ProjPoint(jnp.array((1, 2, 0))) != ProjPoint(jnp.array((1, 2, 1)))


def test_point_and_vector_normalization():
    assert Point.mk([1, 2]) == Point.mk((1, 2))
    assert jnp.array_equal(Vector.mk([1, 2]).vec, jnp.array([1.0, 2.0]))
    assert jnp.array_equal((2 * Vector.mk([1, 2])).vec, jnp.array([2.0, 4.0]))
    assert jnp.array_equal((Vector.mk([2, 4]) / 2).vec, jnp.array([1.0, 2.0]))
    assert PointSequence.mk([]).points.shape == (0, 2)
    assert VectorSequence.mk([]).vectors.shape == (0, 2)

    with pytest.raises(AssertionError):
        Point.mk([1, 2, 3])
    with pytest.raises(AssertionError):
        Vector.mk([1])


def test_projective_transformation_accepts_zero_bottom_right_entry():
    matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    transformation = ProjectiveTransformation(matrix)

    assert transformation(Point.mk((2, 4))) == Point.mk((0.5, 0.25))


def test_transformation_classes_validate_their_invariants():
    with pytest.raises(AssertionError):
        ProjectiveTransformation(jnp.zeros((3, 3)))
    with pytest.raises(AssertionError):
        AffineTransformation(jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    with pytest.raises(AssertionError):
        SimilarTransformation(Scaling((2, 3)).matrix)
    with pytest.raises(AssertionError):
        RigidTransformation(UniformScaling(2).matrix)


def test_transformation_composition_preserves_specific_type():
    assert isinstance(Translation((1, 2)) @ Translation((3, 4)), Translation)
    assert isinstance(Rotation(0.1) @ Rotation(0.2), Rotation)
    assert isinstance(UniformScaling(2) @ UniformScaling(3), UniformScaling)
    assert isinstance(Scaling((2, 3)) @ Scaling((4, 5)), Scaling)
    assert isinstance(ShearX(1) @ ShearX(2), ShearX)
    assert isinstance(ShearY(1) @ ShearY(2), ShearY)
    assert isinstance(Elation((1, 2)) @ Elation((3, 4)), Elation)
    assert isinstance(Translation((1, 2)) @ Rotation(0.1), RigidTransformation)
    assert isinstance(Rotation(0.1) @ UniformScaling(2), SimilarTransformation)
    assert isinstance(Rotation(0.1) @ Scaling((2, 3)), AffineTransformation)


def test_centered_transformations_preserve_type_and_fixed_point():
    center = Point.mk((1, 2))
    rotation = Rotation.centered(0.5, center)
    scaling = UniformScaling.centered_uniform(2, center)
    generic_uniform_scaling = Scaling.centered((2, 2), center)

    assert isinstance(rotation, RigidTransformation)
    assert isinstance(scaling, SimilarTransformation)
    assert isinstance(generic_uniform_scaling, SimilarTransformation)
    assert rotation(center) == center
    assert scaling(center) == center
    assert generic_uniform_scaling(center) == center


@pytest.mark.parametrize(
    "transformation_type",
    [Translation, Rotation, UniformScaling, Scaling, ShearX, ShearY, Elation],
)
def test_transformation_identity_preserves_type(transformation_type):
    identity = transformation_type.identity()

    assert type(identity) is transformation_type
    assert jnp.array_equal(identity.matrix, jnp.eye(3))


@pytest.mark.parametrize(
    "transformation",
    [
        Translation((1, 2)),
        Rotation(0.3),
        UniformScaling(2),
        Scaling((2, 3)),
        ShearX(0.5),
        ShearY(0.5),
        Elation((0.1, 0.2)),
    ],
)
def test_transformation_inverse_preserves_type(transformation):
    inverse = transformation.inverse()

    assert type(inverse) is type(transformation)
    assert jnp.allclose((transformation @ inverse).matrix, jnp.eye(3), atol=Global.approx_eps)


def nonsingular(m: np.ndarray):
    if np.isnan(m).any() or np.isinf(m).any():
        return False
    d = np.linalg.det(m)
    return not jnp.isclose(d, 0.0, atol=Global.approx_eps) and not np.isinf(d) and not np.isnan(d)


@given(
    m=lists(floats(-10.0, 10.0, allow_nan=False, allow_infinity=False, width=32), min_size=9, max_size=9),
)
@settings(deadline=None)
def test_affine_transformation_decomposition(m):
    # generate random transformation by generating a 3x3 matrix
    # and then decomposing it
    m = np.asarray(m).reshape(3, 3)
    m = jnp.asarray(m).at[2, :2].set(0.0)
    m = jnp.clip(m, -10.0, 10.0)  # somehow hypothesis generates values outside the specified range
    if not nonsingular(np.asarray(m)):
        return
    m = m / m[2, 2]
    t = AffineTransformation(jnp.asarray(m))
    tr, rot, shx, sc = t.decompose()
    reconstructed = tr @ rot @ shx @ sc
    scale = jnp.max(jnp.abs(t.matrix))
    assert jnp.allclose(t.matrix, reconstructed.matrix, atol=Global.approx_eps * scale)


def test_affine_transformation_decomposition_is_jittable():
    def decompose(matrix):
        translation, rotation, shear, scaling = AffineTransformation(matrix).decompose()
        return translation.vec.vec, rotation.angle, shear.factor, scaling.scale.vec

    matrix = jnp.array([[2.0, 1.0, 3.0], [1.0, 4.0, 5.0], [0.0, 0.0, 1.0]])
    translation, angle, shear, scaling = jax.jit(decompose)(matrix)

    assert jnp.allclose(translation, jnp.array([3.0, 5.0]))
    assert jnp.all(jnp.isfinite(jnp.array([angle, shear, *scaling])))


def test_affine_transformation_decomposition_is_differentiable():
    def decomposition_sum(x):
        matrix = jnp.array([[x, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        _, rotation, shear, scaling = AffineTransformation(matrix).decompose()
        return rotation.angle + shear.factor + jnp.sum(scaling.scale.vec)

    derivative = jax.jit(jax.grad(decomposition_sum))(2.0)

    assert jnp.isfinite(derivative)


@pytest.mark.parametrize(
    "matrix",
    [
        jnp.array([[2.0, 1.0, 3.0], [1.0, 4.0, 5.0], [0.0, 0.0, 1.0]]),
        jnp.array([[0.0, -2.0, 3.0], [3.0, 0.0, 5.0], [0.0, 0.0, 1.0]]),
        jnp.array([[-1.0, 0.5, 3.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_affine_transformation_svd_decomposition(matrix):
    transformation = AffineTransformation(matrix)
    translation, rotation_left, scaling, rotation_right = transformation.decompose_svd()
    reconstructed = translation @ rotation_left @ scaling @ rotation_right

    assert jnp.allclose(transformation.matrix, reconstructed.matrix, atol=Global.approx_eps)
    assert isinstance(rotation_left, Rotation)
    assert isinstance(rotation_right, Rotation)
    assert jnp.sign(scaling.scale.x * scaling.scale.y) == jnp.sign(jnp.linalg.det(matrix[:2, :2]))
