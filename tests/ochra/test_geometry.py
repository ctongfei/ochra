import numpy as np
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis.strategies import floats, lists

from ochra import Global
from ochra.geometry import ProjPoint, AffineTransformation


def test_proj_point():
    assert ProjPoint(jnp.array((1, 2, 3))) == ProjPoint(jnp.array((2, 4, 6)))
    assert ProjPoint(jnp.array((1, 2, 3))) != ProjPoint(jnp.array((2, 4, 7)))
    assert ProjPoint(jnp.array((1, 2, 0))) == ProjPoint(jnp.array((2, 4, 0)))
    assert ProjPoint(jnp.array((1, 2, 0))) != ProjPoint(jnp.array((1, 2, 1)))


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
    if not nonsingular(m):
        return
    m = m / m[2, 2]
    t = AffineTransformation(jnp.asarray(m))
    tr, rot, shx, sc = t.decompose()
    reconstructed = tr @ rot @ shx @ sc
    assert jnp.allclose(t.matrix, reconstructed.matrix, atol=Global.approx_eps)
