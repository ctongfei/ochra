import jax.numpy as jnp
from ochra.geometry import ProjPoint


def test_proj_point():
    assert ProjPoint(jnp.array((1, 2, 3))) == ProjPoint(jnp.array((2, 4, 6)))
    assert ProjPoint(jnp.array((1, 2, 3))) != ProjPoint(jnp.array((2, 4, 7)))
    assert ProjPoint(jnp.array((1, 2, 0))) == ProjPoint(jnp.array((2, 4, 0)))
    assert ProjPoint(jnp.array((1, 2, 0))) != ProjPoint(jnp.array((1, 2, 1)))
