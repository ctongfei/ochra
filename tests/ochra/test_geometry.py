import jax.numpy as jnp
import ochra as ox

def test_proj_point():
    assert ox.ProjPoint(jnp.array((1, 2, 3))) == ox.ProjPoint(jnp.array((2, 4, 6)))
    assert ox.ProjPoint(jnp.array((1, 2, 3))) != ox.ProjPoint(jnp.array((2, 4, 7)))
    assert ox.ProjPoint(jnp.array((1, 2, 0))) == ox.ProjPoint(jnp.array((2, 4, 0)))
    assert ox.ProjPoint(jnp.array((1, 2, 0))) != ox.ProjPoint(jnp.array((1, 2, 1)))

