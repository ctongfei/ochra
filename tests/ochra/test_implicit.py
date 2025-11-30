import pytest
import jax.numpy as jnp

import ochra as ox
from .test_common import implicit_parametric_test_cases


def dup[E](e: E) -> tuple[E, E]:
    return e, e


def check_implicit_parametric_matching(impl: ox.Implicit, par: ox.Parametric):
    points = par.approx_as_polyline().points.points
    imp_func_vals = impl._raw_implicit_func_batched(points)
    max_error = jnp.max(jnp.abs(imp_func_vals))
    print(f"Max error = {max_error} for {impl}")
    assert jnp.isclose(0, max_error, atol=1e-3)


@pytest.mark.parametrize("elem_type", implicit_parametric_test_cases.keys())
def test_implicit_parametric_matching(elem_type: type):
    """
    Checks that the implicit and parametric representations of a curve match.
    """
    for e in implicit_parametric_test_cases[elem_type]:
        check_implicit_parametric_matching(*dup(e))
