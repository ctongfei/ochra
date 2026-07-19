import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Float

import ochra as ox
from ochra.core import InferredTransformMixin
from ochra.geometry import Elation, Point
from ochra.style import Color, Stroke
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


def test_projective_transform_uses_implicit_fallback():
    class TestImplicit(InferredTransformMixin, ox.Implicit):
        def __init__(self):
            self.styles = [Stroke(color=Color(1, 0, 0))]

        def implicit_func(self, p: Point) -> Float[jax.Array, ""]:
            return jnp.asarray(p.x**2 + p.y**2 - 1)

    implicit = TestImplicit()
    transformation = Elation((0.1, -0.05))
    transformed = implicit.transform(transformation)
    source_point = Point.mk((1, 0))
    transformed_point = transformation(source_point)

    assert isinstance(transformed, ox.ImplicitCurve)
    assert transformed.styles == implicit.styles
    assert transformed_point is not None
    assert jnp.isclose(
        transformed.implicit_func(transformed_point),
        implicit.implicit_func(source_point),
        atol=ox.Global.approx_eps,
    )
