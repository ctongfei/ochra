import ochra as ox
import jax.numpy as jnp



def dup[E](e: E) -> tuple[E, E]:
    return e, e


def check_implicit_parametric_matching(impl: ox.Implicit, par: ox.Parametric):
    points = par.approx_as_polyline().points.points
    imp_func_vals = impl._raw_implicit_func_batched(points)
    max_error = jnp.max(jnp.abs(imp_func_vals))
    print(f"Max error = {max_error} for {impl}")
    assert jnp.isclose(0, max_error, atol=ox.Global.approx_eps)


def test_line():
    check_implicit_parametric_matching(*dup(ox.Line.from_two_points((0, 0), (1, 1))))

def test_circle():
    check_implicit_parametric_matching(*dup(ox.Circle(10, (0, 0))))
