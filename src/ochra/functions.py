from typing import Any, Iterable, TYPE_CHECKING, overload
import jax.numpy as jnp
import jax

from ochra.util import Global
from ochra.geometry import Point, Scalar, τ, PointSequenceI, PointSequence

if TYPE_CHECKING:
    from ochra.core import AxisAlignedRectangle


def ui2r(x: Scalar) -> Scalar:
    r"""A continuous monotonic function that maps $[0, 1] \to (-\infty, +\infty)$."""
    return jnp.tan((x - 0.5) * τ / 2)


def ui2pr(x: Scalar) -> Scalar:
    r"""A continuous monotonic function that maps $[0, 1] \to (0, +\infty)$."""
    return jnp.tan(x * τ / 2)


def r2ui(x: Scalar) -> Scalar:
    r"""$(-\infty, +\infty) \to [0, 1]$"""
    return 0.5 + jnp.arctan(x) * 2 / τ


def lerp(a: Scalar, b: Scalar, t: Scalar) -> Scalar:
    return (1.0 - t) * a + t * b


def lerp_point(a: Point, b: Point, t: Scalar) -> Point:
    return Point((1.0 - t) * a.loc + t * b.loc)


def dist(a: Point, b: Point) -> Scalar:
    return jnp.linalg.norm(a.loc - b.loc)


def rad_to_deg(rad: float) -> float:
    return rad * 360 / τ


def deg_to_rad(deg: float) -> float:
    return deg * τ / 360


def turn_to_rad(turn: float) -> float:
    return turn * τ


def solve_linear(a: Scalar, b: Scalar) -> list[Scalar]:
    """
    Solves the linear equation $ax + b = 0$.
    :return: a list of solutions, which may contain 0 or 1 elements.
    """
    if jnp.allclose(a, 0, atol=Global.approx_eps):
        return []
    return [-b / a]


def solve_quadratic(a: Scalar, b: Scalar, c: Scalar) -> list[Scalar]:
    """
    Solves the quadratic equation $ax^2 + bx + c = 0$.
    :return: a list of solutions, which may contain 0, 1, or 2 elements.
    """
    if jnp.allclose(a, 0, atol=Global.approx_eps):
        return solve_linear(b, c)
    d = b * b - 4 * a * c
    if jnp.allclose(d, 0, atol=Global.approx_eps):
        return [-b / (2 * a)]
    elif d < 0:
        return []
    else:
        sqrt_d = jnp.sqrt(d)
        return [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]


def aabb_from_points(ps: PointSequenceI) -> "AxisAlignedRectangle":
    """Computes the smallest axis-aligned bounding box that contains all the given points."""
    from ochra.core import AxisAlignedRectangle

    ps = PointSequence.mk(ps)
    if len(ps) == 0:
        return AxisAlignedRectangle((0, 0), (0, 0))
    l, b = jnp.min(ps.points, axis=0)
    r, u = jnp.max(ps.points, axis=0)
    return AxisAlignedRectangle((l, b), (r, u))


@overload
def aabb_from_boxes(bboxes: "Iterable[AxisAlignedRectangle]") -> "AxisAlignedRectangle": ...
@overload
def aabb_from_boxes(bboxes: "Iterable[AxisAlignedRectangle | None]") -> "AxisAlignedRectangle | None": ...


def aabb_from_boxes(bboxes: "Iterable[AxisAlignedRectangle | None]") -> "AxisAlignedRectangle | None":
    """Computes the smallest axis-aligned bounding box that contains all the given bounding boxes."""
    from ochra.core import AxisAlignedRectangle

    if any(bbox is None for bbox in bboxes):
        return None
    lbs = [bbox.bottom_left.loc for bbox in bboxes]
    rus = [bbox.top_right.loc for bbox in bboxes]
    if len(lbs) == 0 or len(rus) == 0:
        return None
    l, b = jnp.min(jnp.stack(lbs), axis=0)
    r, u = jnp.max(jnp.stack(rus), axis=0)
    return AxisAlignedRectangle((l, b), (r, u))
