from typing import Any, Iterable, TYPE_CHECKING
import jax.numpy as jnp
import jax

from ochra.util import Global
from ochra.geometry import Point, Scalar, τ

if TYPE_CHECKING:
    from ochra.core import AxisAlignedRectangle


def f2s(x: Any) -> str:
    if isinstance(x, jax.Array):
        return f2s(x.item())
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    else:
        return str(x)


def ui2r(x: Scalar) -> Scalar:
    return jnp.tan((x - 0.5) * τ / 2)


def ui2pr(x: Scalar) -> Scalar:
    return jnp.tan(x * τ / 2)


def r2ui(x: Scalar) -> Scalar:
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
    Solves the linear equation ax + b = 0.
    :return: a list of solutions, which can contain 0 or 1 elements.
    """
    if jnp.allclose(a, 0, atol=Global.approx_eps):
        return []
    return [-b / a]


def solve_quadratic(a: Scalar, b: Scalar, c: Scalar) -> list[Scalar]:
    """
    Solves the quadratic equation ax^2 + bx + c = 0.
    :return: a list of solutions, which can contain 0, 1, or 2 elements.
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


def aligned_bbox_from_points(ps: Iterable[Point]) -> 'AxisAlignedRectangle':
    from ochra.core import AxisAlignedRectangle
    vecs = [p.loc for p in ps]
    if len(vecs) == 0:
        return AxisAlignedRectangle((0, 0), (0, 0))
    all_vecs = jnp.stack(vecs)
    l, b = jnp.min(all_vecs, axis=0)
    r, u = jnp.max(all_vecs, axis=0)
    return AxisAlignedRectangle((l, b), (r, u))


def aligned_bbox_from_bboxes(bboxes: 'Iterable[AxisAlignedRectangle]') -> 'AxisAlignedRectangle':
    from ochra.core import AxisAlignedRectangle
    lbs = [bbox.bottom_left.loc for bbox in bboxes]
    rus = [bbox.top_right.loc for bbox in bboxes]
    l, b = jnp.min(jnp.stack(lbs), axis=0)
    r, u = jnp.max(jnp.stack(rus), axis=0)
    return AxisAlignedRectangle((l, b), (r, u))
