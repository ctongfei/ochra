import math
from typing import Any, Iterable, TYPE_CHECKING

from ochra.plane import Point

if TYPE_CHECKING:
    from ochra.rect import AxisAlignedRectangle


def f2s(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    else:
        return str(x)


def logit(x: float) -> float:
    eps = 1e-10
    x = min(1 - eps, max(eps, x))
    return math.log(x / (1 - x))


def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b


def lerp_point(a: Point, b: Point, t: float) -> Point:
    return Point(lerp(a.x, b.x, t), lerp(a.y, b.y, t))


def dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def rad_to_deg(rad: float) -> float:
    return rad * 360 / math.tau


def deg_to_rad(deg: float) -> float:
    return deg * math.tau / 360


def aligned_bbox_from_points(ps: Iterable[Point]) -> 'AxisAlignedRectangle':
    from ochra.rect import AxisAlignedRectangle
    l = float('inf')
    u = -float('inf')
    r = -float('inf')
    b = float('inf')
    for p in ps:
        l = min(l, p.x)
        r = max(r, p.x)
        b = min(b, p.y)
        u = max(u, p.y)
    return AxisAlignedRectangle((l, b), (r, u))


def aligned_bbox_from_bboxes(bboxes: 'Iterable[AxisAlignedRectangle]') -> 'AxisAlignedRectangle':
    from ochra.rect import AxisAlignedRectangle
    l = float('inf')
    u = -float('inf')
    r = -float('inf')
    b = float('inf')
    for bbox in bboxes:
        l = min(l, bbox.bottom_left.x)
        r = max(r, bbox.top_right.x)
        b = min(b, bbox.bottom_left.y)
        u = max(u, bbox.top_right.y)
    return AxisAlignedRectangle((l, b), (r, u))
