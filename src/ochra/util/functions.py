import math
from scipy.special import erfinv
from ochra.plane import Point


def f2s(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def logit(x: float) -> float:
    eps = 1e-10
    x = min(1 - eps, max(eps, x))
    return math.log(x / (1 - x))


def probit(x: float) -> float:
    eps = 1e-10
    x = min(1 - eps, max(eps, x))
    return math.sqrt(2) * erfinv(2 * x - 1)


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def lerp_point(a: Point, b: Point, t: float) -> Point:
    return Point(lerp(a.x, b.x, t), lerp(a.y, b.y, t))


def dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def rad_to_deg(rad: float) -> float:
    return rad * 360 / math.tau


def deg_to_rad(deg: float) -> float:
    return deg * math.tau / 360
