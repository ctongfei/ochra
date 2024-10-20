from abc import ABC, abstractmethod
import bisect
from collections.abc import Callable, Collection, Iterator, Sequence
from functools import cached_property
from itertools import chain
from typing import Optional, TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ochra.geometry import *
from ochra.style import Stroke, Fill, Font, _text_extents, TextExtents
from ochra.functions import lerp, lerp_point, dist, aligned_bbox_from_points, \
    solve_quadratic

if TYPE_CHECKING:
    from ochra.mark import Marker, MarkerConfig


class Element(ABC):
    """
    Base class for all drawable elements.
    """

    def transform(self, f: Transformation) -> Self:
        """
        Transforms this element using the given transformation.
        Should be overridden by subclasses if possible.
        :param f: The given transformation.
        :return: A new element where every point is transformed.
        """
        return AnyTransformed(self, f)  # fallback

    def aabb(self) -> 'AxisAlignedRectangle | None':
        """
        Returns the axis-aligned bounding box of this element.
        None if the bounding box is not defined in the case that the element is infinite.
        """
        raise NotImplementedError(f"aabb() is not implemented for type {type(self)}.")

    def visual_center(self) -> Point:
        """
        Returns the visual center of this element.
        This is not necessarily the same as the geometric center.
        For example, for text.py, the visual center should be placed at the center of the
        x-height of the text.py, instead of the real height including ascenders and descenders.
        """
        return self.aabb().center

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        """
        Returns the visual bounding box of this element.
        This is not necessarily the same as the geometric bounding box.
        For example, for text.py, the visual bounding box should be placed at the baseline of the text.py.
        """
        return self.aabb()

    def translate(self, dx: float, dy: float) -> Self:
        return self.transform(translate((dx, dy)))

    def rotate(self, angle: float, anchor: PointI = Point.origin) -> Self:
        return self.transform(rotate(angle, Point.mk(anchor)))

    def scale(self, sx: float, sy: float) -> Self:
        return self.transform(scale(Vector.mk((sx, sy))))

    def reflect(self, axis: LineI) -> Self:
        return self.transform(reflect(axis))


class AnyTransformed(Element):
    def __init__(self, element: Element, transformation: Transformation):
        self.element = element
        self.transformation = transformation

    def transform(self, f: Transformation) -> Self:
        return AnyTransformed(self.element, f @ self.transformation)

    def aabb(self) -> 'Optional[AxisAlignedRectangle]':
        return self.element.aabb().transform(self.transformation).aabb()


class Group(Element):

    def __init__(self, elements: Collection[Element]):
        self.elements = elements

    def aabb(self) -> 'AxisAlignedRectangle':
        from ochra.functions import aligned_bbox_from_bboxes
        bboxes = [e.aabb() for e in self.elements]
        return aligned_bbox_from_bboxes(bboxes)

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        from ochra.functions import aligned_bbox_from_bboxes
        bboxes = [e.visual_bbox() for e in self.elements]
        return aligned_bbox_from_bboxes(bboxes)

    def recursive_children(self) -> Iterator[Element]:
        for e in self.elements:
            if isinstance(e, Group):
                yield from e.recursive_children()
            else:
                yield e

    def transform(self, f: Transformation) -> Self:
        new_elements = [e.transform(f) for e in self.elements]
        return Group(new_elements)


class Annotation(Element):
    """
    Annotations are special elements that do not scale or rotate by transformations.
    """

    def __init__(self,
                 anchor: PointI,
                 func: Callable[[Point], Element],
                 ):
        super().__init__()
        self.anchor = Point.mk(anchor)
        self.materialize_at = func

    def aabb(self) -> 'Optional[AxisAlignedRectangle]':
        return self.materialize().aabb()

    def materialize(self) -> Element:
        return self.materialize_at(self.anchor)

    def transform(self, f: Transformation) -> Self:
        return Annotation(f(self.anchor), self.materialize_at)


class Parametric(Element):
    """
    Represents any shape whose points can be parameterized by a single parameter `t` in [0, 1].
    """

    @property
    def pieces(self) -> Sequence[Tuple[float, float]]:
        """
        Returns the sequence of continuous pieces of the parameterization.
        Defaults to a single piece from 0 to 1, which says that the whole element is continuous.
        Should be overridden if not continuous, e.g. in the case of the two branches of a hyperbola.
        """
        return [(0.0, 1.0)]

    def at(self, t: Scalar) -> Point:
        """
        Returns the point at parameter `t`.
        :param t: A parameter in [0, 1].
        """
        raise NotImplementedError

    def tangent_vector_at(self, t: Scalar) -> Vector:
        """
        Returns the normalized tangent vector at parameter `t`.
        :param t: A parameter in [0, 1].
        """
        g = jax.jacobian(self.at)(t)
        return g.to_vector().normalize()

    def tangent_line_at(self, t: Scalar) -> "Line":
        """
        Returns the tangent line at parameter `t`.
        :param t: A parameter in [0, 1].
        """
        return Line.from_point_and_vector(self.at(t), self.tangent_vector_at(t))

    def approx_as_polyline(
            self,
            num_samples_per_piece: int = Global.num_first_order_steps,
            boundary_eps: float = Global.boundary_eps
    ) -> Element:
        """
        Approximates the element as a polyline, or a group of polylines if not continuous.
        """
        stroke = getattr(self, "stroke", None)
        pls = []
        for p in self.pieces:
            ts = jnp.linspace(p[0], p[1], num_samples_per_piece)
            ts = ts.at[0].set(p[0] + boundary_eps).at[-1].set(p[1] - boundary_eps)
            pl = Polyline([self.at(t) for t in ts], stroke=stroke)
            pls.append(pl)

        if len(pls) == 1:
            return pls[0]
        else:
            return Group(pls)

    def approx_as_quadratic_bezier_path(
            self,
            num_samples_per_piece: int = Global.num_second_order_steps,
            eps: float = Global.boundary_eps,
            **kwargs
    ) -> Element:
        """
        Approximates the element as a quadratic Bézier path.
        """
        qbps = []
        for p in self.pieces:
            n = num_samples_per_piece
            ts = jnp.linspace(p[0], p[1], n)
            ts = ts.at[0].set(p[0] + eps).at[-1].set(p[1] - eps)
            points = [self.at(t) for t in ts]
            tangents = [self.tangent_vector_at(t) for t in ts]
            control_points = [
                get_quadratic_bezier_curve_control_point_by_tangent(
                    points[i], tangents[i], points[i + 1], tangents[i + 1]
                )
                for i in range(n - 1)
            ]
            all_points = list(
                chain(*zip(points, control_points))
            ) + [points[-1]]
            qbp = QuadraticBezierPath(jnp.array([x.loc for x in all_points]), **kwargs)
            qbps.append(qbp)

        if len(qbps) == 1:
            return qbps[0]
        else:
            return Group(qbps)

    @staticmethod
    def from_func(func: Callable[[Scalar], Point], stroke: Stroke = Stroke()) -> 'Parametric':
        """
        Creates a parametric element from a function.
        """
        return ParametricFromFunction(func, stroke=stroke)

    @staticmethod
    def join(*shapes: 'Parametric') -> 'Parametric':
        """
        Joins multiple shapes into a single element.
        """
        return JoinedParametric(shapes)


class ParametricFromFunction(Parametric):
    """
    Represents a shape defined by a function.
    """

    def __init__(self, func: Callable[[Scalar], Point], stroke: Stroke = Stroke()):
        self.func = func
        self.stroke = stroke

    def at(self, t: Scalar) -> Point:
        return self.func(t)

    def transform(self, f: Transformation) -> Self:
        return ParametricFromFunction(lambda t: f(self.func(t)), stroke=self.stroke)


class JoinedParametric(Group, Parametric):
    def __new__(cls, shapes: Sequence[Parametric]):
        self = super().__new__(cls)
        self.shapes = shapes
        return self

    @property
    def pieces(self) -> Sequence[Tuple[float, float]]:
        n = len(self.shapes)
        return [(i / n, (i + 1) / n) for i in range(n)]

    def at(self, t: float) -> Point:
        n = len(self.shapes)
        if t == 1.0:
            return self.shapes[-1].at(1.0)
        else:
            i = int(t * n)
            t0 = t * n - i
            return self.shapes[i].at(t0)

    def transform(self, f: Transformation) -> Self:
        return JoinedParametric([s.transform(f) for s in self.shapes])


class FunctionGraph(Parametric):
    """
    A graph of a function over a specific interval.
    """

    def __init__(self,
                 f: Callable[[Scalar], Scalar],
                 lower_bound: Scalar,
                 upper_bound: Scalar,
                 stroke: Stroke = Stroke()
                 ):
        self.f = f
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stroke = stroke

    def at(self, t: Scalar):
        x = lerp(self.lower_bound, self.upper_bound, t)
        y = self.f(x)
        return Point.mk(x, y)


class Implicit(ABC):
    @abstractmethod
    def implicit_func(self, p: Point) -> jax.Array:
        """A function z = f(x, y) such that z = 0 defines the implicit curve."""
        raise NotImplementedError

    @cached_property
    def derivative(self):
        """Computes the derivative of the implicit function."""
        return jax.jit(jax.jacobian(self.implicit_func))

    def _normal_vector_at_point(self, p: Point):
        g = self.derivative(p).to_vector()
        return g.normalize()

    def _tangent_vector_at_point(self, p: Point):
        return self._normal_vector_at_point(p).rotate(math.tau / 4)

    def _get_point(self, p0: Point, eps: float = Global.approx_eps):
        dist = float('inf')
        while dist > eps:
            g = self.derivative(p0).to_vector()
            g_norm = g.norm()
            if g_norm == 0.0 or jnp.isnan(g_norm):
                p0 = p0 + Vector(jnp.array(np.random.normal(size=2)))
                continue
            p1 = p0 - g * (self.implicit_func(p0) / g_norm / g_norm)
            dist = (p1 - p0).norm()
            p0 = p1
        return p0

    def _trace_point(self, p0: Point, direction: Vector, step: float, eps: float = Global.approx_eps):
        p1 = p0 + direction * step
        return self._get_point(p1, eps)

    def trace_polyline(
            self,
            p0: Point = Point.origin,
            num_steps: int = Global.num_first_order_steps,
            step: float = 2,
            eps: float = Global.approx_eps,
            **kwargs
    ) -> Element:
        p0 = self._get_point(p0, eps)
        t = self._tangent_vector_at_point(p0)
        points = [p0]
        for _ in range(num_steps):
            p1 = self._trace_point(points[-1], t, step, eps)
            t = self._tangent_vector_at_point(p1)
            points.append(p1)
        return Polyline(points, **kwargs)

    def trace_quadratic_bezier_path(
            self,
            p0: Point = Point.origin,
            num_steps: int = Global.num_second_order_steps,
            step: float = Global.step_size,
            eps: float = Global.approx_eps,
            **kwargs
    ) -> Element:
        p0 = self._get_point(p0, eps)
        points = [p0]
        tangent_vectors = [self._tangent_vector_at_point(p0)]
        for _ in range(num_steps - 1):
            p1 = self._trace_point(points[-1], tangent_vectors[-1], step, eps)
            points.append(p1)
            t1 = self._tangent_vector_at_point(p1)
            tangent_vectors.append(t1)
        control_points = [
            get_quadratic_bezier_curve_control_point_by_tangent(
                points[i], tangent_vectors[i],
                points[i + 1], tangent_vectors[i + 1]
            )
            for i in range(num_steps - 1)
        ]
        qbp = QuadraticBezierPath(points, control_points, **kwargs)
        return qbp


class ImplicitCurve(Implicit):

    def __init__(self, func: Callable[[Point], Scalar]):
        self._func = func

    def implicit_func(self, p: Point) -> Scalar:
        return self._func(p)


class Canvas(Group):

    def __init__(self,
                 elements: Collection[Element],
                 viewport: Optional['AxisAlignedRectangle'] = None
                 ):
        super().__init__(elements)
        if viewport is None:
            viewport = self.visual_bbox()
        self.viewport = viewport


@dataclass(init=False)
class EmbeddedCanvas(Group):
    canvas: Canvas
    left_bottom: Point

    def __init__(self, canvas: Canvas, left_bottom: PointI):
        self.canvas = canvas
        self.left_bottom = Point.mk(left_bottom)
        super().__init__(elements=[
            canvas.translate(self.left_bottom.x, self.left_bottom.y)
        ])



class Line(Parametric, Implicit):

    def __init__(self, coef: LineI, stroke: Stroke = Stroke()):
        if isinstance(coef, tuple):
            coef = jnp.array(coef)
        self.coef = jnp.array(coef)  # [a, b, c], projective coefficients
        assert self.coef.shape == (3,)
        self.stroke = stroke

    @property
    def _a(self) -> float:
        return self.coef[0]

    @property
    def _b(self) -> float:
        return self.coef[1]

    @property
    def _c(self) -> float:
        return self.coef[2]

    @property
    def normal_vector(self):
        return Vector.mk(self._a, self._b)

    @property
    def direction_vector(self):
        return Vector.mk(self._b, -self._a)

    @property
    def angle(self) -> float:
        return -math.atan2(self._a, self._b)

    @property
    def slope(self) -> float:
        return -self._a / self._b

    @property
    def y_intercept(self) -> float:
        return +self._c / self._b

    def x_intercept(self) -> float:
        return +self._c / self._a

    def aabb(self) -> 'Optional[AxisAlignedRectangle]':
        return None

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        raise NotImplementedError

    def __contains__(self, p: Point):
        return jnp.isclose(self.implicit_func(p), 0.0, atol=Global.approx_eps)

    def implicit_func(self, p: Point) -> jax.Array:
        return jnp.dot(self.coef, p.to_proj_point().loc)

    def at(self, t: float):
        s = jsp.special.logit(t) * 1000  # maps (0, 1) to (-∞, +∞)
        h = jnp.hypot(self._a, self._b)
        p = -self._c / h
        θ = jnp.atan2(self._b, self._a)
        x = p * jnp.cos(θ) - s * jnp.sin(θ)
        y = p * jnp.sin(θ) + s * jnp.cos(θ)
        return Point.mk((x, y))

    def transform(self, f: Transformation) -> 'Line':
        v = f.inverse().matrix.T.dot(self.coef)
        return Line(v, stroke=self.stroke)

    def closest_to(self, p: PointI) -> Point:
        """
        Returns the point on the line that is closest to the given point.
        """
        a, b, c = self.coef
        p = Point.mk(p)
        x = (b * (b * p.x - a * p.y) - a * c) / (a ** 2 + b ** 2)
        y = (a * (-b * p.x + a * p.y) - b * c) / (a ** 2 + b ** 2)
        return Point.mk((x, y))

    def __str__(self):
        return f"{self._a:.2f}x + {self._b:.2f}y + {self._c:.2f} = 0"

    @classproperty
    def y_axis(cls):
        return HorizontalLine(0)

    @classproperty
    def x_axis(cls):
        return VerticalLine(0)

    @classmethod
    def from_two_points(cls, p0: PointI, p1: PointI, **kwargs):
        p0 = Point.mk(p0)
        p1 = Point.mk(p1)
        return cls(
            (
                p0.y - p1.y,
                p1.x - p0.x,
                p0.x * p1.y - p0.y * p1.x,
            ),
            **kwargs
        )

    @classmethod
    def from_point_and_vector(cls, p: PointI, v: VectorI, **kwargs):
        p = Point.mk(p)
        v = Vector.mk(v)
        return cls.from_two_points(p, p + v, **kwargs)

    @classmethod
    def from_intercepts(cls, x_intercept: float, y_intercept: float, **kwargs):
        return cls(
            (
                y_intercept,
                x_intercept,
                -x_intercept * y_intercept,
            ),
            **kwargs
        )

    @classmethod
    def from_slope_and_y_intercept(cls, slope: float, y_intercept: float, **kwargs):
        return cls(
            (
                slope,
                -1,
                y_intercept,
            ),
            **kwargs
        )


class HorizontalLine(Line):
    def __init__(self, y: float):
        super().__init__((0, 1, -y))

    @classproperty
    def x_axis(cls):
        return cls(0)


class VerticalLine(Line):
    def __init__(self, x: float):
        super().__init__((1, 0, -x))

    @classproperty
    def y_axis(cls):
        return cls(0)


class LineSegment(Parametric):

    def __init__(self,
                 p0: PointI,
                 p1: PointI,
                 stroke: Stroke = Stroke(),
                 marker_start: Optional['Marker'] = None,
                 marker_end: Optional['Marker'] = None,
                 ):
        p0 = Point.mk(p0)
        p1 = Point.mk(p1)
        if p0.x > p1.x or (p0.x == p1.x and p0.y > p1.y):
            p0, p1 = p1, p0
        # always make that p0 is the bottom-left-most point
        self.p0 = p0
        self.p1 = p1
        self.stroke = stroke
        self.marker_start = marker_start
        self.marker_end = marker_end

    @property
    def midpoint(self):
        return self.at(0.5)

    @property
    def center(self):
        return self.midpoint

    @property
    def slope(self):
        return (self.p1.y - self.p0.y) / (self.p1.x - self.p0.x)

    @property
    def angle(self):
        return math.atan2(self.p1.y - self.p0.y, self.p1.x - self.p0.x)

    @property
    def length(self):
        return dist(self.p0, self.p1)

    def extend_as_line(self) -> Line:
        return Line.from_two_points(self.p0, self.p1)

    def as_vector(self) -> Vector:
        return Vector(self.p1.x - self.p0.x, self.p1.y - self.p0.y)

    def at(self, t: float):
        return lerp_point(self.p0, self.p1, t)

    def __contains__(self, p: Point):
        return (p in self.extend_as_line()) and (
            min(self.p0.x, self.p1.x) <= p.x <= max(self.p0.x, self.p1.x) and
            min(self.p0.y, self.p1.y) <= p.y <= max(self.p0.y, self.p1.y)
        )

    def __eq__(self, other):
        return isinstance(other, LineSegment) and self.p0 == other.p0 and self.p1 == other.p1

    def aabb(self) -> 'AxisAlignedRectangle | None':
        return AxisAlignedRectangle(
            Point.mk(min(self.p0.x, self.p1.x), min(self.p0.y, self.p1.y)),
            Point.mk(max(self.p0.x, self.p1.x), max(self.p0.y, self.p1.y))
        )


    def transform(self, f: Transformation) -> 'LineSegment':
        return LineSegment(
            f(self.p0), f(self.p1),
            stroke=self.stroke,
            marker_start=self.marker_start,
            marker_end=self.marker_end
        )



class Polyline(Parametric):

    def __init__(self,
                 vertices: Sequence[PointI],
                 stroke: Stroke = Stroke(),
                 marker_start: Optional['Marker'] = None,
                 marker_mid: Optional['Marker'] = None,
                 marker_end: Optional['Marker'] = None,
                 ):
        self.vertices = [Point.mk(v) for v in vertices]
        self.stroke = stroke
        self.marker_start = marker_start
        self.marker_mid = marker_mid
        self.marker_end = marker_end
        for m in [marker_start, marker_mid, marker_end]:
            if m is not None:
                Marker.register_as_marker(m)

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def edges(self):
        return [
            LineSegment(self.vertices[i], self.vertices[i + 1])
            for i in range(self.num_vertices - 1)
        ]

    def aabb(self) -> 'AxisAlignedRectangle':
        from ochra.functions import aligned_bbox_from_points
        return aligned_bbox_from_points(self.vertices)

    def at(self, t: float):
        if t == 1.0:
            return self.vertices[-1]
        x = t * (self.num_vertices - 1)
        i = int(x)
        t0 = x - i
        return self.edges[i].at(t0)

    def transform(self, f: Transformation) -> 'Polyline':
        return Polyline(
            [f(v) for v in self.vertices],
            stroke=self.stroke,
            marker_start=self.marker_start,
            marker_mid=self.marker_mid,
            marker_end=self.marker_end
        )


class Polygon(Parametric):

    def __init__(self,
                 vertices: Sequence[PointI],
                 stroke: Stroke = Stroke(),
                 fill: Fill = Fill(),
                 marker: Optional['Marker'] = None,
                 ):
        self.vertices = [Point.mk(v) for v in vertices]
        self.stroke = stroke
        self.fill = fill
        self.marker = marker

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def edges(self):
        return [
            LineSegment(self.vertices[i], self.vertices[(i + 1) % self.num_vertices])
            for i in range(self.num_vertices)
        ]

    def aabb(self) -> 'AxisAlignedRectangle':
        from ochra.functions import aligned_bbox_from_points
        return aligned_bbox_from_points(self.vertices)

    def at(self, t: float):
        if t == 1.0:
            return self.vertices[0]
        x = t * self.num_vertices
        i = int(x)
        t0 = x - i
        return self.edges[i].at(t0)

    def transform(self, f: Transformation) -> 'Polygon':
        return Polygon([f(v) for v in self.vertices], self.stroke, self.fill, self.marker)

    @classmethod
    def regular(cls,
                n: int,
                *,
                circumradius: Optional[float] = None,
                side_length: Optional[float] = None,
                apothem: Optional[float] = None,
                **kwargs
                ):
        """
        Draws a regular convex polygon with n vertices and n edges.
        :param n: The number of vertices.
        :param circumradius: The radius of the circumcircle.
        :param side_length: Length of each side.
        :param apothem: Length of the center to an side edge.
        :param kwargs: Style parameters.
        :return:
        """
        circumradius = _get_circumradius(n, circumradius, side_length, apothem)
        return cls(
            [
                Point.mk(circumradius * math.cos(math.tau * i / n), circumradius * math.sin(math.tau * i / n))
                for i in range(n)
            ],
            **kwargs
        )

    @classmethod
    def regular_star(cls,
                     p: int,
                     q: int,
                     *,
                     circumradius: float,
                     **kwargs
                     ):
        """
        Draws a regular star polygon with Schläfli symbol {p/q}.
        :param p: The number of vertices.
        :param q: The step size.
        """
        def mk_part(j: int) -> "Polygon":
            return cls(
                [
                    Point(circumradius * math.cos(math.tau * i / p), circumradius * math.sin(math.tau * i / p))
                    for i in range(j, p * q, q)
                ],
                **kwargs
            )

        num_parts = math.gcd(p, q)
        if num_parts == 1:
            return mk_part(0)
        else:
            return JoinedParametric([mk_part(i) for i in range(num_parts)])


def _get_circumradius(
        n: int,
        circumradius: Optional[float],
        side_length: Optional[float],
        apothem: Optional[float]
):
    if circumradius is not None:
        return circumradius
    elif side_length is not None:
        return side_length / (2 * math.sin(math.tau / (2 * n)))
    elif apothem is not None:
        return apothem / math.cos(math.tau / (2 * n))
    else:
        raise ValueError("One of circumradius, side_length, or apothem must be provided.")



class Rectangle(Polygon, Parametric):

    def __init__(self, bottom_left: PointI, top_right: PointI, angle: float = 0.0, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        self.bottom_left = Point.mk(bottom_left)
        self.top_right = Point.mk(top_right)
        ray = Line.from_point_and_vector(self.bottom_left, Vector.unit(angle))
        self.bottom_right = ray.closest_to(top_right)
        self.top_left = self.top_right + (self.bottom_left - self.bottom_right)
        vertices = [self.bottom_left, self.bottom_right, self.top_right, self.top_left]
        self.angle = angle
        self.width = dist(self.bottom_left, self.bottom_right)
        self.height = dist(self.bottom_right, self.top_right)
        super().__init__(vertices, stroke, fill)

    @property
    def center(self):
        return lerp_point(self.bottom_left, self.top_right, 0.5)

    @property
    def top_center(self):
        return lerp_point(self.top_left, self.top_right, 0.5)

    @property
    def right_center(self):
        return lerp_point(self.bottom_right, self.top_right, 0.5)

    @property
    def bottom_center(self):
        return lerp_point(self.bottom_left, self.bottom_right, 0.5)

    @property
    def left_center(self):
        return lerp_point(self.bottom_left, self.top_left, 0.5)

    def aabb(self) -> 'AxisAlignedRectangle':
        l = min(p.x for p in self.vertices)
        r = max(p.x for p in self.vertices)
        b = min(p.y for p in self.vertices)
        u = max(p.y for p in self.vertices)
        return AxisAlignedRectangle(bottom_left=(l, b), top_right=(r, u))

    def translate(self, dx: float, dy: float) -> 'Rectangle':
        return Rectangle(
            self.bottom_left + Vector.mk((dx, dy)),
            self.top_right + Vector.mk((dx, dy)),
            self.angle,
            stroke=self.stroke,
            fill=self.fill
        )

    def rotate(self, angle: float, anchor: PointI = Point.origin) -> 'Rectangle':
        rot = rotate(angle, anchor)
        return Rectangle(
            rot(self.bottom_left),
            rot(self.top_right),
            self.angle + angle,
            stroke=self.stroke,
            fill=self.fill
        )


class AxisAlignedRectangle(Rectangle, Parametric):

    def __init__(self, bottom_left: PointI, top_right: PointI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        super().__init__(bottom_left, top_right, 0.0, stroke, fill)

    def aabb(self) -> 'AxisAlignedRectangle':
        return self

    def pad(self, dx: float, dy: float) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.translate(-dx, -dy),
            self.top_right.translate(dx, dy),
            stroke=self.stroke,
            fill=self.fill
        )

    def translate(self, dx: float, dy: float) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.translate(dx, dy),
            self.top_right.translate(dx, dy),
            stroke=self.stroke,
            fill=self.fill
        )

    def scale(self, sx: float, sy: float) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.scale(sx, sy),
            self.top_right.scale(sx, sy),
            stroke=self.stroke,
            fill=self.fill
        )

    def with_stroke(self, stroke: Stroke) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(self.bottom_left, self.top_right, stroke, self.fill)


class Conic(Parametric):
    """
    Represents any conic section in the plane.
    """
    def __init__(cls, coef: ConicI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        if isinstance(coef, tuple):
            a, b, c, d, e, f = coef
            coef = jnp.array([
                [a, b/2, d/2],
                [b/2, c, e/2],
                [d/2, e/2, f],
            ])
        assert coef.shape == (3, 3)
        cls.proj_matrix = (coef + coef.T) / 2  # symmetrize the quadratic form
        cls.stroke = stroke
        cls.fill = fill

    def materialize(self):
        d3 = jnp.linalg.det(self.proj_matrix)
        d2 = jnp.linalg.det(self.proj_matrix[:2, :2])
        if d3 == 0.0:  # degenerate
            if d2 < 0.0:
                # two intersecting lines, degenerate hyperbola
                pass
            elif d2 == 0:
                # two parallel lines, degenerate parabola
                pass
            else:
                # a dot, degenerate ellipse
                pass
        else:  # non-degenerate
            if d2 < 0.0:
                # hyperbola
                pass
            elif d2 == 0:
                # parabola
                pass
            else:
                # ellipse
                pass


class Ellipse(Conic, Parametric):

    def __init__(self, coef: ConicI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        super().__init__(coef, stroke=stroke, fill=fill)
        assert self.proj_matrix.shape == (3, 3)
        assert jnp.linalg.det(self.proj_matrix) != 0.0
        assert jnp.linalg.det(self.proj_matrix[:2, :2]) > 0.0

        self.center = Point(-jnp.linalg.inv(self.proj_matrix[:2, :2]) @ self.proj_matrix[:2, 2])
        self.angle = 0.5 * jnp.atan2(
            self.proj_matrix[1, 0] + self.proj_matrix[0, 1],
            self.proj_matrix[1, 1] - self.proj_matrix[0, 0]
        )
        if self.angle < 0.0:
            self.angle = -self.angle
        t = (translate(self.center.loc) @ rotate(self.angle)).matrix
        m = t.T @ self.proj_matrix @ t
        self.a = jnp.sqrt(-m[2, 2] / m[0, 0])
        self.b = jnp.sqrt(-m[2, 2] / m[1, 1])
        self.c = jnp.sqrt(self.a ** 2 - self.b ** 2)
        self.focus0 = self.center - Vector.unit(self.angle) * self.c
        self.focus1 = self.center + Vector.unit(self.angle) * self.c

    @classmethod
    def from_foci_and_major_axis(
            cls,
            focus0: PointI,
            focus1: PointI,
            major_axis: Scalar,
            stroke: Stroke = Stroke(),
            fill: Fill = Fill()
    ):
        """
        Defines an ellipse from its definition:
        the locus of points whose sum of distances (major axis) to two foci is constant.
        """
        focus0 = Point.mk(focus0)
        focus1 = Point.mk(focus1)
        center = lerp_point(focus0, focus1, 0.5)
        a = major_axis / 2
        c = dist(focus0, focus1) / 2
        b = jnp.sqrt(a ** 2 - c ** 2)
        θ = jnp.atan2(focus1.y - focus0.y, focus1.x - focus0.x)
        m0 = jnp.array([
            [b * b, 0, 0],
            [0, a * a, 0],
            [0, 0, -a * a * b * b],
        ])
        t = (translate(center.loc) @ rotate(θ)).inverse().matrix
        m = t.T @ m0 @ t
        return cls(m, stroke=stroke, fill=fill)

    @property
    def semi_major_axis(self):
        return self.a

    @property
    def semi_minor_axis(self):
        return self.b

    @property
    def eccentricity(self):
        return self.c / self.a

    @property
    def vertex0(self):
        return self.center - Vector.unit(self.angle) * self.a

    @property
    def vertex1(self):
        return self.center + Vector.unit(self.angle) * self.a

    @property
    def covertex0(self):
        return self.center - Vector.unit(self.angle + math.tau / 4) * self.b

    @property
    def covertex1(self):
        return self.center + Vector.unit(self.angle + math.tau / 4) * self.b

    def circumscribed_rectangle(self) -> Rectangle:
        pass

    def arc_between(self, start: float, end: float):
        return Arc(self, start, end)

    def __contains__(self, p: PointI):
        p = Point.mk(p)
        return dist(self.focus0, p) + dist(self.focus1, p) <= self.a * 2

    def at(self, t: float):
        θ = t * math.tau
        φ = self.angle
        x = self.center.x + self.a * jnp.cos(θ) * jnp.cos(φ) - self.b * jnp.sin(θ) * jnp.sin(φ)
        y = self.center.y + self.b * jnp.sin(θ) * jnp.cos(φ) + self.a * jnp.cos(θ) * jnp.sin(φ)
        return Point(jnp.array([x, y]))

    @classmethod
    def standard(cls, a: float, b: float, **kwargs):
        if a >= b:
            return cls((b * b, 0, a * a, 0, 0, -a*a*b*b), **kwargs)
        else:
            return cls((a * a, 0, b * b, 0, 0, -a*a*b*b), **kwargs)


class Circle(Ellipse, Parametric):

    def __init__(self, radius: float, center: PointI = (0, 0), stroke: Stroke = Stroke(), fill: Fill = Fill()):
        center = Point.mk(center)
        tr = translate(center.loc)
        std_matrix = jnp.diag(jnp.array([1, 1, -radius ** 2]))
        matrix = tr.inverse().matrix.T @ std_matrix @ tr.matrix
        super().__init__(matrix, stroke=stroke, fill=fill)
        self.center = center
        self.radius = radius

    def at(self, t: float):
        x, y = self.center.x, self.center.y
        θ = t * math.tau  # [0, 1] -> [0, τ]
        return Point(x + self.radius * math.cos(θ), y + self.radius * math.sin(θ))

    def __contains__(self, p: PointI):
        p = Point.mk(p)
        return dist(self.center, p) <= self.radius

    def aabb(self) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.center - Vector.mk((self.radius, self.radius)),
            self.center + Vector.mk((self.radius, self.radius))
        )

    def transform(self, f: Transformation) -> 'Circle':
        # TODO: wrong! transforms into an ellipse
        return Circle(self.radius, center=f(self.center), stroke=self.stroke, fill=self.fill)

    @classmethod
    def from_center_and_radius(cls, center: PointI, radius: float):
        center = Point.mk(center)
        return cls(center, radius)


class Arc(Parametric):
    def __init__(self, ellipse: Ellipse, start: float, end: float, stroke: Stroke = Stroke()):
        assert 0 <= start < end <= 1
        self.ellipse = ellipse
        self.start = start
        self.end = end
        self.stroke = stroke

    def at(self, t: float):
        return self.ellipse.at(lerp(self.start, self.end, t))



class QuadraticBezierCurve(Parametric):

    def __init__(self, mat: jax.Array, stroke: Stroke = Stroke()):
        assert mat.shape == (3, 2)
        self.mat = mat
        self.stroke = stroke

    @property
    def p0(self) -> Point:
        return Point(self.mat[0, :])

    @property
    def p1(self) -> Point:
        return Point(self.mat[1, :])

    @property
    def p2(self) -> Point:
        return Point(self.mat[2, :])

    def at(self, t: float):
        s = 1 - t
        v = jnp.array([s * s, 2 * s * t, t * t])
        return Point(self.mat.T @ v)

    def aabb(self) -> 'AxisAlignedRectangle | None':
        # TODO: control point not correct
        return aligned_bbox_from_points([self.p0, self.p1, self.p2])

    def transform(self, f: Transformation) -> 'QuadraticBezierCurve':
        return QuadraticBezierCurve(f.apply_batch(self.mat), stroke=self.stroke)

    @classmethod
    def from_points(cls, p0: PointI, p1: PointI, p2: PointI, **kwargs):
        return cls(jnp.stack([Point.mk(p).loc for p in [p0, p1, p2]]), **kwargs)


class QuadraticBezierPath(Parametric):
    def __init__(
            self,
            mat: jax.Array,
            stroke: Stroke = Stroke(),
            markers: 'MarkerConfig | None' = None,
    ):
        from ochra.mark import Marker
        m, n = mat.shape
        self.num_segments = (m - 1) // 2
        assert n == 2
        assert m >= 3 and m % 2 == 1
        self.mat = mat
        self.stroke = stroke
        self.markers = markers
        if self.markers:
            for m in [self.markers.start, self.markers.mid, self.markers.end]:
                if m is not None:
                    Marker.register_as_marker(m)

    def points(self) -> Sequence[Point]:
        return [Point(self.mat[i, :]) for i in range(0, self.mat.shape[0], 2)]

    def aabb(self) -> 'AxisAlignedRectangle | None':
        # TODO: not correct
        return aligned_bbox_from_points(self.points())

    def at(self, t: float):
        if t == 1.0:
            return Point(self.mat[-1, :])
        else:
            i = int(t * self.num_segments)
            t0 = t * self.num_segments - i
            qbc = QuadraticBezierCurve(self.mat[2 * i:2 * i + 3, :])
            return qbc.at(t0)

    def transform(self, f: Transformation) -> Self:
        return QuadraticBezierPath(f.apply_batch(self.mat), stroke=self.stroke, markers=self.markers)


class CubicBezierCurve(Parametric):

    def __init__(self, p0: PointI, p1: PointI, p2: PointI, p3: PointI, stroke: Stroke = Stroke()):
        self.p0 = Point.mk(p0)
        self.p1 = Point.mk(p1)
        self.p2 = Point.mk(p2)
        self.p3 = Point.mk(p3)
        self.stroke = stroke

    def at(self, t: float):
        f = lambda x, y: lerp_point(x, y, t)
        return f(
            f(
                f(self.p0, self.p1),
                f(self.p1, self.p2)
            ),
            f(
                f(self.p1, self.p2),
                f(self.p2, self.p3)
            )
        )

    def transform(self, f: Transformation) -> 'CubicBezierCurve':
        return CubicBezierCurve(f(self.p0), f(self.p1), f(self.p2), f(self.p3), stroke=self.stroke)





def intersect_interval_interval(i0: tuple[float, float], i1: tuple[float, float]) -> tuple[float, float] | float | None:
    """
    Intersect two intervals.
    """
    s0, t0 = i0
    if s0 > t0:
        s0, t0 = t0, s0
    s1, t1 = i1
    if s1 > t1:
        s1, t1 = t1, s1
    if t0 < s1 or t1 < s0:
        return None
    else:
        l = max(s0, s1)
        r = min(t0, t1)
        if l == r:
            return r
        else:
            return l, r


def intersect_line_line(l0: Line, l1: Line) -> Point | Line | None:
    """
    Intersect two lines.
    :return: The intersection point, the intersection line, or None if the lines are parallel.
    """
    pp = jnp.cross(l0.coef, l1.coef)  # projective point
    if jnp.all(pp == 0):  # coincident lines
        return l0
    elif pp[2] == 0:  # parallel lines
        return None
    else:
        return ProjPoint(pp).to_point()


def intersect_segment_segment(s0: LineSegment, s1: LineSegment) -> Point | LineSegment | None:
    """
    Intersect two line segments.
    """
    l0 = s0.extend_as_line()
    l1 = s1.extend_as_line()
    x = intersect_line_line(l0, l1)
    if isinstance(x, Point):
        if x in s0 and x in s1:
            return x
        else:
            return None
    elif isinstance(x, Line):
        a = intersect_interval_interval((s0.p0.x, s0.p1.x), (s1.p0.x, s1.p1.x))
        b = intersect_interval_interval((s0.p0.y, s0.p1.y), (s1.p0.y, s1.p1.y))
        if isinstance(a, tuple) and isinstance(b, tuple):
            return LineSegment(Point.mk(a[0], b[0]), Point.mk(a[1], b[1]))
        elif isinstance(a, float) and isinstance(b, float):
            return Point.mk(a, b)
    else:
        return None


def intersect_line_segment(l: Line, s: LineSegment) -> Point | LineSegment | None:
    """
    Intersect a line and a line segment.
    """
    x = intersect_line_line(l, s.extend_as_line())
    if isinstance(x, Point):
        if x in s:
            return x
        else:
            return None
    elif isinstance(x, Line):
        return s  # line overlaps with segment


def intersect_line_conic(l: Line, c: Conic) -> tuple[Point, Point] | Point | None:
    """
    Intersect a line and a conic.
    """
    A = c.proj_matrix
    x0 = l.at(1/2).to_proj_point().loc
    v = jnp.concat([l.direction_vector.vec, jnp.array([0])])
    ts = solve_quadratic(
        v.T @ A @ v,
        2 * x0.T @ A @ v,
        x0.T @ A @ x0
    )
    if ts is None:
        return None
    if isinstance(ts, Scalar):
        return ProjPoint(x0 + ts * v).to_point()
    else:
        t0, t1 = ts
        p0 = ProjPoint(x0 + t0 * v).to_point()
        p1 = ProjPoint(x0 + t1 * v).to_point()
        return p0, p1


def intersect_line_aabb(l: Line, aabb: AxisAlignedRectangle) -> Point | list[Point] | LineSegment | None:
    scores = [l.implicit_func(v).item() for v in aabb.vertices]
    num_ge_0 = sum(s >= 0 for s in scores)
    if num_ge_0 == 0:
        return None  # no intersection
    else:
        ps = [intersect_line_segment(l, s) for s in aabb.edges]
        return [p for p in ps if p is not None]
    raise ValueError("Should never happen.")


def get_quadratic_bezier_curve_control_point_by_tangent(
        p0: Point, t0: Vector,
        p1: Point, t1: Vector,
) -> Point:
    """
    Given two points and their tangent vectors,
    returns the control point for a quadratic Bézier curve connecting them.
    """
    l0 = Line.from_point_and_vector(p0, t0)
    l1 = Line.from_point_and_vector(p1, t1)
    x = intersect_line_line(l0, l1)
    if isinstance(x, Point):
        return x
    else:
        return lerp_point(p0, p1, 0.5)
