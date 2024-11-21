from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
import math
from typing import Optional, TYPE_CHECKING, Self, cast, override

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ochra.util import Global, classproperty
from ochra.geometry import (
    τ,
    Scalar,
    Point,
    PointI,
    ProjPoint,
    Vector,
    VectorI,
    LineI,
    Transformation,
    Translation,
    Rotation,
    Scaling,
    Reflection,
    ConicI,
    ShearX,
    ShearY
)
from ochra.style import Stroke, Fill
from ochra.functions import lerp, lerp_point, dist, aligned_bbox_from_points, aligned_bbox_from_bboxes, \
    solve_quadratic, solve_linear

if TYPE_CHECKING:
    from ochra.mark import Marker, MarkerConfig


class Element(ABC):
    """
    Base class for all drawable elements.
    """

    def transform(self, f: Transformation) -> 'Element':
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
        Should always be overridden by subclasses.
        None if the bounding box is not defined in the case that the element is infinite.
        """
        raise NotImplementedError(f"aabb() is not implemented for type {type(self)}.")

    def visual_center(self) -> Point:
        """
        Returns the visual center of this element.
        This is not necessarily the same as the geometric center.
        For example, for text, the visual center should be placed at the center of the
        x-height of the text, instead of the real height including ascenders and descenders.
        """
        aabb = self.aabb()
        assert aabb is not None
        return aabb.center

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        """
        Returns the visual bounding box of this element.
        This is not necessarily the same as the geometric bounding box.
        For example, for text, the visual bounding box should be placed at the baseline of the text.
        """
        aabb = self.aabb()
        assert aabb is not None
        return aabb

    def translate(self, dx: float, dy: float) -> 'Element':
        return self.transform(Translation((dx, dy)))

    def rotate(self, angle: float, anchor: PointI = Point.origin) -> 'Element':
        return self.transform(Rotation.centered(angle, Point.mk(anchor)))

    def scale(self, sx: float, sy: float, anchor: PointI = Point.origin) -> 'Element':
        return self.transform(Scaling.centered((sx, sy), Point.mk(anchor)))

    def shear_x(self, m: Scalar) -> 'Element':
        return self.transform(ShearX(m))

    def shear_y(self, m: Scalar) -> 'Element':
        return self.transform(ShearY(m))

    def reflect(self, axis: LineI) -> 'Element':
        return self.transform(Reflection(axis))


class AnyTransformed(Element):
    """
    Fallback class for transformed elements.
    At rendering time, objects of this class will be rendered by the SVG transform attribute.
    """

    def __init__(self, element: Element, transformation: Transformation):
        self.element = element
        self.transformation = transformation

    def transform(self, f: Transformation) -> 'AnyTransformed':
        return AnyTransformed(self.element, f @ self.transformation)

    def aabb(self) -> 'AxisAlignedRectangle | None':
        old_bbox = self.element.aabb()
        if old_bbox is None:
            return None
        return old_bbox.transform(self.transformation).aabb()


class Group(Element):
    """
    Represents a group of elements.
    """

    def __init__(self, elements: Collection[Element]):
        self.elements = elements

    def aabb(self) -> 'AxisAlignedRectangle':
        bboxes = [e.aabb() for e in self.elements]
        return aligned_bbox_from_bboxes(bboxes)

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        bboxes = [e.visual_bbox() for e in self.elements]
        return aligned_bbox_from_bboxes(bboxes)

    def recursive_children(self) -> Iterator[Element]:
        for e in self.elements:
            if isinstance(e, Group):
                yield from e.recursive_children()
            else:
                yield e

    def transform(self, f: Transformation) -> 'Group':
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

    def transform(self, f: Transformation) -> 'Annotation':
        return Annotation(f(self.anchor), self.materialize_at)


class Parametric(Element):
    """
    Represents any shape whose points can be parameterized by a single parameter `t` in [0, 1].
    """

    stroke: Stroke

    @property
    def pieces(self) -> Sequence[tuple[float, float]]:
        """
        Returns the sequence of continuous pieces of the parameterization.
        Defaults to a single piece from 0 to 1, which says that the whole element is continuous.
        Should be overridden if not continuous, e.g. in the case of the two branches of a hyperbola.
        """
        return [(0.0, 1.0)]

    @cached_property
    def derivative(self):
        """
        Computes the derivative of the parametric function.
        """
        return jax.jit(jax.jacfwd(self.at))  # cache this

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
        g = self.derivative(t)
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
            pl = Polyline([self.at(t) for t in ts], stroke=stroke or Stroke())
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

    def transform(self, f: Transformation) -> 'Parametric':
        return ParametricFromFunction(lambda t: cast(Point, f(self.at(t))), stroke=self.stroke)

    def slice(self, t0: Scalar, t1: Scalar) -> 'Parametric':
        return ParametricSlice(self, t0, t1, stroke=self.stroke)

    @staticmethod
    def from_func(func: Callable[[Scalar], Point], stroke: Stroke = Stroke()) -> 'Parametric':
        """
        Creates a parametric element from a function.
        """
        return ParametricFromFunction(func, stroke=stroke)

    @staticmethod
    def join(*shapes: 'Parametric', stroke: Stroke = Stroke()) -> 'Parametric':
        """
        Joins multiple shapes into a single element.
        """
        return JoinedParametric(shapes, stroke=stroke)


class ParametricSlice(Parametric):
    def __init__(self, outer: Parametric, a: Scalar, b: Scalar, stroke: Stroke = Stroke()):
        self.outer = outer
        self.a = a
        self.b = b
        self.stroke = stroke

    def at(self, t: Scalar):
        return self.outer.at(lerp(self.a, self.b, t))

    @property
    def pieces(self) -> Sequence[tuple[float, float]]:
        ps = []
        for p in self.outer.pieces:
            i = intersect_interval_interval(p, (self.a, self.b))
            if isinstance(i, tuple):
                ps.append(i)
        t = lambda x: (x - self.a) / (self.b - self.a)
        return [(t(p[0]), t(p[1])) for p in ps]


class ParametricFromFunction(Parametric):
    """
    Represents a shape defined by a function.
    """

    def __init__(self, func: Callable[[Scalar], Point], stroke: Stroke = Stroke()):
        self.func = func
        self.stroke = stroke

    def at(self, t: Scalar) -> Point:
        return self.func(t)

    def transform(self, f: Transformation) -> Parametric:
        return ParametricFromFunction(lambda t: f(self.func(t)), stroke=self.stroke)


class JoinedParametric(Group, Parametric):
    def __init__(self, shapes: Sequence[Parametric], stroke: Stroke = Stroke()):
        super().__init__(shapes)
        self.shapes = shapes
        self.stroke = stroke

    @property
    def pieces(self) -> Sequence[tuple[float, float]]:
        n = len(self.shapes)
        return [(i / n, (i + 1) / n) for i in range(n)]

    def at(self, t: Scalar) -> Point:
        n = len(self.shapes)
        if t == 1.0:
            return self.shapes[-1].at(1.0)
        else:
            i = int(t * n)
            t0 = t * n - i
            return self.shapes[i].at(t0)

    def transform(self, f: Transformation) -> 'JoinedParametric':
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
        return self._normal_vector_at_point(p).rotate(τ / 4)

    def _get_point(self, p0: Point, eps: float = Global.approx_eps):
        dist = float('inf')
        while dist > eps:
            g = self.derivative(p0).to_vector()
            g_norm = g.norm()
            if g_norm == 0.0 or jnp.isnan(g_norm):
                p0 = p0 + Vector(jnp.array(np.random.normal(size=2)))
                continue
            p1 = p0 + g * (-self.implicit_func(p0) / g_norm / g_norm)
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
            step: float = Global.first_order_step_size,
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
            step: float = Global.second_order_step_size,
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
        qbp = QuadraticBezierPath.from_points(points, control_points, **kwargs)
        return qbp


class ImplicitCurve(Implicit):

    def __init__(self, func: Callable[[Point], jax.Array]):
        self._func = func

    def implicit_func(self, p: Point) -> jax.Array:
        return self._func(p)


class Line(Parametric, Implicit):

    def __init__(self, coef: LineI, stroke: Stroke = Stroke()):
        if isinstance(coef, tuple):
            coef = jnp.array(coef)
        self.coef = jnp.array(coef)  # [a, b, c], projective coefficients
        assert self.coef.shape == (3,)
        self.stroke = stroke

    @property
    def _a(self) -> Scalar:
        return self.coef[0]

    @property
    def _b(self) -> Scalar:
        return self.coef[1]

    @property
    def _c(self) -> Scalar:
        return self.coef[2]

    @property
    def normal_vector(self):
        return Vector.mk(self._a, self._b)

    @property
    def direction_vector(self):
        return Vector.mk(self._b, -self._a)

    @property
    def angle(self) -> Scalar:
        return -jnp.atan2(self._a, self._b)

    @property
    def slope(self) -> Scalar:
        return -self._a / self._b

    @property
    def y_intercept(self) -> Scalar:
        return +self._c / self._b

    def x_intercept(self) -> Scalar:
        return +self._c / self._a

    def aabb(self) -> 'Optional[AxisAlignedRectangle]':
        return None

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        raise NotImplementedError

    def __contains__(self, p: Point):
        return jnp.isclose(self.implicit_func(p), 0.0, atol=Global.approx_eps)

    def implicit_func(self, p: Point) -> jax.Array:
        return jnp.dot(self.coef, p.to_proj_point().loc)

    def at(self, t: Scalar) -> Point:
        o = Point.origin
        p0 = self.closest_to(o)
        d = dist(p0, o)
        t = (t - 0.5) * τ / 2  # maps (0, 1) to (-τ/4, τ/4)
        return p0 + (Vector.unit(self.angle) * (d * jnp.tan(t)))

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
        return f"Line({self._a:.4f}x + {self._b:.4f}y + {self._c:.4f} = 0)"

    @classmethod
    def y_axis(cls):
        return cls((1, 0, 0))

    @classmethod
    def x_axis(cls):
        return cls((0, 1, 0))

    @classmethod
    def from_two_points(cls, p0: PointI, p1: PointI, **kwargs):
        pp0 = Point.mk(p0).to_proj_point()
        pp1 = Point.mk(p1).to_proj_point()
        return cls(
            jnp.cross(pp0.loc, pp1.loc),
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
        return jnp.atan2(self.p1.y - self.p0.y, self.p1.x - self.p0.x)

    @property
    def length(self):
        return dist(self.p0, self.p1)

    def extend_as_line(self) -> Line:
        return Line.from_two_points(self.p0, self.p1)

    def as_vector(self) -> Vector:
        return Vector(self.p1.x - self.p0.x, self.p1.y - self.p0.y)

    def at(self, t: Scalar):
        return lerp_point(self.p0, self.p1, t)

    def __contains__(self, p: Point):
        return (p in self.extend_as_line()) and (
            min(self.p0.x, self.p1.x) <= p.x <= max(self.p0.x, self.p1.x) and
            min(self.p0.y, self.p1.y) <= p.y <= max(self.p0.y, self.p1.y)
        )

    def __str__(self):
        return f"LineSegment({self.p0}, {self.p1})"

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

    def at(self, t: Scalar):
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
                     circumradius: float = None,
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
                    Point.mk(circumradius * math.cos(math.tau * i / p), circumradius * math.sin(math.tau * i / p))
                    for i in range(j, p * q, q)
                ],
                **kwargs
            )
        num_parts = math.gcd(p, q)
        if num_parts == 1:
            return mk_part(0)
        else:
            return JoinedParametric([mk_part(i) for i in range(num_parts)])


class Rectangle(Polygon, Parametric):

    def __init__(self, bottom_left: PointI, top_right: PointI, angle: Scalar = 0.0, stroke: Stroke = Stroke(), fill: Fill = Fill()):
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

    def translate(self, dx: Scalar, dy: Scalar) -> 'Rectangle':
        return Rectangle(
            self.bottom_left + Vector.mk(dx, dy),
            self.top_right + Vector.mk(dx, dy),
            self.angle,
            stroke=self.stroke,
            fill=self.fill
        )

    def rotate(self, angle: Scalar, anchor: PointI = Point.origin) -> 'Rectangle':
        rot = Rotation.centered(angle, anchor)
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

    def pad(self, dx: Scalar, dy: Scalar) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.translate(-dx, -dy),
            self.top_right.translate(dx, dy),
            stroke=self.stroke,
            fill=self.fill
        )

    def translate(self, dx: Scalar, dy: Scalar) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.translate(dx, dy),
            self.top_right.translate(dx, dy),
            stroke=self.stroke,
            fill=self.fill
        )

    def scale(self, sx: Scalar, sy: Scalar, anchor: PointI = Point.origin) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.scale(sx, sy),  # TODO: wrong
            self.top_right.scale(sx, sy),
            stroke=self.stroke,
            fill=self.fill
        )

    def with_stroke(self, stroke: Stroke) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(self.bottom_left, self.top_right, stroke, self.fill)


class Conic(Implicit, Parametric):
    """
    Represents any conic section in the plane.
    """
    def __new__(cls, *args, **kwargs):
        if cls is Conic:
            raise TypeError("Conic cannot be instantiated directly.")
        return super().__new__(cls)

    def __init__(self, coef: ConicI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        if isinstance(coef, tuple):
            a, b, c, d, e, f = coef
            coef = jnp.array([
                [a, b / 2, d / 2],
                [b / 2, c, e / 2],
                [d / 2, e / 2, f],
            ])
        assert coef.shape == (3, 3)
        self.proj_matrix = (coef + coef.T) / 2  # symmetrize the quadratic form
        self.aff_matrix = self.proj_matrix[:2, :2]
        self.stroke = stroke
        self.fill = fill

    def implicit_func(self, p: Point) -> jax.Array:
        pp = p.to_proj_point().loc
        return pp.T @ self.proj_matrix @ pp

    @classmethod
    def create(cls, coef: ConicI, stroke: Stroke = Stroke(), fill: Fill = Fill()) -> 'Conic':
        if isinstance(coef, tuple):
            a, b, c, d, e, f = coef
            coef = jnp.array([
                [a, b / 2, d / 2],
                [b / 2, c, e / 2],
                [d / 2, e / 2, f],
            ])
        proj_matrix = (coef + coef.T) / 2  # symmetrize the quadratic form
        aff_matrix = proj_matrix[:2, :2]
        d3 = jnp.linalg.det(proj_matrix)
        d2 = jnp.linalg.det(aff_matrix)
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
            if jnp.isclose(d2, 0.0):
                return Parabola(coef, stroke, fill)
            elif d2 < 0.0:
                pass  # hyperbola
            else:
                return Ellipse(coef, stroke, fill)

    def transform(self, f: Transformation) -> 'Conic':
        t = f.inverse().matrix
        return Conic.create(t.T @ self.proj_matrix @ t, stroke=self.stroke, fill=self.fill)

    @classmethod
    def from_focus_directrix_eccentricity(cls, focus: PointI, directrix: Line, eccentricity: float, **kwargs):
        """
        Defines a conic section from its focus, directrix, and eccentricity.
        """
        a, b, c = directrix.coef
        e = eccentricity
        p, q = Point.mk(focus).loc
        A = (a * a + b * b) - e * e * a * a
        B = -2 * e * e * a * b
        C = (b * b + a * a) - e * e * b * b
        D = -2 * p * (a * a + b * b) - 2 * e * e * a * c
        E = -2 * q * (b * b + a * a) - 2 * e * e * b * c
        F = (p * p + q * q) * (a * a + b * b) + 2 * e * e * c * c
        return cls.create((A, B, C, D, E, F), **kwargs)


class Ellipse(Conic, Parametric):

    def __init__(self, coef: ConicI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        super().__init__(coef, stroke=stroke, fill=fill)
        assert self.proj_matrix.shape == (3, 3)
        assert jnp.linalg.det(self.proj_matrix) != 0.0
        d = jnp.linalg.det(self.aff_matrix)
        assert d > 0.0, "This is not an ellipse."
        eigvals, eigvecs = jnp.linalg.eigh(self.aff_matrix)
        eigvals = jnp.real(eigvals)
        eigvecs = jnp.real(eigvecs)
        l0, l1 = eigvals[0], eigvals[1]
        self.a = jnp.sqrt(d / l0)
        self.b = jnp.sqrt(d / l1)
        self.c = jnp.sqrt(d / l0 - d / l1)
        self.center = Point.mk(-jnp.linalg.inv(self.aff_matrix) @ self.proj_matrix[:2, 2])
        self.angle = jnp.atan2(eigvecs[1, 0], eigvecs[0, 0])
        self.focus0: Point = self.center + (-Vector.unit(self.angle) * self.c)
        self.focus1: Point = self.center + Vector.unit(self.angle) * self.c

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
        t = (Translation(center.loc) @ Rotation(θ)).inverse().matrix
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
    def vertex0(self) -> Point:
        return self.center + (-Vector.unit(self.angle) * self.a)

    @property
    def vertex1(self) -> Point:
        return self.center + Vector.unit(self.angle) * self.a

    @property
    def covertex0(self) -> Point:
        return self.center + (-Vector.unit(self.angle + τ / 4) * self.b)

    @property
    def covertex1(self) -> Point:
        return self.center + Vector.unit(self.angle + τ / 4) * self.b

    def circumscribed_rectangle(self) -> Rectangle:
        bottom_left: Point = self.vertex0 + (self.covertex0 - self.center)
        top_right: Point = self.vertex1 + (self.covertex1 - self.center)
        return Rectangle(bottom_left, top_right, self.angle)

    def directrix0(self) -> Line:
        minor_axis = Line.from_two_points(self.center, self.covertex0)
        d = (self.vertex0 - self.center) * (1 + (self.a - self.c) / self.a * self.eccentricity)
        return minor_axis.translate(d.x, d.y)

    def directrix1(self) -> Line:
        minor_axis = Line.from_two_points(self.center, self.covertex1)
        d = (self.vertex1 - self.center) * (1 + (self.a - self.c) / self.a * self.eccentricity)
        return minor_axis.translate(d.x, d.y)

    def arc_between(self, start: float, end: float):
        return Arc(self, start, end)

    def aabb(self) -> 'AxisAlignedRectangle | None':
        return self.circumscribed_rectangle().aabb()

    def __contains__(self, p: PointI) -> bool:
        p = Point.mk(p)
        return float(dist(self.focus0, p) + dist(self.focus1, p)) <= float(self.a * 2)

    def __str__(self):
        return f"Ellipse(f0 = {self.focus0}, f1 = {self.focus1}, a = {self.a}, angle = {self.angle})"

    def at(self, t: Scalar):
        θ = t * τ
        φ = self.angle
        x = self.center.x + self.a * jnp.cos(θ) * jnp.cos(φ) - self.b * jnp.sin(θ) * jnp.sin(φ)
        y = self.center.y + self.b * jnp.sin(θ) * jnp.cos(φ) + self.a * jnp.cos(θ) * jnp.sin(φ)
        return Point(jnp.array([x, y]))

    @classmethod
    def standard(cls, a: float, b: float, **kwargs):
        assert a >= b
        return cls((b * b, 0, a * a, 0, 0, -a*a*b*b), **kwargs)


class Circle(Ellipse, Parametric):

    def __init__(self, radius: float, center: PointI = (0, 0), stroke: Stroke = Stroke(), fill: Fill = Fill()):
        center = Point.mk(center)
        tr = Translation(center.loc)
        std_matrix = jnp.diag(jnp.array([1, 1, -radius ** 2]))
        matrix = tr.inverse().matrix.T @ std_matrix @ tr.matrix
        super().__init__(matrix, stroke=stroke, fill=fill)
        self.center = center
        self.radius = radius

    def at(self, t: float):
        x, y = self.center.x, self.center.y
        θ = t * τ  # [0, 1] -> [0, τ]
        return Point(x + self.radius * math.cos(θ), y + self.radius * math.sin(θ))

    def __contains__(self, p: PointI) -> bool:
        p = Point.mk(p)
        return float(dist(self.center, p)) <= float(self.radius)

    def aabb(self) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.center + (-Vector.mk((self.radius, self.radius))),
            self.center + Vector.mk((self.radius, self.radius))
        )

    def transform(self, f: Transformation) -> 'Ellipse':
        # TODO: wrong! transforms into an ellipse
        return Circle(self.radius, center=f(self.center), stroke=self.stroke, fill=self.fill)

    @classmethod
    def from_center_and_radius(cls, center: PointI, radius: float, **kwargs):
        center = Point.mk(center)
        return cls(radius, center, **kwargs)


class Arc(Parametric):
    def __init__(self, ellipse: Ellipse, start: float, end: float, stroke: Stroke = Stroke()):
        assert 0 <= start < end <= 1
        self.ellipse = ellipse
        self.start = start
        self.end = end
        self.stroke = stroke

    def at(self, t: Scalar):
        return self.ellipse.at(lerp(self.start, self.end, t))


class HyperbolaBranch(Conic, Parametric):
    pass


class Parabola(Conic, Parametric):

    def slice(self, t0: Scalar, t1: Scalar) -> 'QuadraticBezierCurve':
        p0 = self.at(t0)
        p1 = self.at(t1)
        return QuadraticBezierCurve.from_points(
            p0,
            get_quadratic_bezier_curve_control_point_by_tangent(
                p0, self.tangent_vector_at(t0),
                p1, self.tangent_vector_at(t1)
            ),
            p1
        )

    @classmethod
    def from_focus_and_directrix(cls, focus: PointI, directrix: Line, **kwargs):
        return Conic.from_focus_directrix_eccentricity(focus, directrix, 1.0, **kwargs)


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

    def at(self, t: Scalar):
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

    def bezier_segments(self) -> Sequence[QuadraticBezierCurve]:
        return [
            QuadraticBezierCurve(self.mat[i:i + 3, :])
            for i in range(0, self.mat.shape[0], 2)
        ]

    def aabb(self) -> 'AxisAlignedRectangle | None':
        # TODO: not correct
        return al

    def at(self, t: float):
        if t == 1.0:
            return Point(self.mat[-1, :])
        else:
            i = int(t * self.num_segments)
            t0 = t * self.num_segments - i
            qbc = QuadraticBezierCurve(self.mat[2 * i:2 * i + 3, :])
            return qbc.at(t0)

    @classmethod
    def from_points(cls, points: Sequence[PointI], control_points: Sequence[PointI], **kwargs):
        assert len(points) == len(control_points) + 1
        arr: list[Point] = [Point.origin for _ in range(2 * len(points) - 1)]
        arr[::2] = [Point.mk(p) for p in points]
        arr[1::2] = [Point.mk(p) for p in control_points]
        return cls(jnp.stack([p.loc for p in arr]), **kwargs)

    def transform(self, f: Transformation) -> Self:
        return QuadraticBezierPath(f.apply_batch(self.mat), stroke=self.stroke, markers=self.markers)


class CubicBezierCurve(Parametric):
    pass


class CubicBezierPath(Parametric):
    pass


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


def intersect_interval_interval(i0: tuple[Scalar, Scalar], i1: tuple[Scalar, Scalar]) -> tuple[Scalar, Scalar] | Scalar | None:
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
        l = jnp.maximum(s0, s1)
        r = jnp.minimum(t0, t1)
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
    if jnp.allclose(pp, 0, Global.approx_eps):  # coincident lines
        return l0
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


def intersect_line_segment_param(l: Line, s: LineSegment) -> list[Scalar]:
    a = l.coef.dot(jnp.concat([s.p1.loc - s.p0.loc, jnp.array([0])]))
    b = l.implicit_func(s.p0)
    return [x for x in solve_linear(a, b) if 0 <= x <= 1]


def intersect_line_segment(l: Line, s: LineSegment) -> Point | LineSegment | None:
    """
    Intersect a line and a line segment.
    """
    x = intersect_line_line(l, s.extend_as_line())
    if isinstance(x, Point):
        ts = intersect_line_segment_param(l, s)
        if len(ts) == 1:
            return x
        elif len(ts) == 2:
            return LineSegment(s.at(ts[0]), s.at(ts[1]))
    elif isinstance(x, Line):
        return s  # line overlaps with segment


def intersect_line_conic(l: Line, c: Conic) -> list[Point]:
    A = c.proj_matrix
    x0 = l.at(1/2).to_proj_point().loc
    v = jnp.concat([l.direction_vector.vec, jnp.array([0])])
    ts = solve_quadratic(
        v.T @ A @ v,
        2 * x0.T @ A @ v,
        x0.T @ A @ x0
    )
    def to_point(t: Scalar) -> Point:
        p = ProjPoint(x0 + t * v).to_point()
        assert p is not None
        return p
    return [to_point(t) for t in ts]


def intersect_parabola_segment_parametric(p: Parabola, s: LineSegment) -> list[Scalar]:
    pass


def intersect_line_aabb(l: Line, aabb: AxisAlignedRectangle) -> Point | list[Point] | LineSegment | None:
    scores = [l.implicit_func(v).item() for v in aabb.vertices]
    num_ge_0 = sum(s >= 0 for s in scores)
    if num_ge_0 == 0 or num_ge_0 == 4:
        return None  # no intersection
    else:
        ps = [intersect_line_segment(l, s) for s in aabb.edges]
        return [p for p in ps if p is not None]


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
