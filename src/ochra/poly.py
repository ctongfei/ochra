import math
from typing import Sequence, Optional

from ochra.element import Element
from ochra.marker import MarkerConfig
from ochra.parameterizable import Parameterizable1, Joined
from ochra.plane import Point, Transformation, PointI
from ochra.segment import LineSegment
from ochra.style.stroke import Stroke
from ochra.style.fill import Fill


class Polyline(Parameterizable1):

    def __init__(self,
                 vertices: Sequence[PointI],
                 stroke: Stroke = Stroke(),
                 markers: MarkerConfig = MarkerConfig()
                 ):
        self.vertices = [Point.mk(v) for v in vertices]
        self.stroke = stroke
        self.markers = markers

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def edges(self):
        return [
            LineSegment(self.vertices[i], self.vertices[i + 1])
            for i in range(self.num_vertices - 1)
        ]

    def at(self, t: float):
        if t == 1.0:
            return self.vertices[-1]
        x = t * (self.num_vertices - 1)
        i = int(x)
        t0 = x - i
        return self.edges[i].at(t0)

    def transform(self, f: Transformation) -> 'Polyline':
        return Polyline([f(v) for v in self.vertices], stroke=self.stroke, markers=self.markers)


class Polygon(Parameterizable1):

    def __init__(self,
                 vertices: Sequence[PointI],
                 stroke: Stroke = Stroke(),
                 fill: Fill = Fill(),
                 markers: MarkerConfig = MarkerConfig(),
                 ):
        self.vertices = [Point.mk(v) for v in vertices]
        self.stroke = stroke
        self.fill = fill
        self.markers = markers

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def edges(self):
        return [
            LineSegment(self.vertices[i], self.vertices[(i + 1) % self.num_vertices])
            for i in range(self.num_vertices)
        ]

    def at(self, t: float):
        if t == 1.0:
            return self.vertices[0]
        x = t * self.num_vertices
        i = int(x)
        t0 = x - i
        return self.edges[i].at(t0)

    def transform(self, f: Transformation) -> 'Polygon':
        return Polygon([f(v) for v in self.vertices], self.stroke, self.fill, self.markers)

    @classmethod
    def regular(cls,
                n: int,
                *,
                circumradius: Optional[float] = None,
                side_length: Optional[float] = None,
                apothem: Optional[float] = None,
                **kwargs
                ):
        circumradius = _get_circumradius(n, circumradius, side_length, apothem)
        return cls(
            [
                Point(circumradius * math.cos(math.tau * i / n), circumradius * math.sin(math.tau * i / n))
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
            return Joined([mk_part(i) for i in range(num_parts)])


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
