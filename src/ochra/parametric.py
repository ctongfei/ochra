from typing import Sequence, Tuple, Optional
import bisect
import numpy as np

from ochra import Element
from ochra.plane import Point, Transformation
from ochra.group import Group


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

    def at(self, t: float) -> Point:
        """
        Returns the point at parameter `t`.
        :param t: A parameter in [0, 1].
        """
        raise NotImplementedError

    def approx_as_polyline(self, ts: Optional[Sequence[float]] = None) -> Element:
        """
        Approximates the element as a polyline, or a group of polylines if not continuous.
        :param ts: Sampling points in [0, 1]. Defaults to uniformly distributed 256 points in [0, 1] if not provided.
        """
        from ochra.poly import Polyline
        stroke = getattr(self, "stroke", None)

        if ts is None:
            ts = np.linspace(0, 1, 256)

        pls = []
        for p in self.pieces:
            l = bisect.bisect(ts, p[0])
            r = bisect.bisect(ts, p[1])
            pl = Polyline([self.at(t) for t in ts[l:r]], stroke=stroke)
            pls.append(pl)

        if len(pls) == 1:
            return pls[0]
        else:
            return Group(pls)

    @staticmethod
    def join(*shapes: 'Parametric') -> 'Element':
        """
        Joins multiple shapes into a single element.
        """
        return Joined(shapes)


class Joined(Group, Parametric):
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
        i = int(t * n)
        t0 = t * n - i
        return self.shapes[i].at(t0)

    def transform(self, f: Transformation) -> 'Element':
        return Joined([s.transform(f) for s in self.shapes])
