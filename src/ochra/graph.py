from typing import Callable

from ochra import Element
from ochra.parameterizable import Parameterizable1
from ochra.plane import Point
from ochra.util.functions import lerp
from ochra.style.stroke import Stroke


class FunctionGraph(Parameterizable1):
    """
    A graph of a function over a specific interval.
    """

    def __init__(self, f: Callable[[float], float], lower_bound: float, upper_bound: float, stroke: Stroke = Stroke()):
        self.f = f
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stroke = stroke

    def at(self, t: float):
        x = lerp(self.lower_bound, self.upper_bound, t)
        y = self.f(x)
        return Point(x, y)
