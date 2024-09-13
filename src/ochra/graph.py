from typing import Callable

from ochra.parametric import Parametric
from ochra.plane import Point
from ochra.style.stroke import Stroke
from ochra.util.functions import lerp


class FunctionGraph(Parametric):
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
