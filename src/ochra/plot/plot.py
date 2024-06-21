from abc import ABC, abstractmethod
from typing import Collection, Generic, Tuple

from ochra.element import Element
from ochra.plane import Point
from ochra.poly import Polyline
from ochra.plot import Axis
from ochra.plot.typedef import X, Y
from ochra.style.stroke import Stroke


class Plot(ABC, Generic[X, Y]):
    @abstractmethod
    def draw(self, x_axis: Axis, y_axis: Axis) -> Element:
        pass


class LinePlot(Plot[X, Y]):
    def __init__(self, data: Collection[Tuple[X, Y]], stroke: Stroke = Stroke()):
        self.data = data
        self.stroke = stroke

    def draw(self, x_axis: Axis, y_axis: Axis) -> Element:
        return Polyline(
            sorted(
                [Point(x_axis.locate(x), y_axis.locate(y)) for x, y in self.data],
                key=lambda p: p.x
            ),
            stroke=self.stroke
        )
