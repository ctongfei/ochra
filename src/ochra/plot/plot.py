from abc import ABC, abstractmethod
from typing import Collection, Generic, Tuple, Optional

from ochra.group import Group
from ochra.mark import Mark
from ochra.marker import Marker
from ochra.element import Element
from ochra.plane import Point
from ochra.poly import Polyline
from ochra.plot import ContinuousAxis
from ochra.plot.typedef import X, Y
from ochra.style.stroke import Stroke


class Plot(ABC, Generic[X, Y]):
    @abstractmethod
    def draw(self, x_axis: ContinuousAxis, y_axis: ContinuousAxis) -> Element:
        pass


class LinePlot(Plot[X, Y]):
    """
    Represents a line plot.
    """
    def __init__(self,
                 data: Collection[Tuple[X, Y]],
                 stroke: Stroke = Stroke(),
                 marker: Optional[Marker] = None,
                 ):
        self.data = data
        self.stroke = stroke
        self.marker = marker

    def draw(self, x_axis: ContinuousAxis, y_axis: ContinuousAxis) -> Element:
        points = sorted(
            [Point(x_axis.locate(x), y_axis.locate(y)) for x, y in self.data],
            key=lambda p: p.x
        )
        if self.marker is not None:
            marks = [
                Mark(p, self.marker)
                for p in points
            ]
        else:
            marks = []
        return Group(
            elements=[
                Polyline(
                    points,
                    stroke=self.stroke,
                ),
                *marks
            ]
        )


class ScatterPlot(Plot[X, Y]):
    def __init__(self,
                 data: Collection[Tuple[X, Y]],
                 marker: Marker,
                 ):
        self.data = data
        self.marker = marker

    def draw(self, x_axis: ContinuousAxis, y_axis: ContinuousAxis) -> Element:
        return Group(
            elements=[
                Mark(Point(x_axis.locate(x), y_axis.locate(y)), self.marker)
                for x, y in self.data
            ]
        )

