from typing import Collection, Optional, Tuple

from ochra.element import Element
from ochra.group import Group
from ochra.mark import Mark
from ochra.marker import Marker
from ochra.plane import Point
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.poly import Polyline
from ochra.style.stroke import Stroke


class LinePlot(Plot[float, float]):
    """
    Represents a line plot.
    """
    def __init__(self,
                 data: Collection[Tuple[float, float, ...]],
                 stroke: Stroke = Stroke(),
                 marker: Optional[Marker] = None,
                 ):
        self.data = data
        self.stroke = stroke
        self.marker = marker

    def draw(self, x_axis: Axis[float], y_axis: Axis[float]) -> Element:
        points = sorted(
            [
                Point(x_axis.locate(x), y_axis.locate(y))
                for x, y, *_ in self.data
                if x in x_axis and y in y_axis
            ],
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

