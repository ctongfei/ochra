from typing import Collection, Tuple

from ochra.core import Element, Group
from ochra.text import Text
from ochra.mark import Mark, Marker
from ochra.geometry import Point
from ochra.style import Font
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.plot.collections import X, Y


class ScatterPlot(Plot[float, float]):
    def __init__(
        self,
        name: str,
        data: Collection[Tuple[float, float]],
        marker: Marker,
    ):
        self.name = name
        self.data = data
        self.marker = marker

    def draw(self, x_axis: Axis[X], y_axis: Axis[Y]) -> Element:
        return Group(
            elements=[
                Mark(Point(x_axis.locate(x[0]), y_axis.locate(x[1])), self.marker)
                for x in self.data
                if x[0] in x_axis and x[1] in y_axis
            ]
        )

    def legend(self, font: Font) -> list[tuple[Element, Element]]:
        mark = Mark((0, 0), self.marker)
        text = Text(self.name, (0, 0), font=font)
        return [(mark, text)]
