from typing import Collection, Tuple

from ochra.core import Element, Group, Text
from ochra.style import Font, Fill, Stroke
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.plot.collections import X, Y


class BarPlot(Plot[X, Y]):

    def __init__(self,
                 name: str,
                 data: Collection[Tuple[X, Y, ...]],
                 stroke: Stroke = Stroke(),
                 bar_width: float = 1.0,  # TODO: better name
                 fill: Fill = Fill(),
                 ):
        self.name = name
        self.data = data
        self.stroke = stroke
        self.bar_width = bar_width
        self.fill = fill

    def draw(self, x_axis: Axis[X], y_axis: Axis[Y]) -> Element:
        from ochra.rect import AxisAlignedRectangle

        def bar(t: tuple[X, Y, ...]) -> Element:
            x, y = x_axis.locate(t[0]), y_axis.locate(t[1])
            return AxisAlignedRectangle(
                (x - self.bar_width / 2, 0),
                (x + self.bar_width / 2, y),
                stroke=self.stroke,
                fill=self.fill
            )

        return Group([bar(t) for t in self.data])

    def legend(self, font: Font) -> list[tuple[Element, Element]]:
        size = font.extents.height
        mark = AxisAlignedRectangle((0, 0), (size, size), fill=self.fill, stroke=self.stroke)
        text = Text(self.name, (0, 0), font=font)
        return [(mark, text)]
