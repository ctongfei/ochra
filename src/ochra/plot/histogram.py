from collections.abc import Collection, Sequence

from ochra.core import AxisAlignedRectangle, Element, Group
from ochra.text import Text
from ochra.plot import Plot, Axis
from ochra.plot.transform import histogram
from ochra.style import Stroke, Fill, Font
from ochra.plot.collections import Xc, Yc


class Histogram(Plot[Xc, Yc]):
    def __init__(self,
                 name: str,
                 data: Collection[float],
                 ticks: Sequence[float] | None = None,
                 stroke: Stroke = Stroke(),
                 fill: Fill = Fill(),
                 ):
        self.name = name
        self.data = data
        self.ticks = ticks
        self.stroke = stroke
        self.fill = fill

    def draw(self, x_axis: Axis[float], y_axis: Axis[float]) -> Element:
        ticks = self.ticks if self.ticks else x_axis.minor_ticks if x_axis.minor_ticks else x_axis.major_ticks
        hist = histogram(self.data, ticks)
        return Group(
            elements=[
                AxisAlignedRectangle(
                    (x_axis.locate(l), 0),
                    (x_axis.locate(r), y_axis.locate(count)),
                    stroke=self.stroke,
                    fill=self.fill
                )
                for l, r, count in hist
            ]
        )

    def legend(self, font: Font) -> list[tuple[Element, Element]]:
        size = font.size
        mark = AxisAlignedRectangle((0, 0), (size, size), fill=self.fill, stroke=self.stroke)
        text = Text(self.name, (0, 0), font=font)
        return [(mark, text)]