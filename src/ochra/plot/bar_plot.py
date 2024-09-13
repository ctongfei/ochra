from typing import Collection, Tuple, TYPE_CHECKING

from ochra.element import Element
from ochra.group import Group
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.plot.typedef import X
from ochra.style.fill import Fill
from ochra.style.stroke import Stroke

if TYPE_CHECKING:
    from ochra.rect import AxisAlignedRectangle


class BarPlot(Plot[X, float]):

    def __init__(self,
                 data: Collection[Tuple[X, float, ...]],
                 stroke: Stroke = Stroke(),
                 bar_width: float = 1.0,  # TODO: better name
                 fill: Fill = Fill(),
                 ):
        self.data = data
        self.stroke = stroke
        self.bar_width = bar_width
        self.fill = fill

    def draw(self, x_axis: Axis[X], y_axis: Axis[float]) -> Element:
        return Group(
            elements=[
                AxisAlignedRectangle(
                    (x_axis.locate(x) - self.bar_width / 2, 0),
                    (x_axis.locate(x) + self.bar_width / 2, y_axis.locate(y)),
                    stroke=self.stroke,
                    fill=self.fill
                )
                for x, y in self.data
            ]
        )
