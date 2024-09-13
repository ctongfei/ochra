from typing import Callable, Collection, Optional, Tuple

from ochra.element import Element
from ochra.group import Group
from ochra.mark import Mark
from ochra.marker import Marker
from ochra.plane import Point
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.plot.typedef import X, Y


class ScatterPlot(Plot[float, float]):
    def __init__(self,
                 data: Collection[Tuple[float, float, ...]],
                 *,
                 marker: Optional[Marker] = None,
                 marker_fn: Optional[Callable[[Tuple[float, float, ...]], Marker]] = None
                 ):
        self.data = data
        if marker_fn is None:
            assert marker is not None, "One of marker and marker_fn must be set"
            self.marker_fn = lambda t: marker
        else:
            self.marker_fn = marker_fn

    def draw(self, x_axis: Axis[X], y_axis: Axis[Y]) -> Element:
        return Group(
            elements=[
                Mark(Point(x_axis.locate(x[0]), y_axis.locate(x[1])), self.marker_fn(x))
                for x in self.data if x[0] in x_axis and x[1] in y_axis
            ]
        )

