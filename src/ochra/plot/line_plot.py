from typing import Collection, Optional, Tuple

from ochra.core import Element, Group, Polyline
from ochra.text import Text
from ochra.style import Font, Stroke
from ochra.mark import Mark, Marker
from ochra.geometry import Point
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot



class LinePlot(Plot[float, float]):
    """
    Represents a line plot.
    """
    def __init__(self,
                 name: str,
                 data: Collection[Tuple[float, float, ...]],
                 stroke: Stroke = Stroke(),
                 marker: Optional[Marker] = None,
                 ):
        self.name = name
        self.data = data
        self.stroke = stroke
        self.marker = marker

    def draw(self, x_axis: Axis[float], y_axis: Axis[float]) -> Element:
        points = sorted(
            [
                Point.mk((x_axis.locate(x), y_axis.locate(y)))
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

    def legend(self, font: Font) -> list[tuple[Element, Element]]:
        size = font.extents.height
        mark = Polyline([(0, 0), (size / 2, 0), (size, 0)], stroke=self.stroke, marker_mid=self.marker)
        text = Text(self.name, (0, 0), font=font)
        return [(mark, text)]
