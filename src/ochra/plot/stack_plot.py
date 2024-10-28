from typing import Collection, Tuple, Sequence, Optional

import numpy as np

from ochra.core import Element, AxisAlignedRectangle, Group, Polygon
from ochra.plot import Plot, Axis
from ochra.style import Fill, Stroke, Palette


class StackPlot(Plot[float, float]):
    def __init__(self,
                 data: Collection[Tuple[float, Sequence[float], ...]],
                 fills: Optional[Sequence[Fill]] = None,
                 stroke: Optional[Stroke] = None,
                 palette: Optional[Palette] = None
                 ):
        self.data = data
        self.stroke = stroke
        self.fills = fills
        if fills is None and palette is not None:
            n = len(data[0][1])
            colors = palette.default_color_sequence(n)
            self.fills = [Fill(color=colors[i]) for i in range(n)]
        if stroke is None and palette is not None:
            self.stroke = Stroke(color=palette.default_light_color)

    def draw(self, x_axis: Axis[float], y_axis: Axis[float]):
        m = len(self.fills)
        n = len(self.data)
        x = [t[0] for t in self.data]
        ys = np.array([t[1] for t in self.data])  # R[DataPoint, Stack]
        stacked = np.concatenate(
            [
                np.zeros((ys.shape[0], 1)),
                np.cumsum(ys, axis=1)
            ],
            axis=1
        )  # R[DataPoint, Stack + 1]

        polygons = [
            Polygon(
                vertices=[
                    (x_axis.locate(x[i]), y_axis.locate(stacked[i, j].item()))
                    for i in range(n)
                ] + [
                    (x_axis.locate(x[i]), y_axis.locate(stacked[i, j + 1].item()))
                    for i in range(n - 1, -1, -1)
                ],
                stroke=self.stroke,
                fill=self.fills[j]
            )
            for j in range(m)
        ]
        return Group(elements=polygons)

    def legend(self, size: float) -> Element:
        return Group([
            AxisAlignedRectangle(
                (0.25 * size, 0.25 * size), (0.75 * size, 0.75 * size), fill=self.fills[i])
        ])