from ochra.core import Element, Group
from ochra.plot import Plot, Axis
from ochra.plot.collections import X, Y, standardize_2d, Mapping2D
from ochra.style import Stroke, Palette, Fill, Font, Colormap, InterpolatedColormap


class HeatMap(Plot[X, Y]):
    """
    Represents a heatmap.
    """
    def __init__(self,
                 data: Mapping2D,
                 colormap: Colormap = InterpolatedColormap.viridis,
                 border_stroke: Stroke | None = None,
                 palette: Palette | None = None
                 ):
        self.data = standardize_2d(data)
        self.colormap = colormap
        self.border_stroke = border_stroke if border_stroke is not None else Stroke(color=palette.default_gray_color)

    def draw(self, x_axis: Axis[X], y_axis: Axis[Y]):
        max_value = max(self.data.values())
        min_value = min(self.data.values())
        normalize = lambda x: (x - min_value) / (max_value - min_value)
        return Group(
            elements=[
                AxisAlignedRectangle(
                    (x_axis.locate(x) - 0.5, y_axis.locate(y) - 0.5),
                    (x_axis.locate(x) + 0.5, y_axis.locate(y) + 0.5),
                    fill=Fill(color=self.colormap(normalize(self.data[x, y]))),
                    stroke=self.border_stroke
                )
                for x, y in self.data.keys()
            ]
        )

    def legend(self, font: Font) -> list[tuple[Element, Element]]:
        return []
