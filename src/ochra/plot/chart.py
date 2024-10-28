import itertools
import math
from enum import Enum
from functools import partial
from typing import Collection, Generic, Tuple, Optional

from ochra.core import *
from ochra.mark import Marker
from ochra.mark import Mark
from ochra.geometry import Point, Vector, Translation, scale
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.plot.collections import X, Y
from ochra.style import Fill, Palette
from ochra.style import Font
from ochra.style import Stroke
from ochra.table import Table
from ochra import Text
from ochra.functions import f2s


class AxisOrientation(Enum):
    X_PRIMARY = 0
    Y_PRIMARY = 1
    X_SECONDARY = 2
    Y_SECONDARY = 3


class Chart(Generic[X, Y]):

    def __init__(self,
                 size: Tuple[float, float],
                 x_axis: Axis[X],
                 y_axis: Axis[Y],
                 plots: Collection[Plot[X, Y]],
                 secondary_y_axis: Optional[Axis[Y]] = None,
                 secondary_y_plots: Optional[Collection[Plot[X, Y]]] = None,
                 extra: Optional[Collection[Element]] = None,
                 background: Optional[Fill] = None,
                 border_stroke: Optional[Stroke] = None,
                 grid_stroke: Optional[Stroke] = None,
                 tick_stroke: Optional[Stroke] = None,
                 font: Font = Font(),
                 text_padding: float = 2,
                 major_tick_length: float = 3,
                 minor_tick_length: float = 2,
                 palette: Optional[Palette] = None
                 ):
        self.x_size, self.y_size = size
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.secondary_y_axis = secondary_y_axis
        self.plots = plots
        self.secondary_y_plots = secondary_y_plots if secondary_y_plots is not None else []
        self.font = font
        self.extra = extra if extra is not None else []

        self.background = background or Fill(color=palette.default_light_color)
        self.border_stroke = border_stroke or Stroke(color=palette.default_dark_color)
        self.grid_stroke = grid_stroke or Stroke(color=palette.default_gray_color)
        self.tick_stroke = tick_stroke or Stroke(color=palette.default_dark_color)
        self.text_style = font
        self.text_padding = text_padding

        self.major_tick_length = major_tick_length
        self.minor_tick_length = minor_tick_length

        X = x_axis
        Y = y_axis
        Y2 = secondary_y_axis
        rect_bg = AxisAlignedRectangle(
            (0, 0),
            (self.x_size, self.y_size),
            fill=self.background
        )
        border = AxisAlignedRectangle(
            (0, 0),
            (self.x_size, self.y_size),
            stroke=self.border_stroke
        )
        x_axis = self._draw_axis(self.x_axis, AxisOrientation.X_PRIMARY)
        y_axis = self._draw_axis(self.y_axis, AxisOrientation.Y_PRIMARY)
        if Y2 is not None:
            y2_axis = self._draw_axis(Y2, AxisOrientation.Y_SECONDARY)
        legend = self._draw_legend()
        legend_bbox = legend.aabb()

        x_scale = self.x_size / (X.locate_upper_bound() - X.locate_lower_bound())
        y_scale = self.y_size / (Y.locate_upper_bound() - Y.locate_lower_bound())
        grid = self._draw_grid()
        tr_x = (Translation((-X.locate_lower_bound(), 0))
                @ scale((x_scale, 1)))
        tr_y = (Translation((0, -Y.locate_lower_bound()))
                @ scale((1, y_scale)))
        tr_xy = Translation((-X.locate_lower_bound(), -Y.locate_lower_bound())) \
                @ scale((x_scale, y_scale))

        self.background_rectangle = rect_bg
        self.grid = grid.transform(tr_xy)
        self.plot_layer = Group(
            elements=[
                plot.draw(X, Y)
                for plot in self.plots
            ],
        ).transform(tr_xy)
        self.extra_layer = Group(elements=self.extra).transform(tr_xy)
        self.border_rectangle = border
        self.legend = legend.translate(
            self.x_size - legend_bbox.width - self.text_padding,
            self.y_size - legend_bbox.height - self.text_padding
        )
        self.x_axis_element = x_axis.transform(tr_x)
        self.y_axis_element = y_axis.transform(tr_y)

    def draw(self,
             border: bool = True,
             legend: bool = True,
             x_axis: bool = True,
             y_axis: bool = True,
             ) -> Group:
        return Group(
            [
                self.background_rectangle,
                self.grid,
                self.plot_layer,
                self.extra_layer
            ] +
            ([self.border_rectangle] if border else []) +
            ([self.legend] if legend else []) +
            ([self.x_axis_element] if x_axis else []) +
            ([self.y_axis_element] if y_axis else [])
        )

    def _draw_axis(self, axis: Axis, ori: AxisOrientation) -> Group:
        X, Y = self.x_axis, self.y_axis
        po = Point.mk((X.locate_lower_bound(), Y.locate_lower_bound()))
        px = Point.mk((X.locate_upper_bound(), Y.locate_lower_bound()))
        py = Point.mk((X.locate_lower_bound(), Y.locate_upper_bound()))
        pxy = Point.mk((X.locate_upper_bound(), Y.locate_upper_bound()))

        axis_line = LineSegment(
            {
                AxisOrientation.X_PRIMARY: po,
                AxisOrientation.Y_PRIMARY: po,
                AxisOrientation.X_SECONDARY: py,
                AxisOrientation.Y_SECONDARY: px
            }[ori],
            {
                AxisOrientation.X_PRIMARY: px,
                AxisOrientation.Y_PRIMARY: py,
                AxisOrientation.X_SECONDARY: pxy,
                AxisOrientation.Y_SECONDARY: pxy
            }[ori],
            stroke=self.border_stroke
        )

        label_text_cont = {
            AxisOrientation.X_PRIMARY: Text.top_centered,
            AxisOrientation.Y_PRIMARY: Text.right_centered,
            AxisOrientation.X_SECONDARY: Text.bottom_centered,
            AxisOrientation.Y_SECONDARY: Text.left_centered,
        }[ori]
        axis_label_cont = {
            AxisOrientation.X_PRIMARY: Text.top_centered,
            AxisOrientation.Y_PRIMARY: partial(Text.bottom_centered, angle=math.tau / 4),
            AxisOrientation.X_SECONDARY: Text.bottom_centered,
            AxisOrientation.Y_SECONDARY: partial(Text.top_centered, angle=math.tau / 4)
        }[ori]

        def make_label(x: object):
            return lambda p: label_text_cont(f2s(x), p, font=self.font)

        tick_point_fn = {
            AxisOrientation.X_PRIMARY: lambda x: Point.mk((X.locate(x), 0)),
            AxisOrientation.Y_PRIMARY: lambda y: Point.mk((0, Y.locate(y))),
            AxisOrientation.X_SECONDARY: lambda x: Point.mk((X.locate(x), self.y_size)),
            AxisOrientation.Y_SECONDARY: lambda y: Point.mk((self.x_size, Y.locate(y)))
        }
        tick_angle = {
            AxisOrientation.X_PRIMARY: 0.75 * math.tau,
            AxisOrientation.Y_PRIMARY: 0.5 * math.tau,
            AxisOrientation.X_SECONDARY: 0.25 * math.tau,
            AxisOrientation.Y_SECONDARY: 0
        }
        major_tick_marker = Marker.tick(self.major_tick_length, tick_angle[ori], stroke=self.tick_stroke)
        major_ticks = [] if axis.major_ticks is None else [
            Group([
                Mark(tick_point_fn[ori](x), marker=major_tick_marker),
                Annotation(
                    tick_point_fn[ori](x) + Vector.unit(tick_angle[ori]) * (self.major_tick_length + self.text_padding),
                    make_label(x)
                )
            ])
            for x in axis.major_ticks
        ]
        get_size = {
            AxisOrientation.X_PRIMARY: lambda t: t.materialize().height,
            AxisOrientation.Y_PRIMARY: lambda t: t.materialize().width,
            AxisOrientation.X_SECONDARY: lambda t: t.materialize().height,
            AxisOrientation.Y_SECONDARY: lambda t: t.materialize().width
        }
        text_size = max(
            get_size[ori](t)
            for tick in major_ticks
            for t in tick.recursive_children() if isinstance(t, Annotation)
        )
        minor_tick_marker= Marker.tick(self.minor_tick_length, tick_angle[ori], stroke=self.tick_stroke)
        minor_ticks = [] if axis.minor_ticks is None else [
            Mark(tick_point_fn[ori](x), marker=minor_tick_marker)
            for x in axis.minor_ticks
        ]
        axis_label = Annotation(
            axis_line.midpoint + Vector.unit(tick_angle[ori]) * (self.major_tick_length + text_size + 2 * self.text_padding),
            lambda p: axis_label_cont(axis.label, p, font=self.font.bolded())
        )
        return Group([axis_line, axis_label, *major_ticks, *minor_ticks])

    def _draw_grid(self) -> Group:
        X = self.x_axis
        Y = self.y_axis
        x_ticks_to_draw = [] if X.major_ticks is None else [x for x in X.major_ticks if x in X]
        y_ticks_to_draw = [] if Y.major_ticks is None else [y for y in Y.major_ticks if y in Y]
        x_lines = [
            LineSegment(
                Point.mk((X.locate(x), Y.locate_lower_bound())),
                Point.mk((X.locate(x), Y.locate_upper_bound())),
                stroke=self.grid_stroke
            )
            for x in x_ticks_to_draw
        ]
        y_lines = [
            LineSegment(
                Point.mk((X.locate_lower_bound(), Y.locate(y))),
                Point.mk((X.locate_upper_bound(), Y.locate(y))),
                stroke=self.grid_stroke
            )
            for y in y_ticks_to_draw
        ]
        return Group(
            elements=[
                *x_lines,
                *y_lines
            ]
        )

    def _draw_legend(self) -> Element:
        legends = list(itertools.chain(*[plot.legend(self.font) for plot in self.plots]))
        if len(legends) == 0:
            return Group([])
        legend = Table(
            legends,
            cell_horizontal_padding=2,
            cell_vertical_padding=(self.font.extents.height - self.font.size),
            border_stroke=self.border_stroke,
            background=self.background,
            col_alignment="cl",
            row_alignment="c" * len(self.plots),
        )
        return legend
