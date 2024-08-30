import math
from dataclasses import dataclass, field
from typing import Generic, Collection, Tuple

from ochra.group import Group, Annotation
from ochra.plane import Point
from ochra.segment import LineSegment
from ochra.rect import AxisAlignedRectangle
from ochra.style import Fill
from ochra.text import Text
from ochra.canvas import Canvas
from ochra.plot.typedef import X, Y
from ochra.plot.axis import ContinuousAxis
from ochra.plot.plot import Plot
from ochra.style.font import Font
from ochra.style.stroke import Stroke
from ochra.util.functions import f2s


class ChartArea(Canvas, Generic[X, Y]):

    def __init__(self,
                 size: Tuple[float, float],
                 x_axis: ContinuousAxis,
                 y_axis: ContinuousAxis,
                 plots: Collection[Plot[X, Y]],
                 font: Font = Font(),
                 background: Fill = Fill(),
                 border_stroke: Stroke = Stroke(),
                 grid_stroke: Stroke = Stroke(),
                 ):
        self.x_size, self.y_size = size
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.plots = plots
        self.font = font
        self.background = background
        self.border_stroke = border_stroke
        self.grid_stroke = grid_stroke
        X = x_axis
        Y = y_axis
        rect = AxisAlignedRectangle(
            (0, 0),
            (self.x_size, self.y_size),
            stroke=border_stroke,
            fill=background
        )
        x_axis = self._draw_x_axis()
        y_axis = self._draw_y_axis()
        x_scale = self.x_size / (X.upper_bound - X.lower_bound)
        y_scale = self.y_size / (Y.upper_bound - Y.lower_bound)
        grid = self._draw_grid()
        super().__init__(
            [
                rect,
                x_axis.translate(-X.lower_bound, 0).scale(x_scale, 1),
                y_axis.translate(0, -Y.lower_bound).scale(1, y_scale),
                Group(
                    elements=[
                        grid,
                        *[
                            plot.draw(X, Y)
                            for plot in self.plots
                        ]
                    ],
                ).translate(-X.lower_bound, -Y.lower_bound).scale(x_scale, y_scale)
            ],
            viewport=rect,
        )

    def _draw_x_axis(self) -> Group:
        X = self.x_axis
        axis_line = LineSegment(
            Point(X.locate(X.lower_bound), 0),
            Point(X.locate(X.upper_bound), 0),
            stroke=self.border_stroke
        )

        def make_label(x: float):
            return lambda p: Text.top_centered(f2s(x), p, font=self.font)

        major_ticks = [] if X.major_ticks is None else [
            Group([
                LineSegment(
                    Point(X.locate(x), 0),
                    Point(X.locate(x), -X.major_tick_length),
                    stroke=X.tick_stroke
                ),
                Annotation(
                    Point(X.locate(x), -X.major_tick_length - X.text_padding),
                    make_label(x)
                )
            ])
            for x in X.major_ticks
        ]
        text_height = max(
            t.materialize().height
            for tick in major_ticks
            for t in tick.recursive_children() if isinstance(t, Annotation)
        )
        minor_ticks = [] if X.minor_ticks is None else [
            LineSegment(
                Point(X.locate(x), 0),
                Point(X.locate(x), -X.minor_tick_length),
                stroke=X.tick_stroke
            )
            for x in X.minor_ticks
        ]
        axis_label = Annotation(
            Point(
                (X.locate(X.lower_bound) + X.locate(X.upper_bound)) / 2,
                - X.major_tick_length - text_height - 2 * X.text_padding
            ),
            lambda p: Text.top_centered(
                X.label,
                top_center=p,
                font=self.font.bolded()
            )
        )
        return Group(
            elements=[
                axis_line,
                axis_label,
                *major_ticks,
                *minor_ticks
            ]
        )

    def _draw_y_axis(self) -> Group:
        Y = self.y_axis
        axis_line = LineSegment(
            Point(0, Y.locate(Y.lower_bound)),
            Point(0, Y.locate(Y.upper_bound)),
            stroke=self.border_stroke
        )

        def make_label(y: float):
            return lambda p: Text.right_centered(f2s(y), p, font=self.font)

        major_ticks = [] if Y.major_ticks is None else [
            Group([
                LineSegment(
                    Point(0, Y.locate(y)),
                    Point(-Y.major_tick_length, Y.locate(y)),
                    stroke=Y.tick_stroke
                ),
                Annotation(
                    Point(-Y.major_tick_length - Y.text_padding, Y.locate(y)),
                    make_label(y)
                )
            ])
            for y in Y.major_ticks
        ]
        text_width = max(
            t.materialize().width
            for tick in major_ticks
            for t in tick.recursive_children() if isinstance(t, Annotation)
        )
        minor_ticks = [] if Y.minor_ticks is None else [
            LineSegment(
                Point(0, Y.locate(y)),
                Point(-Y.minor_tick_length, Y.locate(y)),
            )
            for y in Y.minor_ticks
        ]

        axis_label = Annotation(
            Point(
                -Y.major_tick_length - text_width - 2 * Y.text_padding,
                (Y.locate(Y.lower_bound) + Y.locate(Y.upper_bound)) / 2
            ),
            lambda p: Text.bottom_centered(
                text=Y.label,
                bottom_center=p,
                angle=math.tau / 4,
                font=self.font.bolded()
            )
        )
        return Group(
            elements=[
                axis_line,
                axis_label,
                *major_ticks,
                *minor_ticks
            ]
        )

    def _draw_grid(self) -> Group:
        X = self.x_axis
        Y = self.y_axis
        x_ticks_to_draw = [] if X.major_ticks is None else [x for x in X.major_ticks if X.lower_bound < x < X.upper_bound]
        y_ticks_to_draw = [] if Y.major_ticks is None else [y for y in Y.major_ticks if Y.lower_bound < y < Y.upper_bound]
        x_lines = [
            LineSegment(
                Point(X.locate(x), Y.locate(Y.lower_bound)),
                Point(X.locate(x), Y.locate(Y.upper_bound)),
                stroke=self.grid_stroke
            )
            for x in x_ticks_to_draw
        ]
        y_lines = [
            LineSegment(
                Point(X.locate(X.lower_bound), Y.locate(y)),
                Point(X.locate(X.upper_bound), Y.locate(y)),
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
