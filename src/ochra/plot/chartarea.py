from dataclasses import dataclass, field
from typing import Generic, Collection

from ochra.group import Group
from ochra.plane import Point
from ochra.segment import LineSegment
from ochra.rect import AxisAlignedRectangle
from ochra.style import Fill
from ochra.text import Text
from ochra.canvas import Canvas
from ochra.plot.typedef import X, Y
from ochra.plot.axis import Axis
from ochra.plot.plot import Plot
from ochra.style.font import Font
from ochra.style.stroke import Stroke


@dataclass
class ChartArea(Canvas, Generic[X, Y]):
    x_axis: Axis
    y_axis: Axis
    plots: Collection[Plot[X, Y]]
    font: Font = field(default_factory=Font)
    background: Fill = field(default_factory=Fill)
    border_stroke: Stroke = field(default_factory=Stroke)
    grid_stroke: Stroke = field(default_factory=Stroke)

    def draw_x_axis(self) -> Group:
        X = self.x_axis
        axis_line = LineSegment(
            Point(X.locate(X.lower_bound), 0),
            Point(X.locate(X.upper_bound), 0),
            stroke=self.border_stroke
        )
        major_ticks = [] if X.major_ticks is None else [
            Group([
                LineSegment(
                    Point(X.locate(x) * X.scale, 0),
                    Point(X.locate(x) * X.scale, -X.major_tick_length),
                    stroke=X.tick_stroke
                ),
                Text.top_centered(
                    f"{x:.2f}",
                    Point(X.locate(x) * X.scale, -X.major_tick_length - X.text_padding),
                    font=self.font
                )
            ])
            for x in X.major_ticks
        ]
        minor_ticks = [] if X.minor_ticks is None else [
            LineSegment(
                Point(X.locate(x) * X.scale, 0),
                Point(X.locate(x) * X.scale, -X.minor_tick_length),
                stroke=X.tick_stroke
            )
            for x in X.minor_ticks
        ]
        return Group(
            elements=[
                axis_line,
                *major_ticks,
                *minor_ticks
            ]
        )

    def draw_y_axis(self) -> Group:
        Y = self.y_axis
        axis_line = LineSegment(
            Point(0, Y.locate(Y.lower_bound)),
            Point(0, Y.locate(Y.upper_bound))
        )
        major_ticks = [] if Y.major_ticks is None else [
            Group([
                LineSegment(
                    Point(0, Y.locate(y) * Y.scale),
                    Point(-Y.major_tick_length, Y.locate(y) * Y.scale),
                    stroke=Y.tick_stroke
                ),
                Text.right_centered(
                    f"{y:.2f}",
                    Point(-Y.major_tick_length - Y.text_padding, Y.locate(y) * Y.scale),
                    font=self.font
                )
            ])
            for y in Y.major_ticks
        ]
        minor_ticks = [] if Y.minor_ticks is None else [
            LineSegment(
                Point(0, Y.locate(y) * Y.scale),
                Point(-Y.minor_tick_length, Y.locate(y) * Y.scale),
            )
            for y in Y.minor_ticks
        ]
        return Group(
            elements=[
                axis_line,
                *major_ticks,
                *minor_ticks
            ]
        )

    def draw_grid(self) -> Group:
        X = self.x_axis
        Y = self.y_axis
        x_ticks_to_draw = [] if X.major_ticks is None else [x for x in X.major_ticks if X.lower_bound < x < X.upper_bound]
        y_ticks_to_draw = [] if Y.major_ticks is None else [y for y in Y.major_ticks if Y.lower_bound < y < Y.upper_bound]
        x_lines = [
            LineSegment(
                Point(X.locate(x) * X.scale, Y.locate(Y.lower_bound) * Y.scale),
                Point(X.locate(x) * X.scale, Y.locate(Y.upper_bound) * Y.scale),
                stroke=self.grid_stroke
            )
            for x in x_ticks_to_draw
        ]
        y_lines = [
            LineSegment(
                Point(X.locate(X.lower_bound) * X.scale, Y.locate(y) * Y.scale),
                Point(X.locate(X.upper_bound) * X.scale, Y.locate(y) * Y.scale),
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

    def __post_init__(self):
        X = self.x_axis
        Y = self.y_axis
        rect = AxisAlignedRectangle(
            Point(X.locate(X.lower_bound) * X.scale, Y.locate(Y.lower_bound) * Y.scale),
            Point(X.locate(X.upper_bound) * Y.scale, Y.locate(Y.upper_bound) * Y.scale),
            stroke=self.border_stroke,
            fill=self.background
        )
        x_axis = self.draw_x_axis()
        y_axis = self.draw_y_axis()
        grid = self.draw_grid()
        super().__init__(
            [
                rect,
                grid,
                x_axis,
                y_axis,
                Group(
                    elements=[
                        plot.draw(X, Y)
                        for plot in self.plots
                    ]
                ).scale(X.scale, Y.scale).translate(X.lower_bound, Y.lower_bound)
            ],
            viewport=rect,
        )
