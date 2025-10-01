from enum import Enum
from typing import Optional, Union
import jax.numpy as jnp

from ochra.core import AxisAlignedRectangle, Element, Group
from ochra.style import Color, Fill, Stroke


class HorizontalAlignment(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

    @classmethod
    def mk(cls, s: Union["HorizontalAlignment", str]) -> Optional["HorizontalAlignment"]:
        return (
            s
            if isinstance(s, HorizontalAlignment)
            else {
                "left": cls.LEFT,
                "center": cls.CENTER,
                "right": cls.RIGHT,
                "l": cls.LEFT,
                "c": cls.CENTER,
                "r": cls.RIGHT,
            }[s]
        )


class VerticalAlignment(Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"

    @classmethod
    def mk(cls, s: Union["VerticalAlignment", str]) -> Optional["VerticalAlignment"]:
        return (
            s
            if isinstance(s, VerticalAlignment)
            else {
                "top": cls.TOP,
                "center": cls.CENTER,
                "bottom": cls.BOTTOM,
                "t": cls.TOP,
                "c": cls.CENTER,
                "b": cls.BOTTOM,
            }[s]
        )


def _displacement(
    inner: AxisAlignedRectangle, outer: AxisAlignedRectangle, ha: HorizontalAlignment, va: VerticalAlignment
) -> tuple[float, float]:
    dx = {
        HorizontalAlignment.LEFT: outer.bottom_left.x - inner.bottom_left.x,
        HorizontalAlignment.CENTER: outer.visual_center().x - inner.visual_center().x,
        HorizontalAlignment.RIGHT: outer.bottom_right.x - inner.bottom_right.x,
    }[ha]
    dy = {
        VerticalAlignment.TOP: outer.top_left.y - inner.top_left.y,
        VerticalAlignment.CENTER: outer.visual_center().y - inner.visual_center().y,
        VerticalAlignment.BOTTOM: outer.bottom_left.y - inner.bottom_left.y,
    }[va]
    return dx, dy


class Table(Group):
    def __init__(
        self,
        cells: list[list[Element]],
        row_heights: list[float | None] | None = None,
        col_widths: list[float | None] | None = None,
        row_alignment: list[VerticalAlignment | str] | str | None = None,
        col_alignment: list[HorizontalAlignment | str] | str | None = None,
        cell_horizontal_padding: float = 2,
        cell_vertical_padding: float = 2,
        border_stroke: Stroke | None = None,
        background: Fill | None = None,
    ):
        self.cells = cells
        self.num_rows = len(cells)
        self.num_cols = max(len(cells[i]) for i in range(self.num_rows))

        col_alignment = col_alignment or [HorizontalAlignment.CENTER for _ in range(self.num_cols)]
        row_alignment = row_alignment or [VerticalAlignment.CENTER for _ in range(self.num_rows)]

        cell_bboxes = [[cell.visual_bbox() for cell in row] for row in cells]
        row_heights = row_heights or [None for _ in range(self.num_rows)]
        col_widths = col_widths or [None for _ in range(self.num_cols)]
        self.row_heights = [
            row_heights[i] or (max(cell_bboxes[i][j].height for j in range(self.num_cols)) + 2 * cell_vertical_padding)
            for i in range(self.num_rows)
        ]
        self.col_widths = [
            col_widths[j] or (max(cell_bboxes[i][j].width for i in range(self.num_rows)) + 2 * cell_horizontal_padding)
            for j in range(self.num_cols)
        ]
        cols_x = jnp.cumsum(jnp.array([0] + self.col_widths)).tolist()
        rows_y = list(reversed(jnp.cumsum(jnp.array([0] + self.row_heights))))
        boxes = [
            [
                AxisAlignedRectangle(
                    (cols_x[j] + cell_horizontal_padding, rows_y[i + 1] + cell_vertical_padding),
                    (cols_x[j + 1] - cell_horizontal_padding, rows_y[i] - cell_vertical_padding),
                ).with_stroke(Stroke(Color(0, 0, 1)))
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]
        aligned_cells = [
            (
                cells[i][j].translate(
                    *_displacement(
                        cells[i][j].aabb(),
                        boxes[i][j],
                        HorizontalAlignment.mk(col_alignment[j] or HorizontalAlignment.LEFT),
                        VerticalAlignment.mk(row_alignment[i] or VerticalAlignment.BOTTOM),
                    )
                )
            )
            for i in range(self.num_rows)
            for j in range(self.num_cols)
        ]
        border = AxisAlignedRectangle((0, 0), (cols_x[-1], rows_y[0]), stroke=border_stroke, fill=background)

        super().__init__(elements=[border] + aligned_cells)
