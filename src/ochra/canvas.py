from dataclasses import dataclass
from typing import Collection, Optional

from ochra.element import Element
from ochra.group import Group
from ochra.plane import Point, PointI
from ochra.rect import AxisAlignedRectangle


class Canvas(Group):

    def __init__(self,
                 elements: Collection[Element],
                 viewport: Optional[AxisAlignedRectangle] = None
                 ):
        super().__init__(elements)
        if viewport is None:
            viewport = self.axis_aligned_bbox()
        self.viewport = viewport


@dataclass(init=False)
class EmbeddedCanvas(Group):
    canvas: Canvas
    left_bottom: Point

    def __init__(self, canvas: Canvas, left_bottom: PointI):
        self.canvas = canvas
        self.left_bottom = Point.mk(left_bottom)
        super().__init__(elements=[
            canvas.translate(self.left_bottom.x, self.left_bottom.y)
        ])
