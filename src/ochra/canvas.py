from dataclasses import dataclass
from typing import Collection

from ochra.plane import Point, PointI
from ochra.group import Group
from ochra.element import Element
from ochra.rect import AxisAlignedRectangle


class Canvas(Group):

    def __init__(self,
                 elements: Collection[Element],
                 viewport: AxisAlignedRectangle
                 ):
        super().__init__(elements)
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
