from dataclasses import dataclass
from typing import Collection

from ochra.plane import Point, PointI
from ochra.group import Group
from ochra.element import Element
from ochra.rect import AxisAlignedRectangle


@dataclass
class Canvas(Group):
    elements: Collection[Element]
    viewport: AxisAlignedRectangle


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


