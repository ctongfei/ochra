from ochra.segment import LineSegment
from ochra.parameterizable import Parameterizable1
from ochra.plane import Point, Vector, PointI
from ochra.poly import Polygon
from ochra.style import Fill
from ochra.style.stroke import Stroke


class AxisAlignedRectangle(Polygon, Parameterizable1):

    def __init__(self, left_bottom: PointI, top_right: PointI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        left_bottom = Point.mk(left_bottom)
        top_right = Point.mk(top_right)
        self.width = top_right.x - left_bottom.x
        self.height = top_right.y - left_bottom.y
        assert self.width > 0
        assert self.height > 0
        vertices = [
            left_bottom,
            left_bottom + Vector(self.width, 0),
            top_right,
            left_bottom + Vector(0, self.height),
        ]
        super().__init__(vertices, stroke=stroke, fill=fill)
        self.left_bottom = left_bottom
        self.top_right = top_right

    @property
    def center(self):
        return self.left_bottom + Vector(self.width / 2, self.height / 2)

    @property
    def top_center(self):
        return self.left_bottom + Vector(self.width / 2, self.height)

    @property
    def right_center(self):
        return self.left_bottom + Vector(self.width, self.height / 2)

    @property
    def diagonal0(self):
        return LineSegment(self.vertices[0], self.vertices[2])

    @property
    def diagonal1(self):
        return LineSegment(self.vertices[1], self.vertices[3])

    def scale(self, sx: float, sy: float) -> 'Element':
        return AxisAlignedRectangle(
            self.left_bottom.scale(sx, sy),
            self.top_right.scale(sx, sy),
            stroke=self.stroke,
            fill=self.fill
        )
