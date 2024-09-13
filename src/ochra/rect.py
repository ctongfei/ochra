import math

from ochra.line import Line
from ochra.parametric import Parametric
from ochra.plane import Point, PointI, Transformation, Vector
from ochra.poly import Polygon
from ochra.style import Fill
from ochra.style.stroke import Stroke
from ochra.util.functions import dist, lerp_point


class Rectangle(Polygon, Parametric):

    def __init__(self, bottom_left: PointI, top_right: PointI, angle: float = 0.0, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        self.bottom_left = Point.mk(bottom_left)
        self.top_right = Point.mk(top_right)
        ray = Line.from_two_points(self.bottom_left, self.bottom_left + Vector(math.cos(angle), math.sin(angle)))
        self.bottom_right = ray.closest_to(top_right)
        self.top_left = self.top_right + (self.bottom_left - self.bottom_right)
        vertices = [self.bottom_left, self.bottom_right, self.top_right, self.top_left]
        self.angle = angle
        self.width = dist(self.bottom_left, self.bottom_right)
        self.height = dist(self.bottom_right, self.top_right)
        super().__init__(vertices, stroke, fill)

    @property
    def center(self):
        return lerp_point(self.bottom_left, self.top_right, 0.5)

    @property
    def top_center(self):
        return lerp_point(self.top_left, self.top_right, 0.5)

    @property
    def right_center(self):
        return lerp_point(self.bottom_right, self.top_right, 0.5)

    @property
    def bottom_center(self):
        return lerp_point(self.bottom_left, self.bottom_right, 0.5)

    @property
    def left_center(self):
        return lerp_point(self.bottom_left, self.top_left, 0.5)

    def axis_aligned_bbox(self) -> 'AxisAlignedRectangle':
        l = min(p.x for p in self.vertices)
        r = max(p.x for p in self.vertices)
        b = min(p.y for p in self.vertices)
        u = max(p.y for p in self.vertices)
        return AxisAlignedRectangle(bottom_left=(l, b), top_right=(r, u))

    def translate(self, dx: float, dy: float) -> 'Rectangle':
        return Rectangle(
            self.bottom_left + Vector(dx, dy),
            self.top_right + Vector(dx, dy),
            self.angle,
            stroke=self.stroke,
            fill=self.fill
        )

    def rotate(self, angle: float, anchor: PointI = Point.origin) -> 'Rectangle':
        rot = Transformation.rotate(angle, anchor)
        return Rectangle(
            rot(self.bottom_left),
            rot(self.top_right),
            self.angle + angle,
            stroke=self.stroke,
            fill=self.fill
        )


class AxisAlignedRectangle(Rectangle, Parametric):

    def __init__(self, bottom_left: PointI, top_right: PointI, stroke: Stroke = Stroke(), fill: Fill = Fill()):
        super().__init__(bottom_left, top_right, 0.0, stroke, fill)

    def axis_aligned_bbox(self) -> 'AxisAlignedRectangle':
        return self

    def scale(self, sx: float, sy: float) -> 'AxisAlignedRectangle':
        return AxisAlignedRectangle(
            self.bottom_left.scale(sx, sy),
            self.top_right.scale(sx, sy),
            stroke=self.stroke,
            fill=self.fill
        )
