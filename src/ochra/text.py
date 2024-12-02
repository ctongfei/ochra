from functools import cached_property

from ochra.geometry import Point, PointI, AffineTransformation, Translation, Rotation
from ochra.core import Element, AnyAffinelyTransformed, Rectangle, AxisAlignedRectangle
from ochra.style import Font, TextExtents, _text_extents


class Text(Element):

    def __init__(self, text: str, bottom_left: PointI, angle: float = 0.0,
                 font: Font = Font()):
        self.text = text
        self.bottom_left = Point.mk(bottom_left)
        self.angle = angle
        self.font = font
        self.extents: TextExtents = _text_extents(self.text, self.font)

    @cached_property
    def _rotated_visual_bbox(self) -> 'Rectangle':
        rect = AxisAlignedRectangle(
            Point.origin,
            Point.mk((self.extents.x_advance, -self.extents.y_bearing))
        ).rotate(self.angle).translate(self.bottom_left.x, self.bottom_left.y)
        return rect

    @cached_property
    def _rotated_actual_bbox(self) -> 'Rectangle':
        rect = AxisAlignedRectangle(
            Point.mk(self.extents.x_bearing,
                     -self.extents.y_bearing - self.extents.height),
            Point.mk(
                (self.extents.x_bearing + self.extents.width, -self.extents.y_bearing))
        ).rotate(self.angle).translate(self.bottom_left.x, self.bottom_left.y)
        return rect

    def visual_center(self) -> Point:
        tr = Translation(self.bottom_left.to_vector()) @ Rotation(self.angle)
        midpoint = Point.mk(self.extents.x_advance / 2, self.font.extents.x_height / 2)
        return tr(midpoint)

    def visual_bbox(self) -> 'AxisAlignedRectangle':
        return self._rotated_visual_bbox.aabb()

    @property
    def center(self) -> Point:
        bbox = self._rotated_visual_bbox
        return bbox.center

    @property
    def height(self) -> float:
        return self._rotated_visual_bbox.height

    @property
    def width(self) -> float:
        return self._rotated_visual_bbox.width

    # TODO: use visual center
    @classmethod
    def centered(cls, text: str, center: PointI, angle: float = 0.0,
                 font: Font = Font()) -> 'Text':
        center = Point.mk(center)
        bbox = cls(text, Point.origin, angle, font)._rotated_visual_bbox
        return cls(text, center - bbox.center.to_vector(), angle, font)

    @classmethod
    def top_centered(cls, text: str, top_center: PointI, angle: float = 0.0,
                     font: Font = Font()) -> 'Text':
        top_center = Point.mk(top_center)
        bbox = cls(text, Point.origin, angle, font)._rotated_visual_bbox
        return cls(text, top_center - bbox.top_center.to_vector(), angle, font)

    @classmethod
    def right_centered(cls, text: str, right_center: PointI, angle: float = 0.0,
                       font: Font = Font()) -> 'Text':
        right_center = Point.mk(right_center)
        bbox = cls(text, Point.origin, angle, font)._rotated_visual_bbox
        return cls(text, right_center - bbox.right_center.to_vector(), angle, font)

    @classmethod
    def bottom_centered(cls, text: str, bottom_center: PointI, angle: float = 0.0,
                        font: Font = Font()) -> 'Text':
        bottom_center = Point.mk(bottom_center)
        bbox = cls(text, Point.origin, angle, font)._rotated_visual_bbox
        return cls(text, bottom_center - bbox.bottom_center.to_vector(), angle, font)

    @classmethod
    def left_centered(cls, text: str, left_center: PointI, angle: float = 0.0,
                      font: Font = Font()) -> 'Text':
        left_center = Point.mk(left_center)
        bbox = cls(text, Point.origin, angle, font)._rotated_visual_bbox
        return cls(text, left_center - bbox.left_center.to_vector(), angle, font)

    def translate(self, dx: float, dy: float) -> 'Text':
        tr = Translation((dx, dy))
        return Text(self.text, tr(self.bottom_left), self.angle, self.font)

    def transform(self, f: AffineTransformation) -> 'Element':
        t, r, _, s = f.decompose()
        if s.scale.x == 1.0 and s.scale.y == -1.0:  # TODO
            return Text(self.text, t(self.bottom_left),
                        self.angle + r.angle, self.font)
        else:
            return AnyAffinelyTransformed(self, f)

    def aabb(self) -> AxisAlignedRectangle:
        return self._rotated_actual_bbox.aabb()
