import tempfile

from ochra.element import Element
from ochra.plane import Point, PointI, Transformation
from ochra.rect import AxisAlignedRectangle, Rectangle
from ochra.style.font import Font


class Text(Element):

    def __init__(self, text: str, bottom_left: PointI, angle: float = 0.0, font: Font = Font()):
        self.text = text
        self.bottom_left = Point.mk(bottom_left)
        self.angle = angle
        self.font = font
        self.bbox = self._get_bounding_box()

    def _get_bounding_box(self) -> 'Rectangle':
        import cairo

        from ochra.util.cairo_utils import style_to_cairo, weight_to_cairo
        surface = cairo.SVGSurface(
            tempfile.mktemp(suffix=".svg"),
            len(self.text) * self.font.size * 4,
            self.font.size * 4
        )
        ctx = cairo.Context(surface)
        ctx.set_font_size(self.font.size)
        ctx.select_font_face(self.font.family, style_to_cairo(self.font.style), weight_to_cairo(self.font.weight))
        extents = ctx.text_extents(self.text)
        rect = AxisAlignedRectangle(
            Point(extents.x_bearing + self.bottom_left.x, -extents.y_bearing + self.bottom_left.y - extents.height),
            Point(extents.x_bearing + self.bottom_left.x + extents.width, -extents.y_bearing + self.bottom_left.y)
        ).rotate(self.angle, self.bottom_left)
        return rect

    @property
    def center(self) -> Point:
        bbox = self.bbox
        return bbox.center

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def width(self) -> float:
        return self.bbox.width

    @classmethod
    def centered(cls, text: str, center: PointI, angle: float = 0.0, font: Font = Font()) -> 'Text':
        center = Point.mk(center)
        bbox = cls(text, Point.origin, angle, font).bbox
        return cls(text, center - bbox.center.as_vector(), angle, font)

    @classmethod
    def top_centered(cls, text: str, top_center: PointI, angle: float = 0.0, font: Font = Font()) -> 'Text':
        top_center = Point.mk(top_center)
        bbox = cls(text, Point.origin, angle, font).bbox
        return cls(text, top_center - bbox.top_center.as_vector(), angle, font)

    @classmethod
    def right_centered(cls, text: str, right_center: PointI, angle: float = 0.0, font: Font = Font()) -> 'Text':
        right_center = Point.mk(right_center)
        bbox = cls(text, Point.origin, angle, font).bbox
        return cls(text, right_center - bbox.right_center.as_vector(), angle, font)

    @classmethod
    def bottom_centered(cls, text: str, bottom_center: PointI, angle: float = 0.0, font: Font = Font()) -> 'Text':
        bottom_center = Point.mk(bottom_center)
        bbox = cls(text, Point.origin, angle, font).bbox
        return cls(text, bottom_center - bbox.bottom_center.as_vector(), angle, font)

    def translate(self, dx: float, dy: float) -> 'Text':
        tr = Transformation.translate((dx, dy))
        return Text(self.text, tr(self.bottom_left), self.angle, self.font)

    def axis_aligned_bbox(self) -> AxisAlignedRectangle:
        return self.bbox.axis_aligned_bbox()
