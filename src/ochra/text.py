import tempfile

from ochra.element import Element
from ochra.plane import Point, Transformation, PointI
from ochra.rect import AxisAlignedRectangle
from ochra.style.font import Font


class Text(Element):

    def __init__(self, text: str, left_bottom: PointI, font: Font = Font()):
        self.text = text
        self.left_bottom = Point.mk(left_bottom)
        self.font = font

    @property
    def bounding_box(self) -> 'AxisAlignedRectangle':
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
        return AxisAlignedRectangle(
            Point(extents.x_bearing + self.left_bottom.x, -extents.y_bearing + self.left_bottom.y - extents.height),
            Point(extents.x_bearing + self.left_bottom.x + extents.width, -extents.y_bearing + self.left_bottom.y)
        )

    @property
    def center(self) -> Point:
        bbox = self.bounding_box
        return bbox.center

    @classmethod
    def centered(cls, text: str, center: Point, font: Font = Font()) -> 'Text':
        bbox = Text(text, Point.origin, font).bounding_box
        return cls(text, center - bbox.center.as_vector(), font)

    @classmethod
    def top_centered(cls, text: str, top_center: Point, font: Font = Font()) -> 'Text':
        bbox = Text(text, Point.origin, font).bounding_box
        return cls(text, top_center - bbox.top_center.as_vector(), font)

    @classmethod
    def right_centered(cls, text: str, right_center: Point, font: Font = Font()) -> 'Text':
        bbox = Text(text, Point.origin, font).bounding_box
        return cls(text, right_center - bbox.right_center.as_vector(), font)

    def transform(self, f: Transformation) -> 'Element':
        return Text(self.text, f(self.left_bottom), self.font)
        # TODO: properly transform the text

