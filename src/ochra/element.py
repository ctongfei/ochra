from abc import ABC
from typing import Optional, TYPE_CHECKING

from ochra.plane import LineI, Point, PointI, Transformation, Vector

if TYPE_CHECKING:
    from ochra.rect import AxisAlignedRectangle


class Element(ABC):
    """
    Base class for all drawable elements.
    """

    def transform(self, f: Transformation) -> 'Element':
        """
        Transforms this element using the given transformation.
        Should be overridden by subclasses if possible.
        :param f: The given transformation.
        :return: A new element where every point is transformed.
        """
        return AnyTransformed(self, f)  # fallback

    def axis_aligned_bbox(self) -> 'Optional[AxisAlignedRectangle]':
        raise NotImplementedError

    def translate(self, dx: float, dy: float) -> 'Element':
        return self.transform(Transformation.translate(Vector(dx, dy)))

    def rotate(self, angle: float, anchor: PointI = Point.origin) -> 'Element':
        return self.transform(Transformation.rotate(angle, Point.mk(anchor)))

    def scale(self, sx: float, sy: float) -> 'Element':
        return self.transform(Transformation.scale(Vector(sx, sy)))

    def reflect(self, axis: LineI) -> 'Element':
        return self.transform(Transformation.reflect(axis))


class AnyTransformed(Element):
    def __init__(self, element: Element, transformation: Transformation):
        self.element = element
        self.transformation = transformation

    def transform(self, f: Transformation) -> 'AnyTransformed':
        return AnyTransformed(self.element, f @ self.transformation)

