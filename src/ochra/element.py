from abc import ABC

from ochra.plane import Transformation, Point, Vector, PointI


class Element(ABC):
    """
    Base class for all drawable elements.
    """

    # TODO: add transform_text option
    def transform(self, f: Transformation) -> 'Element':
        """
        Transform this element using the given transformation.
        :param f: The given transform.
        :return:
        """
        return AnyTransformed(self, f)  # fallback

    def translate(self, dx: float, dy: float) -> 'Element':
        return self.transform(Transformation.translate(Vector(dx, dy)))

    def rotate(self, angle: float, anchor: PointI = Point.origin) -> 'Element':
        return self.transform(Transformation.rotate(angle, Point.mk(anchor)))

    def scale(self, sx: float, sy: float) -> 'Element':
        return self.transform(Transformation.scale(Vector(sx, sy)))


class AnyTransformed(Element):
    def __init__(self, element: Element, transformation: Transformation):
        self.element = element
        self.transformation = transformation

    def transform(self, f: Transformation) -> 'AnyTransformed':
        return AnyTransformed(self.element, f @ self.transformation)

