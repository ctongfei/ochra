from typing import Collection, Iterator, Callable

from ochra.element import Element
from ochra.plane import Transformation, PointI, Point


class Group(Element):

    def __init__(self, elements: Collection[Element]):
        self.elements = elements

    def recursive_children(self) -> Iterator[Element]:
        for e in self.elements:
            if isinstance(e, Group):
                yield from e.recursive_children()
            else:
                yield e

    def transform(self, f: Transformation) -> 'Group':
        new_elements = [e.transform(f) for e in self.elements]
        return Group(new_elements)


class Annotation(Element):
    """
    Annotations are special elements that do not scale or rotate by transformations.
    """

    def __init__(self,
                 anchor: PointI,
                 f: Callable[[Point], Element],
                 ):
        super().__init__()
        self.anchor = Point.mk(anchor)
        self.f = f

    def materialize(self) -> Element:
        return self.f(self.anchor)

    def transform(self, f: Transformation) -> "Annotation":
        return Annotation(f(self.anchor), self.f)
