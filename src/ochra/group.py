from typing import Collection, Iterator

from ochra.element import Element
from ochra.plane import Transformation


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
