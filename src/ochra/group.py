from typing import Collection

from ochra.element import Element
from ochra.plane import Transformation


class Group(Element):

    def __init__(self, elements: Collection[Element]):
        self.elements = elements

    def transform(self, f: Transformation) -> 'Group':
        new_elements = [e.transform(f) for e in self.elements]
        return Group(new_elements)
