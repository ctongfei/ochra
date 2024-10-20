from abc import ABC, abstractmethod
from typing import Generic

from ochra.core import Element
from ochra.plot import Axis
from ochra.plot.collections import X, Y
from ochra.style import Font


class Plot(ABC, Generic[X, Y]):
    @abstractmethod
    def draw(self, x_axis: Axis[X], y_axis: Axis[Y]) -> Element:
        pass

    @abstractmethod
    def legend(self, font: Font) -> list[tuple[Element, Element]]:
        pass
