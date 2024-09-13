from abc import ABC, abstractmethod
from typing import Generic

from ochra.element import Element
from ochra.plot import Axis
from ochra.plot.typedef import X, Y


class Plot(ABC, Generic[X, Y]):
    @abstractmethod
    def draw(self, x_axis: Axis[X], y_axis: Axis[Y]) -> Element:
        pass

