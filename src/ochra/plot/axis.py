from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, Tuple

import numpy as np

from ochra.plot.typedef import X
from ochra.style.font import Font
from ochra.style.stroke import Stroke


class Axis(ABC, Generic[X]):

    def __init__(self,
                 label: str,
                 bounds: Tuple[X, X],
                 locate_fn: Callable[[X], float],
                 major_ticks: Sequence[X],
                 minor_ticks: Sequence[X],
                 stroke: Stroke = Stroke(),
                 tick_stroke: Stroke = Stroke(),
                 text_style: Font = Font(),
                 text_padding: float = 2,
                 major_tick_length: float = 5,
                 minor_tick_length: float = 2
                 ):
        self.label = label
        self.bounds = bounds
        self.locate_fn = locate_fn
        self.major_ticks = major_ticks
        self.minor_ticks = minor_ticks
        self.stroke = stroke
        self.tick_stroke = tick_stroke
        self.text_style = text_style
        self.text_padding = text_padding
        self.major_tick_length = major_tick_length
        self.minor_tick_length = minor_tick_length

    @property
    def lower_bound(self) -> X:
        return self.bounds[0]

    @property
    def upper_bound(self) -> X:
        return self.bounds[1]

    @abstractmethod
    def __contains__(self, x: X):
        """Returns if this value should be drawn on the axis."""
        pass

    def locate(self, x: X) -> float:
        """Locates a value on the axis."""
        return self.locate_fn(x)

    def locate_lower_bound(self) -> float:
        return self.locate(self.lower_bound)

    def locate_upper_bound(self) -> float:
        return self.locate(self.upper_bound)


class ContinuousAxis(Axis[float]):

    def __init__(self,
                 label: str,
                 bounds: Tuple[float, float],
                 locate_fn: Callable[[float], float] = lambda x: x,
                 major_ticks: Sequence[float] = None,
                 minor_ticks: Sequence[float] = None,
                 stroke: Stroke = Stroke(),
                 tick_stroke: Stroke = Stroke(),
                 text_style: Font = Font(),
                 text_padding: float = 2,
                 major_tick_length: float = 4,
                 minor_tick_length: float = 2,
                 ):
        super().__init__(label, bounds, locate_fn, major_ticks, minor_ticks, stroke, tick_stroke, text_style, text_padding,
                         major_tick_length, minor_tick_length)

        if self.major_ticks is None:
            self.major_ticks = np.linspace(self.bounds[0], self.bounds[1], 11)

    def __contains__(self, x: X):
        return self.bounds[0] <= x <= self.bounds[1]


class CategoricalAxis(Axis[X]):

    def __init__(self,
                 label: str,
                 categories: Sequence[X],
                 **kwargs
                 ):
        self.categories = categories
        self.locate_dict = {category: i for i, category in enumerate(self.categories)}
        super().__init__(
            label,
            (categories[0], categories[-1]),
            lambda x: self.locate_dict[x],
            major_ticks=categories,
            minor_ticks=[],
            **kwargs
        )

    def locate_lower_bound(self) -> float:
        return self.locate(self.lower_bound) - 0.5

    def locate_upper_bound(self) -> float:
        return self.locate(self.upper_bound) + 0.5

    def __contains__(self, x: X):
        return x in self.locate_dict


# Date/time axis