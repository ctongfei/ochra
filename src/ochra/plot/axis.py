from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple, Generic

import numpy as np

from ochra.style.font import Font
from ochra.style.stroke import Stroke
from ochra.plot.typedef import X


@dataclass
class Axis(Generic[X]):
    label: str
    locate: Callable[[X], float]
    major_ticks: Sequence[X] = None


@dataclass
class ContinuousAxis:
    label: str
    bounds: Tuple[float, float]
    locate: Callable[[float], float] = lambda x: x
    major_ticks: Sequence[float] = None
    minor_ticks: Sequence[float] = None
    stroke: Stroke = field(default_factory=Stroke)
    tick_stroke: Stroke = field(default_factory=Stroke)
    text_style: Font = field(default_factory=Font)
    text_padding: int = 3
    major_tick_length: int = 6
    minor_tick_length: int = 3

    def __post_init__(self):
        if self.major_ticks is None:
            self.major_ticks = np.linspace(self.lower_bound, self.upper_bound, 11)

    @property
    def lower_bound(self):
        return self.bounds[0]

    @property
    def upper_bound(self):
        return self.bounds[1]


@dataclass
class CategoricalAxis(Generic[X]):
    label: str
    categories: Sequence[X]