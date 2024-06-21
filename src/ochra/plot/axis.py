from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ochra.style.stroke import Stroke


@dataclass
class Axis:
    label: str
    lower_bound: float
    upper_bound: float
    major_ticks: Sequence[float] = None
    minor_ticks: Sequence[float] = None
    locate: Callable[[float], float] = lambda x: x
    scale: float = 1.0
    stroke: Stroke = field(default_factory=Stroke)
    tick_stroke: Stroke = field(default_factory=Stroke)
    text_style: dict = field(default_factory=dict)
    text_padding: int = 3
    major_tick_length: int = 6
    minor_tick_length: int = 3

    def __post_init__(self):
        if self.major_ticks is None:
            self.major_ticks = np.linspace(self.lower_bound, self.upper_bound, 11)
