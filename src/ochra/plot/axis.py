from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Literal
from collections.abc import Callable, Sequence

import numpy as np

from ochra.plot.collections import X
from ochra.util import Comparable


class AxisKind(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class Axis[X](ABC):
    def __init__(
        self,
        label: str,
        bounds: tuple[X, X],
        locate_fn: Callable[[X], float],
        major_ticks: Sequence[X],
        minor_ticks: Sequence[X],
    ):
        self.label = label
        self.bounds = bounds
        self.locate_fn = locate_fn
        self.major_ticks = major_ticks
        self.minor_ticks = minor_ticks

    @property
    @abstractmethod
    def kind(self) -> AxisKind:
        raise NotImplementedError

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
    def __init__(
        self,
        label: str,
        bounds: tuple[float, float],
        locate_fn: Callable[[float], float] = lambda x: x,
        major_ticks: Sequence[float] = None,
        minor_ticks: Sequence[float] = None,
    ):
        super().__init__(label, bounds, locate_fn, major_ticks, minor_ticks)

        if self.major_ticks is None:
            self.major_ticks = np.linspace(self.bounds[0], self.bounds[1], 11)

    def __contains__(self, x: X):
        return self.bounds[0] <= x <= self.bounds[1]

    @property
    def kind(self) -> AxisKind:
        return AxisKind.CONTINUOUS


class DiscreteAxis(Axis[X]):
    def __init__(
        self,
        label: str,
        categories: Sequence[X],
    ):
        self.categories = categories
        self.locate_dict = {category: i for i, category in enumerate(self.categories)}
        super().__init__(
            label,
            (categories[0], categories[-1]),
            lambda x: self.locate_dict[x],
            major_ticks=categories,
            minor_ticks=[],
        )

    def locate_lower_bound(self) -> float:
        return self.locate(self.lower_bound) - 0.5

    def locate_upper_bound(self) -> float:
        return self.locate(self.upper_bound) + 0.5

    def __contains__(self, x: X):
        return x in self.locate_dict

    @property
    def kind(self) -> AxisKind:
        return AxisKind.DISCRETE


# Date/time axis
