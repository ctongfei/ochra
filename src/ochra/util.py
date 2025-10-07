import dataclasses
from collections.abc import Sequence, Callable
from typing import Any, Self, overload, Protocol

import jax


class Global:
    """
    Global settings for drawing.
    """

    approx_eps: float = 2**-16
    """The epsilon for approximating floating point numbers as equal."""

    boundary_eps: float = 2**-16
    """The epsilon for avoiding boundary issues."""

    first_order_step_size: float = 2.0
    """The step size for tracing polylines."""

    second_order_step_size: float = 8.0
    """The step size for tracing quadratic Bézier paths."""

    num_first_order_steps: int = 256
    """The number of steps for tracing polylines."""

    num_second_order_steps: int = 64
    """The number of steps for tracing quadratic Bézier paths."""

    num_hermite_steps: int = 64
    """The number of steps for tracing Hermite splines."""


class classproperty:
    """
    Decorator that converts a method with a single cls argument into a property
    that can be accessed directly from the class.
    From https://docs.djangoproject.com/en/5.0/_modules/django/utils/functional/#classproperty.
    """

    def __init__(self, method=None):
        self.fget = method

    def __get__(self, instance, cls=None):
        return self.fget(cls)

    def getter(self, method):
        self.fget = method
        return self


class IndexedSequence[T](Sequence[T]):
    def __init__(self, get: Callable[[int], T], length: int):
        self._get = get
        self._length = length

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> T | Self:
        if isinstance(index, slice):
            return [self._get(i) for i in range(*index.indices(self._length))]
        return self._get(index)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i in range(self._length):
            yield self._get(i)


class Comparable[T](Protocol):
    def __lt__(self, other: T) -> bool: ...


def f2s(x: Any) -> str:
    if isinstance(x, jax.Array):
        return f2s(x.item())
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    else:
        return str(x)
