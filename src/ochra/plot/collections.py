from typing import TypeVar, TypeAlias
from collections.abc import Mapping, Sequence

import numpy as np

X = TypeVar("X")
Y = TypeVar("Y")
Xc = TypeVar("Xc")  # X continuous
Xd = TypeVar("Xd")  # X discrete
Yc = TypeVar("Yc")  # Y continuous
Yd = TypeVar("Yd")  # Y discrete


Mapping1D: TypeAlias = Mapping[Xd, object] | Sequence[object] | np.ndarray
Mapping2D: TypeAlias = Mapping[tuple[Xd, Yd], object] | np.ndarray

Mapping1DInterval: TypeAlias = Mapping[tuple[Xc, Xc], object]
Mapping2DInterval: TypeAlias = Mapping[tuple[tuple[Xc, Xc], tuple[Yc, Yc]], object]


def standardize_1d(data: Mapping1D) -> Mapping[Xd, object]:
    if isinstance(data, np.ndarray):
        assert data.ndim == 1, "Data must be 1D"
        return {x: data[x] for x in range(data.shape[0])}
    if isinstance(data, Sequence):
        return {x: data[x] for x in range(len(data))}
    return data


def standardize_2d(data: Mapping2D) -> Mapping[tuple[Xd, Yd], object]:
    if isinstance(data, np.ndarray):
        assert data.ndim == 2, "Data must be 2D"
        return {
            (x, y): data[x, y]
            for x in range(data.shape[0])
            for y in range(data.shape[1])
        }
    return data
