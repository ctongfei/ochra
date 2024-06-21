from enum import Enum
from dataclasses import dataclass
from typing import Optional


class FontStyle(Enum):
    normal = "normal"
    italic = "italic"
    oblique = "oblique"

    @classmethod
    def from_str(cls, s: str):
        return cls[s]


@dataclass
class FontWeight:
    weight: int = 400

    @classmethod
    def bold(cls):
        return cls(700)


@dataclass
class Font:
    family: str = "sans-serif"
    size: float = 12.0
    size_adjust: Optional[float] = 0.0
    stretch: Optional[str] = None
    style: Optional[FontStyle] = None
    variant: Optional[str] = None  # TODO: full CSS support
    weight: Optional[FontWeight] = None
