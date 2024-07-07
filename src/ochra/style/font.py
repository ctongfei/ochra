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

    def scale(self, s: float):
        return Font(self.family, self.size * s, self.size_adjust, self.stretch, self.style, self.variant, self.weight)

    def bolded(self):
        return Font(self.family, self.size, self.size_adjust, self.stretch, self.style, self.variant, FontWeight.bold())