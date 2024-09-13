from dataclasses import dataclass
from typing import Tuple


@dataclass
class Color:
    red: float
    green: float
    blue: float
    alpha: float = 1.0

    @property
    def hex(self) -> str:
        return f"#{int(self.red * 255):02x}{int(self.green * 255):02x}{int(self.blue * 255):02x}{int(self.alpha * 255):02x}"

    @property
    def rgb(self) -> Tuple[float, float, float]:
        return self.red, self.green, self.blue

    @property
    def rgba(self) -> Tuple[float, float, float, float]:
        return self.red, self.green, self.blue, self.alpha

    @classmethod
    def from_hex(cls, hex: str, alpha: float = 1.0) -> 'Color':
        hex = hex.lstrip("#")
        return cls(
            red=int(hex[0:2], 16) / 255,
            green=int(hex[2:4], 16) / 255,
            blue=int(hex[4:6], 16) / 255,
            alpha=alpha
        )

    @classmethod
    def from_rgb(cls, red: float, green: float, blue: float, alpha: float = 1.0) -> 'Color':
        return cls(red, green, blue, alpha)

    @classmethod
    def from_rgb_int(cls, red: int, green: int, blue: int, alpha: int = 255) -> 'Color':
        return cls(red / 255, green / 255, blue / 255, alpha / 255)

    # TODO: hsl and hsla