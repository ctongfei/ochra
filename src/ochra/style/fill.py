from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ochra.style.color import Color


class FillRule(Enum):
    nonzero = "nonzero"
    even_odd = "evenodd"

    @classmethod
    def from_str(cls, s: str):
        return cls[s]


@dataclass
class Fill:
    color: Optional[Color] = field(default_factory=lambda: Color(0, 0, 0, 0))  # transparent
    opacity: Optional[float] = None
    rule: Optional[FillRule] = None
