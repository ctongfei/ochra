from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ochra.style.color import Color


@dataclass
class LineCap(Enum):
    butt = "butt"
    round = "round"
    square = "square"

    @classmethod
    def from_str(cls, s: str):
        return cls[s]


@dataclass
class LineJoin(Enum):
    miter = "miter"
    round = "round"
    bevel = "bevel"
    arcs = "arcs"
    miter_clip = "miter-clip"

    @classmethod
    def from_str(cls, s: str):
        return cls[s]


@dataclass
class Dash:
    array: List[float]
    offset: Optional[float] = None


@dataclass
class Stroke:
    color: Optional[Color] = field(default_factory=lambda: Color(0, 0, 0, 1))  # black
    dash: Optional[Dash] = None
    line_cap: Optional[LineCap] = None
    line_join: Optional[LineJoin] = None
    miter_limit: Optional[float] = None
    opacity: Optional[float] = None
    width: Optional[float] = None
    