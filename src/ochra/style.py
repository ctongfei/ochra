from collections.abc import Sequence, Mapping, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Tuple, Optional, Union
import cairo

from ochra.util import classproperty
from ochra.functions import lerp
from ochra._matplotlib_colormaps import _viridis_data, _magma_data, _inferno_data, _plasma_data


@dataclass
class Color:
    """
    A color in RGBA space. Each component is in the range [0, 1].
    """

    r: float
    g: float
    b: float
    a: float = 1.0

    @property
    def hex(self) -> str:
        return f"#{int(self.r * 255):02x}{int(self.g * 255):02x}{int(self.b * 255):02x}{int(self.a * 255):02x}"

    @property
    def rgb(self) -> tuple[float, float, float]:
        return self.r, self.g, self.b

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return self.r, self.g, self.b, self.a

    @classmethod
    def from_hex(cls, hex: str, alpha: float = 1.0) -> "Color":
        hex = hex.lstrip("#")
        return cls(r=int(hex[0:2], 16) / 255, g=int(hex[2:4], 16) / 255, b=int(hex[4:6], 16) / 255, a=alpha)

    @classmethod
    def from_rgb(cls, red: float, green: float, blue: float, alpha: float = 1.0) -> "Color":
        return cls(red, green, blue, alpha)

    @classmethod
    def from_rgb_int(cls, red: int, green: int, blue: int, alpha: int = 255) -> "Color":
        return cls(red / 255, green / 255, blue / 255, alpha / 255)

    @classmethod
    def from_hsl(cls):
        # TODO: hsl and hsla
        pass


class Colormap:
    def __call__(self, t: float) -> Color:
        raise NotImplementedError

    @classproperty
    def viridis(cls) -> "Colormap":
        return InterpolatedColormap._from_matplotlib_data(_viridis_data)

    @classproperty
    def magma(cls) -> "Colormap":
        return InterpolatedColormap._from_matplotlib_data(_magma_data)

    @classproperty
    def inferno(cls) -> "Colormap":
        return InterpolatedColormap._from_matplotlib_data(_inferno_data)

    @classproperty
    def plasma(cls) -> "Colormap":
        return InterpolatedColormap._from_matplotlib_data(_plasma_data)


class InterpolatedColormap(Colormap):
    def __init__(self, data: list[Color]):
        self.data = data

    def __call__(self, t: float) -> Color:
        i = int(t * (len(self.data) - 1))
        if i == len(self.data) - 1:
            return self.data[-1]
        else:
            a = self.data[i]
            b = self.data[i + 1]
            _, s = divmod(t * (len(self.data) - 1), 1.0)
            return Color(r=lerp(a.r, b.r, s), g=lerp(a.g, b.g, s), b=lerp(a.b, b.b, s), a=lerp(a.a, b.a, s))

    @classmethod
    def _from_matplotlib_data(cls, data: list[tuple[float, float, float]]) -> Colormap:
        return cls([Color.from_rgb(*rgb) for rgb in data])


class LineCap(Enum):
    butt = "butt"
    round = "round"
    square = "square"

    @classmethod
    def from_str(cls, s: str):
        return cls[s]


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
    array: list[float]
    offset: float | None = None


@dataclass
class Stroke:
    color: Optional[Color] = field(default_factory=lambda: Color(0, 0, 0, 1))  # black
    dash: Optional[Dash] = None
    line_cap: Optional[LineCap] = None
    line_join: Optional[LineJoin] = None
    miter_limit: Optional[float] = None
    opacity: Optional[float] = None
    width: Optional[float] = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


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


class FontStyle(Enum):
    normal = "normal"
    italic = "italic"
    oblique = "oblique"

    def __str__(self):
        return self.value

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
class FontExtents:
    x_height: float
    ascent: float
    descent: float
    height: float
    max_x_advance: float
    max_y_advance: float


@dataclass
class TextExtents:
    x_bearing: float
    y_bearing: float
    width: float
    height: float
    x_advance: float
    y_advance: float


@dataclass
class Font:
    family: str = "sans-serif"
    size: float = 9.0
    size_adjust: Optional[float] = 0.0
    stretch: Optional[str] = None
    style: Optional[FontStyle] = None
    variant: Optional[str] = None  # TODO: full CSS support
    weight: Optional[FontWeight] = None
    extents: FontExtents | None = None

    def __post_init__(self):
        self.extents = _font_extents(self)

    def scale(self, s: float):
        return Font(self.family, self.size * s, self.size_adjust, self.stretch, self.style, self.variant, self.weight)

    def bolded(self):
        return Font(self.family, self.size, self.size_adjust, self.stretch, self.style, self.variant, FontWeight.bold())


def _style_to_cairo(style: Optional[FontStyle]):
    if style == FontStyle.italic:
        return cairo.FONT_SLANT_ITALIC
    if style == FontStyle.oblique:
        return cairo.FONT_SLANT_OBLIQUE
    else:
        return cairo.FONT_SLANT_NORMAL


def _weight_to_cairo(weight: Optional[FontWeight]):
    if weight is None:
        return cairo.FONT_WEIGHT_NORMAL
    if weight.weight >= 700:
        return cairo.FONT_WEIGHT_BOLD
    return cairo.FONT_WEIGHT_NORMAL


def _text_extents(text: str, font: Font) -> TextExtents:
    ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0))
    # TODO: toy font face. Use PyGObject/Pango for real font handling.
    ctx.select_font_face(font.family, _style_to_cairo(font.style), _weight_to_cairo(font.weight))
    ctx.set_font_size(font.size)
    return TextExtents(*ctx.text_extents(text))


def _font_extents(font: Font) -> FontExtents:
    ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0))
    ctx.select_font_face(font.family, _style_to_cairo(font.style), _weight_to_cairo(font.weight))
    ctx.set_font_size(font.size)
    x_height = ctx.text_extents("x").height
    return FontExtents(x_height, *ctx.font_extents())


def _traverse_palette(
    prefix: str, d: Union[Sequence[Union[str, "Palette"]], Mapping[str, Color]]
) -> Iterator[Tuple[str, Color]]:
    if isinstance(d, Mapping):
        for key, value in d.items():
            yield f"{prefix}.{key}", value
    else:
        for i, value in enumerate(d):
            if isinstance(value, Color):
                yield f"{prefix}.{i}", value
            else:
                yield from _traverse_palette(f"{prefix}.{value.name}", value.colors)


class Palette:
    def __init__(
        self,
        name: str,
        colors: Union[Sequence[Union[Color, "Palette"]], Mapping[str, Color]],
        default_light: str | None = None,
        default_dark: str | None = None,
        default_gray: str | None = None,
        color_wheel: Sequence[str] | None = None,
    ):
        self.colors = colors
        self.name = name
        self.colors_dict = {name: color for name, color in _traverse_palette("", colors)}
        self.colors_list = [color for name, color in _traverse_palette("", colors)]
        self.default_light = default_light
        self.default_dark = default_dark
        self.default_gray = default_gray
        self.color_wheel = color_wheel

    def __getitem__(self, key: Union[str, int]) -> Color:
        if isinstance(key, int):
            return self.colors_list[key]
        return self.colors_dict[key]

    def __getattr__(self, item: str) -> Color:
        if isinstance(self.colors, Mapping):
            return self.colors[item]
        else:
            for color in self.colors:
                if color.name == item:
                    return color
        raise AttributeError(f"Palette has no attribute {item}")

    @property
    def default_light_color(self) -> Color:
        return self.colors_dict[self.default_light]

    @property
    def default_dark_color(self) -> Color:
        return self.colors_dict[self.default_dark]

    @property
    def default_gray_color(self) -> Color:
        return self.colors_dict[self.default_gray]

    def default_color_sequence(self, n: int) -> Sequence[Color]:
        assert n <= len(self.color_wheel)
        return [self.colors_dict[self.color_wheel[int(i / n * len(self.color_wheel))]] for i in range(n)]
