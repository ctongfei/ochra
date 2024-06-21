from typing import Optional
import cairo
from ochra.style.font import Font, FontStyle, FontWeight


def style_to_cairo(style: Optional[FontStyle]):
    if style is None:
        return cairo.FONT_SLANT_NORMAL
    if style == FontStyle.italic:
        return cairo.FONT_SLANT_ITALIC
    if style == FontStyle.oblique:
        return cairo.FONT_SLANT_OBLIQUE


def weight_to_cairo(weight: Optional[FontWeight]):
    if weight is None:
        return cairo.FONT_WEIGHT_NORMAL
    if weight.weight >= 700:
        return cairo.FONT_WEIGHT_BOLD
    return cairo.FONT_WEIGHT_NORMAL
