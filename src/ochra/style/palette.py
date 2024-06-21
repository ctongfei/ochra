from typing import Union, Iterator, Tuple, Sequence, Mapping

from ochra.style.color import Color


def _traverse_palette(
        prefix: str,
        d: Union[Sequence[Union[str, 'Palette']], Mapping[str, Color]]
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
    def __init__(self, name: str, colors: Union[Sequence[Union[Color, 'Palette']], Mapping[str, Color]]):
        self.colors = colors
        self.name = name
        self.colors_dict = {name: color for name, color in _traverse_palette(name, colors)}
        self.colors_list = [color for name, color in _traverse_palette(name, colors)]

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


nord = Palette(
    "nord",
    [
        Palette(
            "polar_night",
            [
                Color.from_hex("#2E3440"),
                Color.from_hex("#3B4252"),
                Color.from_hex("#434C5E"),
                Color.from_hex("#4C566A"),
            ]
        ),
        Palette(
            "snow_storm",
            [
                Color.from_hex("#D8DEE9"),
                Color.from_hex("#E5E9F0"),
                Color.from_hex("#ECEFF4"),
            ]
        ),
        Palette(
            "frost",
            [
                Color.from_hex("#8FBCBB"),
                Color.from_hex("#88C0D0"),
                Color.from_hex("#81A1C1"),
                Color.from_hex("#5E81AC"),
            ]
        ),
        Palette(
            "aurora",
            [
                Color.from_hex("#BF616A"),
                Color.from_hex("#D08770"),
                Color.from_hex("#EBCB8B"),
                Color.from_hex("#A3BE8C"),
                Color.from_hex("#B48EAD"),
            ]
        ),
    ]
)

ios = Palette(
    "ios",
    {
        "red": Color.from_rgb_int(255, 59, 48),
        "orange": Color.from_rgb_int(255, 149, 0),
        "yellow": Color.from_rgb_int(255, 204, 0),
        "green": Color.from_rgb_int(52, 199, 89),
        "mint": Color.from_rgb_int(0, 199, 190),
        "teal": Color.from_rgb_int(48, 176, 199),
        "cyan": Color.from_rgb_int(50, 173, 230),
        "blue": Color.from_rgb_int(0, 122, 255),
        "indigo": Color.from_rgb_int(88, 86, 214),
        "purple": Color.from_rgb_int(175, 82, 222),
        "pink": Color.from_rgb_int(255, 45, 85),
        "brown": Color.from_rgb_int(162, 132, 94),
        "gray": Color.from_rgb_int(142, 142, 147),
        "gray2": Color.from_rgb_int(174, 174, 178),
        "gray3": Color.from_rgb_int(199, 199, 204),
        "gray4": Color.from_rgb_int(209, 209, 214),
        "gray5": Color.from_rgb_int(229, 229, 234),
        "gray6": Color.from_rgb_int(242, 242, 247),
    }
)
