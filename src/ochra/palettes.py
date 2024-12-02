from ochra.style import Color, Palette


html_basic: Palette = Palette(
    "html_basic",
    {
        "black": Color.from_hex("#000000"),
        "navy": Color.from_hex("#000080"),
        "green": Color.from_hex("#008000"),
        "teal": Color.from_hex("#008080"),
        "maroon": Color.from_hex("#800000"),
        "purple": Color.from_hex("#800080"),
        "olive": Color.from_hex("#808000"),
        "silver": Color.from_hex("#C0C0C0"),
        "gray": Color.from_hex("#808080"),
        "blue": Color.from_hex("#0000FF"),
        "lime": Color.from_hex("#00FF00"),
        "aqua": Color.from_hex("#00FFFF"),
        "red": Color.from_hex("#FF0000"),
        "fuchsia": Color.from_hex("#FF00FF"),
        "yellow": Color.from_hex("#FFFF00"),
        "white": Color.from_hex("#FFFFFF"),
    },
    default_light=".white",
    default_dark=".black",
    default_gray=".gray",
    color_wheel=[
        ".red",
        ".yellow",
        ".lime",
        ".blue",
        ".navy",
    ]
)


nord: Palette = Palette(
    "nord",
    [
        Palette(
            "polar_night",
            [
                Color.from_hex("#2E3440"),
                Color.from_hex("#3B4252"),
                Color.from_hex("#434C5E"),
                Color.from_hex("#4C566A"),
            ],
        ),
        Palette(
            "snow_storm",
            [
                Color.from_hex("#D8DEE9"),
                Color.from_hex("#E5E9F0"),
                Color.from_hex("#ECEFF4"),
            ],
        ),
        Palette(
            "frost",
            [
                Color.from_hex("#8FBCBB"),
                Color.from_hex("#88C0D0"),
                Color.from_hex("#81A1C1"),
                Color.from_hex("#5E81AC"),
            ],
        ),
        Palette(
            "aurora",
            [
                Color.from_hex("#BF616A"),
                Color.from_hex("#D08770"),
                Color.from_hex("#EBCB8B"),
                Color.from_hex("#A3BE8C"),
                Color.from_hex("#B48EAD"),
            ],
        ),
    ],
    default_light=".snow_storm.2",
    default_dark=".polar_night.0",
    default_gray=".snow_storm.0",
    color_wheel=[
        ".aurora.0",
        ".aurora.1",
        ".aurora.2",
        ".aurora.3",
        ".frost.0",
        ".frost.1",
        ".frost.2",
        ".frost.3",
        ".aurora.4",
    ],
)


solarized: Palette = Palette(
    "solarized",
    {
        "base03": Color.from_hex("#002b36"),
        "base02": Color.from_hex("#073642"),
        "base01": Color.from_hex("#586e75"),
        "base00": Color.from_hex("#657b83"),
        "base0": Color.from_hex("#839496"),
        "base1": Color.from_hex("#93a1a1"),
        "base2": Color.from_hex("#eee8d5"),
        "base3": Color.from_hex("#fdf6e3"),
        "yellow": Color.from_hex("#b58900"),
        "orange": Color.from_hex("#cb4b16"),
        "red": Color.from_hex("#dc322f"),
        "magenta": Color.from_hex("#d33682"),
        "violet": Color.from_hex("#6c71c4"),
        "blue": Color.from_hex("#268bd2"),
        "cyan": Color.from_hex("#2aa198"),
        "green": Color.from_hex("#859900"),
    },
    default_light=".base3",
    default_dark=".base02",
    default_gray=".base2",
    color_wheel=[
        ".red",
        ".orange",
        ".yellow",
        ".green",
        ".cyan",
        ".blue",
        ".violet",
        ".magenta",
    ],
)

# From https://developer.apple.com/design/human-interface-guidelines/color#iOS-iPadOS-system-colors.
ios: Palette = Palette(
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
        "lightgray2": Color.from_rgb_int(174, 174, 178),
        "lightgray3": Color.from_rgb_int(199, 199, 204),
        "lightgray4": Color.from_rgb_int(209, 209, 214),
        "lightgray5": Color.from_rgb_int(229, 229, 234),
        "lightgray6": Color.from_rgb_int(242, 242, 247),
        "darkgray2": Color.from_rgb_int(99, 99, 102),
        "darkgray3": Color.from_rgb_int(72, 72, 74),
        "darkgray4": Color.from_rgb_int(58, 58, 60),
        "darkgray5": Color.from_rgb_int(44, 44, 46),
        "darkgray6": Color.from_rgb_int(28, 28, 30),
    },
    default_light=".lightgray6",
    default_dark=".darkgray6",
    default_gray=".lightgray5",
    color_wheel=[
        ".red",
        ".orange",
        ".yellow",
        ".green",
        ".mint",
        ".teal",
        ".cyan",
        ".blue",
        ".indigo",
        ".purple",
    ],
)
