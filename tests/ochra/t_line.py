import ochra as ox
import ochra.svg
from ochra.palettes import ios
from ochra.functions import deg_to_rad


marker = ox.Marker.bullet(size=1)
line = ox.Line.from_two_points((20, 0), (0, 20))
ellipse = ox.Ellipse.from_foci_and_major_axis((0, 0), (20, 10), 40)
canvas = ox.Canvas(
    [
        ox.Group(
            [
                ox.Mark((0, 0), marker),
                line,
                ellipse,
                ox.Mark(ellipse.focus0, marker),
                ox.Mark(ellipse.focus1, marker),
                ox.Group([ox.Mark(line.at(t), marker) for t in [0.1, 0.5, 0.9]]),
            ]
        ),
        ox.Ray((0, 0), deg_to_rad(45), styles=[ox.Stroke(color=ios.red, width=2)]),
    ],
    viewport=ox.AxisAlignedRectangle((-50, -50), (50, 50)),
)
ochra.svg.save_svg(canvas, "test.svg", horizontal_padding=10, vertical_padding=10)
