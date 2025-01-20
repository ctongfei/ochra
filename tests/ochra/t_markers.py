import math

import ochra as ox

canvas = ox.Canvas(
    elements=[
        ox.Mark((5, 10), ox.Marker.bullet()),
        ox.Mark((10, 10), ox.Marker.circle()),
        *[
            ox.Mark((i * 5, 10), ox.Marker.polygon(i))
            for i in range(3, 10)
        ],
        *[
            ox.Mark((i * 5, 15), ox.Marker.asterisk(i))
            for i in range(1, 10)
        ],
    ],
    viewport=(ox.AxisAlignedRectangle((0, 0), (100, 30)))
)

ox.svg.save_svg(canvas, "marker.svg")