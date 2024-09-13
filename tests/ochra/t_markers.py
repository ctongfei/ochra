import math

import ochra as ox

canvas = ox.Canvas(
    elements=[
        ox.Mark((5, 10), ox.Marker.plus_mark()),
        ox.Mark((10, 10), ox.Marker.x_mark()),
        ox.Mark((15, 10), ox.Marker.circle()),
        ox.Mark((20, 10), ox.Marker.polygon(3)),
        ox.Mark((25, 10), ox.Marker.polygon(4)),
        ox.Mark((30, 10), ox.Marker.polygon(4, angle=math.tau/8)),
        ox.Mark((35, 10), ox.Marker.polygon(6)),
    ],
    viewport=(ox.AxisAlignedRectangle((0, 0), (100, 30)))
)

ox.to_svg_file(canvas, "marker.svg")