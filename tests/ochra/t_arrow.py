import math
from dataclasses import replace
import ochra as ox
import ochra.style as oxs
from ochra.util.functions import deg_to_rad
from ochra.style.palette import ios
from ochra.style.arrowhead import arrow_triangle, arrow_stealth, arrow_line

stroke = oxs.Stroke(ios.blue, width=2, line_join=oxs.LineJoin.miter)

arrow_ends = [
    cont(size=5, angle=deg_to_rad(deg), stroke=replace(stroke, color=ios[i * 3 + j]))
    for i, deg in enumerate([30])
    for j, cont in enumerate([arrow_stealth, arrow_triangle, arrow_line])
]

arrows = [
    ox.LineSegment(
        (25 + 25 * i, 100), (25 + 25 * i, 200),
        stroke=replace(stroke, color=ios[i]),
        marker_end=arrow_end
    )
    for i, arrow_end in enumerate(arrow_ends)
]

rect = ox.AxisAlignedRectangle((0, 0), (400, 400), fill=oxs.Fill(ios.gray6))
m = ox.Mark(arrows[0].p1, marker=ox.Marker.circle())
m2 = ox.Mark(arrows[1].p1, marker=ox.Marker.polygon(4, angle=math.radians(0)))


canvas = ox.Canvas(
    viewport=rect,
    elements=[rect, *arrows, m, m2]
)

ox.to_svg_file(canvas, "test.svg")
