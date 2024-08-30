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
        (5 + 25 * i, 10), (5 + 25 * i, 100),
        stroke=replace(stroke, color=ios[i]),
        marker_end=arrow_end
    )
    for i, arrow_end in enumerate(arrow_ends)
]

rect = ox.AxisAlignedRectangle((0, 0), (400, 400), fill=oxs.Fill(ios.gray4))
r = ox.Rectangle((50, 50), (200, 200), angle=deg_to_rad(70), fill=oxs.Fill(ios[0]))

text = ox.Text.right_centered("Abc", (100, 100), angle=math.tau/6, font=oxs.Font(size=20))
bbox0 = ox.Text("Abc", ox.Point.origin, angle=math.tau/6, font=oxs.Font(size=20)).bbox
text1 = ox.Text("Abc", text.bbox.right_center - bbox0.right_center.as_vector(), angle=math.tau/6, font=oxs.Font(size=20))

canvas = ox.Canvas(
    viewport=rect,
    elements=[
        rect,
        r,
        arrows[0].scale(1, 2.5),
        text.bbox,
        bbox0,
        text1,
        ox.Circle(2, (100, 100), fill=oxs.Fill(ios.blue)),
    ]
)
#    def right_centered(cls, text: str, right_center: PointI, angle: float = 0.0, font: Font = Font()) -> 'Text':
#        right_center = Point.mk(right_center)
#        bbox = Text(text, Point.origin, angle, font).bbox
#        return cls(text, right_center - bbox.right_center.as_vector(), angle, font)

ox.to_svg_file(canvas, "test.svg")
