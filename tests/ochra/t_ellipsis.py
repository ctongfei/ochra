import ochra as ox
import ochra.style as oxs
from ochra.svg import save_svg

cross = ox.Marker.x_mark(size=4)
plus = ox.Marker.plus_mark(size=4)
blue = ox.Stroke(width=8, color=oxs.Palette.solarized.blue)

f0 = (0, 0)
f1 = (-50, 50)
ell = ox.Ellipse.from_foci_and_major_axis(f0, f1, 120, stroke=blue)

canvas = ox.Canvas(
    [
        ox.Group([
            ell,
            ox.Mark((0, 0), plus),
            ox.Mark(f0, plus),
            ox.Mark(f1, plus),
            ox.Mark(ell.focus0, cross),
            ox.Mark(ell.focus1, cross),
            ell.circumscribed_rectangle(),
            ox.LineSegment(ell.vertex0, ell.vertex1),
            ox.LineSegment(ell.covertex0, ell.covertex1),
            ell.directrix0(),
            ell.directrix1(),
            ell.trace_quadratic_bezier_path(step=32, num_steps=16, markers=ox.MarkerConfig(mid=plus))
        ]),
    ],
    viewport=ox.AxisAlignedRectangle((-200, -200), (200, 200))
)

save_svg(canvas, "test.svg")
