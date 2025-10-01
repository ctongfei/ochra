import ochra as ox
import ochra.palettes as oxp
from ochra.svg import save_svg
from ochra.functions import deg_to_rad

cross = ox.Marker.cross(size=4)
plus = ox.Marker.plus(size=4)
blue = ox.Stroke(width=6, color=oxp.ios.blue)

f0 = (0, 0)
f1 = (-50, 70)
ell = ox.Ellipse.from_foci_and_major_axis(f0, f1, 120, stroke=blue)
ell2 = ell.transform(ox.Rotation(deg_to_rad(30)))

canvas = ox.Canvas(
    [
        ox.Group(
            [
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
                ell.trace_quadratic_bezier_path(step=32, num_steps=16, markers=ox.mark.MarkerConfig(mid=plus)),
            ]
        ),
    ],
    viewport=ox.AxisAlignedRectangle((-200, -200), (200, 200)),
)

save_svg(canvas, "test.svg")
