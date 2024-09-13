import ochra as ox
from ochra.util.functions import deg_to_rad

t = ox.Text("hello world", bottom_left=(0, 0), angle=deg_to_rad(45))
p = ox.Polygon.regular(3, circumradius=10).translate(60, 60).rotate(deg_to_rad(45), anchor=(60, 60))

c = ox.Canvas(
    elements=[
        t,
        p,
        t.axis_aligned_bbox(),
        p.axis_aligned_bbox()
    ],
)

ox.to_svg_file(c, "test.svg")