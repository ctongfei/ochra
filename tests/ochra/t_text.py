import math

import ochra as ox
from ochra.svg import save_svg
from ochra.style import Palette

font = ox.style.Font('Linux Libertine', size=60)
blue = ox.style.Stroke(Palette.ios.b)
green = ox.style.Stroke(Palette.ios.g)
red = ox.style.Stroke(Palette.ios.r)

t0 = ox.Text("Abg", (0, 70), font=font)
t1 = ox.Text("\"", (0, 0), font=font)

def text_anchors(t: ox.Text):
    return ox.Group([
        t,
        ox.Circle(1, t.visual_center(), stroke=red),
        t.visual_bbox().with_stroke(blue),
        t.aabb().with_stroke(green),
    ])


c = ox.Canvas([text_anchors(t0), text_anchors(t1)])

save_svg(c, "test.svg", horizontal_padding=20, vertical_padding=20)

