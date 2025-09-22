
import ochra as ox
from ochra.text import Text
from ochra.svg import save_svg
from ochra.palettes import ios

font = ox.style.Font('Linux Libertine', size=60)
blue = ox.style.Stroke(ios.blue)
green = ox.style.Stroke(ios.green)
red = ox.style.Stroke(ios.red)

t0 = Text("Abg", (0, 70), font=font)
t1 = Text("\"", (0, 0), font=font)

def text_anchors(t: Text):
    return ox.Group([
        t,
        ox.Circle(1, t.visual_center(), stroke=red),
        t.visual_bbox().with_stroke(blue),
        t.aabb().with_stroke(green),
    ])


c = ox.Canvas([text_anchors(t0), text_anchors(t1)])

save_svg(c, "test.svg", horizontal_padding=20, vertical_padding=20)

