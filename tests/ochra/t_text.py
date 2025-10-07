import ochra as ox
from ochra.text import Text
from ochra.svg import save_svg
from ochra.palettes import ios

font = ox.style.Font("Linux Libertine", size=60)
blue = ox.style.Stroke(ios.blue)
green = ox.style.Stroke(ios.green)
red = ox.style.Stroke(ios.red)

t0 = Text("Abg", (0, 70), font=font)
t1 = Text('"', (0, 0), font=font)


def text_anchors(t: Text):
    return ox.Group(
        [
            t,
            ox.Circle(3, t.bottom_left, styles=[blue]),
            #ox.Circle(2, t.visual_center(), styles=[red]),
            t.visual_bbox().set_style(blue),
            t.aabb().set_style(green),
        ]
    )


c = ox.Canvas([text_anchors(t0), text_anchors(t1)])

save_svg(c, "test.svg", horizontal_padding=20, vertical_padding=20)
