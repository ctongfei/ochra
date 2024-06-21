import math

import numpy as np

import ochra as ox
import ochra.style as oxs
import ochra.plot as oxp
from ochra.style import Stroke
from ochra.style.palette import ios


data_series = [
    [
        (x, math.sin(x + i * 0.3) * (16 - i) / 4 + 5)
        for x in np.linspace(0, 10, 128)
    ]
    for i in range(16)
]

plot = oxp.ChartArea(
    x_axis=oxp.Axis("x", 0, 10, scale=30),
    y_axis=oxp.Axis("y", 0, 10, scale=30),
    plots=[
        oxp.LinePlot(data, stroke=Stroke(color, width=2, line_cap=oxs.LineCap.round))
        for data, color in zip(data_series, ios)
    ],
    background=oxs.Fill(ios.gray6),
    grid_stroke=oxs.Stroke(ios.gray5, width=1),
)

text = ox.Text(
            "Hello, World!",
            left_bottom=(50, 50),
        )

c = ox.Canvas(
    viewport=ox.AxisAlignedRectangle(ox.Point.origin, (400, 400)),
    elements=[
        ox.EmbeddedCanvas(plot.plot(), left_bottom=(50, 50)),
        text
    ]
)

ox.to_svg_file(c, "test.svg")
