import math

import numpy as np

import ochra as ox
import ochra.style as oxs
import ochra.plot as oxp
from ochra.style import Stroke
from ochra.style.palette import ios


data_series = [
    [
        (x, math.sin(x + i * 0.4) * (16 - i) / 4 + 5)
        for x in np.linspace(0, 10, 128)
    ]
    for i in range(12)
]

plot = oxp.ChartArea(
    x_axis=oxp.ContinuousAxis("x", (0, 10), scale=30),
    y_axis=oxp.ContinuousAxis("y", (0, 10), scale=30),
    plots=[
        oxp.ScatterPlot(data, marker=ox.Marker.circle(stroke=oxs.Stroke(color, width=1)))
        for data, color in zip(data_series, ios)
    ],
    background=oxs.Fill(ios.gray6),
    grid_stroke=oxs.Stroke(ios.gray5, width=1),
)

c = ox.Canvas(
    viewport=ox.AxisAlignedRectangle((0, 0), (400, 400)),
    elements=[
        ox.EmbeddedCanvas(plot, left_bottom=(50, 50)),
    ]
)

ox.to_svg_file(c, "test.svg")
