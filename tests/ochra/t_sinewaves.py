import math

import numpy as np

import ochra as ox
import ochra.plot as oxp
import ochra.style as oxs
from ochra.style.palette import ios

data_series = [
    [
        (x, math.sin(x + i * 0.4) * (16 - i) / 4 + 5)
        for x in np.linspace(0, 10, 128)
    ]
    for i in range(12)
]

plot = oxp.Chart(
    size=(200, 200),
    x_axis=oxp.ContinuousAxis("x", (0, 10)),
    y_axis=oxp.ContinuousAxis("y", (0, 10)),
    plots=[
        oxp.LinePlot("sine", data, stroke=oxs.Stroke(color=color))
        for data, color in zip(data_series, ios)
    ],
    background=oxs.Fill(ios.lightgray6),
    grid_stroke=oxs.Stroke(ios.lightgray5, width=1),
    font=oxs.Font('Monaco'),
    palette=ios
)

c = ox.Canvas(elements=[plot.draw()])

ox.save_svg(c, "test.svg")
