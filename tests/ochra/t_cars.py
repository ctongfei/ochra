import numpy as np
from dataclasses import replace
from vega_datasets import data

import ochra as ox
import ochra.style as oxs
import ochra.plot as oxp
from ochra.style.palette import ios


cars = data.cars()
# filter cars where origin is USA then select only the columns Horsepower and Miles_per_Gallon
series = [
    list(cars[cars.Origin == origin][["Horsepower", "Miles_per_Gallon"]].itertuples(index=False))
    for origin in ["USA", "Europe", "Japan"]
]

plot = oxp.ChartArea(
    size=(300, 300),
    x_axis=oxp.ContinuousAxis("Horsepower", (0, 240), major_ticks=np.linspace(0, 240, 13)),
    y_axis=oxp.ContinuousAxis("Miles per gallon", (0, 50)),
    plots=[
        oxp.ScatterPlot(data, marker=ox.Marker.circle(stroke=oxs.Stroke(replace(color, alpha=0.6), width=1)))
        for data, color in zip(series, ios)
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
