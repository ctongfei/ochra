import numpy as np
from vega_datasets import data

import ochra as ox
import ochra.plot as oxp
import ochra.style as oxs
from ochra.style.palette import ios

cars = data.cars()
# filter cars where origin is USA then select only the columns Horsepower and Miles_per_Gallon
series = [
    list(cars[cars.Origin == origin][["Horsepower", "Miles_per_Gallon"]].itertuples(index=False))
    for origin in ["USA", "Europe", "Japan"]
]

markers = [
    ox.Marker.plus_mark(stroke=oxs.Stroke(ios[0])),
    ox.Marker.x_mark(stroke=oxs.Stroke(ios[1])),
    ox.Marker.polygon(3, stroke=oxs.Stroke(ios[2])),
]

plot = oxp.Chart(
    size=(300, 300),
    x_axis=oxp.ContinuousAxis("Horsepower", (0, 240), major_ticks=np.linspace(0, 240, 13)),
    y_axis=oxp.ContinuousAxis("Miles per gallon", (0, 50)),
    plots=[
        oxp.ScatterPlot(data, marker=marker)
        for data, color, marker in zip(series, ios, markers)
    ],
    background=oxs.Fill(ios.gray6),
    grid_stroke=oxs.Stroke(ios.gray5, width=1),
)

c = ox.Canvas([plot])

ox.to_svg_file(c, "test.svg")
